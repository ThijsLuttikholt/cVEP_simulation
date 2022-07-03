import numpy as np

from copy import deepcopy

from sklearn.cross_decomposition import CCA
from tqdm import tqdm

import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from .extra_functions import splice_data

from .EEGNet import EEGNet


#This file will contain all functions, including the train functions, that I will be using
#in the final evaluations part. 

######################################
# First, the training and testing functions for the EEGNet itself
######################################

#The network training function
def train_net_final(
    net, # the network being trained
    train_dataloader, # the training data
    valid_dataloader, # the validation data
    lr, # the learning rate
    max_epochs=100, # maximum number of pass through the whole training dataset
    max_epochs_without_improvement = 10, # parameter used by the early stopping mechanism
):
    #optimizer = torch.optim.SGD(params=net.parameters(), lr=lr)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    scores = []
    best_valid_accuracy = None
    best_net_params = None
    best_epoch = -1
    valid_accs = np.zeros(max_epochs)
    for epoch in tqdm(range(max_epochs), leave=False, desc='epochs'): 
        # training for one full epoch :
        accuracy_accumulator = 0
        count = 0
        loss_accumulator = 0.
        for batch in train_dataloader:
            X,y = batch
            optimizer.zero_grad() 
            y_pred = net(X)
            loss = loss_func(y_pred, y)
            loss.backward()
            optimizer.step()
            accuracy_accumulator += torch.sum(y==y_pred.argmax(dim=1)).detach().item()
            loss_accumulator += loss.detach().item()
            count += y.shape[0]
        scores.append(dict(epoch=epoch, phase='train', metric='accuracy', value=accuracy_accumulator/count))
        scores.append(dict(epoch=epoch, phase='train', metric='loss',     value=loss_accumulator/count))
        
        # validation step :
        accuracy_accumulator = 0
        count = 0
        loss_accumulator = 0.
        with torch.no_grad():
            for batch in valid_dataloader:
                X,y = batch
                y_pred = net(X)
                loss = loss_func(y_pred, y)
                accuracy_accumulator += torch.sum(y==y_pred.argmax(dim=1)).detach().item()
                loss_accumulator += loss.detach().item()
                count += y.shape[0]
        last_valid_accuracy = accuracy_accumulator/count
        valid_accs[epoch] = last_valid_accuracy
        scores.append(dict(epoch=epoch, phase='valid', metric='accuracy', value=accuracy_accumulator/count))
        scores.append(dict(epoch=epoch, phase='valid', metric='loss',     value=loss_accumulator/count))
        
        # early stopping step :
        update_condition = (best_valid_accuracy==None or last_valid_accuracy>best_valid_accuracy)
        if update_condition:
            best_valid_accuracy = last_valid_accuracy
            best_epoch = epoch
            best_net_params = deepcopy(net.state_dict())
        stop_condition = epoch-best_epoch>max_epochs_without_improvement
        if stop_condition:
            break

    # test step :

    net.load_state_dict(best_net_params)
    
    # prepare scores DataFrame :
    scores_df = pd.DataFrame(scores)
    scores_df['F1'] = net.F1
    scores_df['D'] = net.D
    scores_df['F2'] = net.F2
    scores_df['drop_prob'] = net.drop_prob
    scores_df['lr'] = lr
    scores_df['max_epochs'] = max_epochs
    scores_df['max_epochs_without_improvement'] = max_epochs_without_improvement

    return scores_df, net

#The network testing function
def test_net_final(
    net, # the network being trained
    test__dataloader, # the test data
    ):

    # test step :
    accuracy_accumulator = 0
    count = 0
    with torch.no_grad():
        for batch in test__dataloader:
            X,y = batch
            y_pred = net(X)
            accuracy_accumulator += torch.sum(y==y_pred.argmax(dim=1)).detach().item()
            count += y.shape[0]
    test_accuracy = accuracy_accumulator/count

    return test_accuracy

######################################
# Second, the sim > emp functions for both EEGNet and CCA
######################################

#Needs a final check, possibly unfinished
#Model_ind: 1 = original, 2= noiseDraw, 3= bothDraw  (sub-parts dependent on sGen)
#ind 4 = second draw version with Gamma,  ind 5 = custom draw
def do_EEGNet_sim_emp_final(model_ind,device,sGen,nGen,gener,processor=None,sim_sub_list=[],emp_sub_list=[],sim_trials=[500,50],snr_base=0.68,
                 epochs=5,ep_noImp=10,F1=8,F2=16,D=2,drop=0.1,lr=5e-3,splices=1,batch_size=10,numSub=30,n_folds=5,in_chans=1):

    snr = snr_base
    snrs1 = [snr for i in range(sim_trials[0])]
    snrs2 = [snr for i in range(sim_trials[1])]

    argPink = {"sizeof":[1,1],"mask":[],"exponent":1}
    argGauss = {"sizeof":[1,1]}
    argSpike = {"sizeof":[1,1],"fs":120}
    argAlpha = {"sizeof":[1,1],"fs":120}

    noise_params = {"types":[nGen.genPink,nGen.genGauss,nGen.genFreqSpike,nGen.genAlpha],
                   "weights":[18/40,2/40,17/40,3/40],
                   "params":[argPink,argGauss,argSpike,argAlpha],
                   "channels":1}
    
    if model_ind == 1:
        codesT,targetsT,resultsT = gener.genN(sim_trials[0],snrs1,noise_params,maxRange=1,subjects=sim_sub_list,processor=processor)
        codesV,targetsV,resultsV = gener.genN(sim_trials[1],snrs2,noise_params,maxRange=1,subjects=sim_sub_list,processor=processor)
    elif model_ind == 2:
        codesT,targetsT,resultsT = gener.genNDrawn1(sim_trials[0],snrs1,subjects=sim_sub_list,processor=processor)
        codesV,targetsV,resultsV = gener.genNDrawn1(sim_trials[1],snrs2,subjects=sim_sub_list,processor=processor)
    elif model_ind == 3:
        codesT,targetsT,resultsT = gener.genNDrawn2(sim_trials[0],snrs1,processor=processor)
        codesV,targetsV,resultsV = gener.genNDrawn2(sim_trials[1],snrs2,processor=processor)
    elif model_ind == 4:
        codesT,targetsT,resultsT = gener.genNDrawn2_2(sim_trials[0],snrs1,processor=processor)
        codesV,targetsV,resultsV = gener.genNDrawn2_2(sim_trials[1],snrs2,processor=processor)
    elif model_ind == 5:
        codesT,targetsT,resultsT = gener.genNDrawn2_3(sim_trials[0],snrs1,processor=processor)
        codesV,targetsV,resultsV = gener.genNDrawn2_3(sim_trials[1],snrs2,processor=processor)
    

    resultsT = processor.process(resultsT)
    resultsT = np.transpose(resultsT,(2,0,1))
    resultsT = resultsT * 0.00002

    resultsT,targetsT = splice_data(resultsT,targetsT,splices)

    resultsV = processor.process(resultsV)
    resultsV = np.transpose(resultsV,(2,0,1))
    resultsV = resultsV * 0.00002

    resultsV,targetsV = splice_data(resultsV,targetsV,splices)

    train_X_tensor = torch.tensor(resultsT,dtype=torch.float32, device=device)
    train_y_tensor = torch.tensor(targetsT, device=device) 

    valid_X_tensor = torch.tensor(resultsV,dtype=torch.float32, device=device)
    valid_y_tensor = torch.tensor(targetsV, device=device) 

    train_y_tensor = train_y_tensor.type(torch.LongTensor)
    valid_y_tensor = valid_y_tensor.type(torch.LongTensor)

    train_dataloader = DataLoader(TensorDataset(train_X_tensor, train_y_tensor), batch_size=batch_size, drop_last=False)
    valid_dataloader = DataLoader(TensorDataset(valid_X_tensor, valid_y_tensor), batch_size=batch_size, drop_last=False)

    net = EEGNet(20, F1=F1,F2=F2,D=D,drop_prob=drop,inp_length = resultsT.shape[2],in_chans=in_chans).to(device)

    scores_df, trained_net = train_net_final(
        net, 
        train_dataloader=train_dataloader, 
        valid_dataloader=valid_dataloader,
        lr=lr, 
        max_epochs=epochs, 
        max_epochs_without_improvement=ep_noImp,
    )

    ######################################
    #Next, the testing aspect
    ######################################

    X_emp_orig,y_emp_orig,V,M,fs,n_classes,n_samples_transient = sGen.get_needed_values()
    
    final_evals = np.zeros((numSub,n_folds))

    #The choice of train and test data needs to be adapted, also I need folds
    for index,sub_i in tqdm(enumerate(emp_sub_list)):

        X_emp = X_emp_orig[[sub_i]]
        y_emp = y_emp_orig[[sub_i]]

        X_emp = np.concatenate([x for x in X_emp],axis=2)
        y_emp = np.concatenate([y for y in y_emp],axis=0)

        n_trials = X_emp.shape[2]

        folds = np.resize(np.arange(n_folds), n_trials)

        for i_fold in tqdm(range(n_folds)):
            X_test, y_test = X_emp[:, :,folds == i_fold], y_emp[folds == i_fold]

            X_test = np.transpose(X_test,(2,0,1))
            X_test,y_test = splice_data(X_test,y_test,splices)
            y_test = y_test.astype(int)

            test__X_tensor = torch.tensor(X_test,dtype=torch.float32, device=device)
            test__y_tensor = torch.tensor(y_test, device=device)

            test__y_tensor = test__y_tensor.type(torch.LongTensor)

            test__dataloader = DataLoader(TensorDataset(test__X_tensor, test__y_tensor), batch_size=batch_size, drop_last=False)
            
            test_accuracy = test_net_final(trained_net,test__dataloader)
        
            final_evals[index,i_fold] = 100*test_accuracy

    return scores_df,trained_net,final_evals    

#CCA sim > emp
def do_CCA_sim_emp_final(model_ind,sGen,nGen,gener,processor=None,sim_sub_list=np.arange(30),emp_sub_list=np.arange(30),sim_trials=500,snr_base=0.68,
                 splices=1,numSub=30,n_folds=5,n_channels=1):
    
    snr = snr_base
    snrs1 = [snr for i in range(sim_trials)]

    argPink = {"sizeof":[1,1],"mask":[],"exponent":1}
    argGauss = {"sizeof":[1,1]}
    argSpike = {"sizeof":[1,1],"fs":120}
    argAlpha = {"sizeof":[1,1],"fs":120}

    noise_params = {"types":[nGen.genPink,nGen.genGauss,nGen.genFreqSpike,nGen.genAlpha],
                   "weights":[18/40,2/40,17/40,3/40],
                   "params":[argPink,argGauss,argSpike,argAlpha],
                   "channels":1}
    
    if model_ind == 1:
        codesT,y_train,X_train = gener.genN(sim_trials,snrs1,noise_params,maxRange=1,subjects=sim_sub_list,processor=processor)
    elif model_ind == 2:
        codesT,y_train,X_train = gener.genNDrawn1(sim_trials,snrs1,subjects=sim_sub_list,processor=processor)
    elif model_ind == 3:
        codesT,y_train,X_train = gener.genNDrawn2(sim_trials,snrs1,processor=processor)
    elif model_ind == 4:
        codesT,y_train,X_train = gener.genNDrawn2_2(sim_trials,snrs1,processor=processor)
    elif model_ind == 5:
        codesT,y_train,X_train = gener.genNDrawn2_3(sim_trials,snrs1,processor=processor)
    

    X_train = processor.process(X_train)
    X_train = np.transpose(X_train,(2,0,1))
    X_train = X_train * 0.00002

    X_train,y_train = splice_data(X_train,y_train,splices)

    #Here should be the training of the CCA method, and any and all necessary transposes etc.

    y_train = y_train.astype(int)

    X_train = np.transpose(X_train,(1,2,0))

    X_emp_orig,y_emp_orig,V,M,fs,n_classes,n_samples_transient = sGen.get_needed_values()

    cca = CCA(n_components=1)

    trial_length = 2.1*(15/splices)
    n_trials = X_train.shape[2]
    n_samples = int(trial_length * fs)
            
    # Reshape the training data
    X_ = np.reshape(X_train, (n_channels, -1)).T
    M_ = np.tile(M, (int(np.ceil(n_samples/V.shape[0])), 1, 1))[:n_samples, :, :]
    M_ = M_[:, :, y_train]
    M_ = M_.transpose((1, 0, 2))
    M_ = np.reshape(M_, (2*n_samples_transient, -1)).T

    # Fit CCA using training data
    cca.fit(X_.astype("float32"), M_.astype("float32"))
    w = cca.x_weights_.flatten()
    r = cca.y_weights_.flatten()

    # Predict templates
    T = np.zeros((M.shape[0], n_classes))
    for i_class in range(n_classes):
        T[:, i_class] = np.dot(M[:, :, i_class], r)
    T = np.tile(T, (int(np.ceil(n_samples/V.shape[0])), 1))[:n_samples, :]

    ######################################
    #Next, the testing aspect
    ######################################

    final_evals = np.zeros((numSub,n_folds))

    for index,sub_i in tqdm(enumerate(emp_sub_list)):

        X_emp = X_emp_orig[[sub_i]]
        y_emp = y_emp_orig[[sub_i]]

        X_emp = np.concatenate([x for x in X_emp],axis=2)
        y_emp = np.concatenate([y for y in y_emp],axis=0)
        y_emp = y_emp.astype(int)

        n_trials = X_emp.shape[2]

        folds = np.resize(np.arange(n_folds), n_trials)

        for i_fold in tqdm(range(n_folds)):
            #Actually do the splitting here
            # Split data to train and valid set
            X_test, y_test = X_emp[:, :, folds == i_fold], y_emp[folds == i_fold]

            X_test = np.transpose(X_test,(2,0,1))
            X_test,y_test = splice_data(X_test,y_test,splices)
            y_test = y_test.astype(int)
            X_test = np.transpose(X_test,(1,2,0))

            # Spatially filter validation data
            X_filtered = np.zeros((n_samples, y_test.size))
            for i_trial in range(y_test.size):
                X_filtered[:, i_trial] = np.dot(w, X_test[:, :, i_trial])

            # Template matching
            prediction = np.zeros(y_test.size)
            for i_trial in range(y_test.size):
                rho = np.corrcoef(X_filtered[:, i_trial], T.T)[0, 1:]
                prediction[i_trial] = np.argmax(rho)

            final_evals[index,i_fold] = 100*np.mean(prediction == y_test)

    return final_evals


######################################
# Third, the emp > emp functions for both EEGNet and CCA
######################################

def do_EEGNet_emp_emp_final(device,sGen,emp_sub_list=[],epochs=200,ep_noImp=10,F1=8,F2=16,D=2,n_folds=5,
            drop=0.1,lr=5e-3,splices=1,batch_size=10,numSub=30,val_portion=0.2,in_chans=1):
    
    X_emp_orig,y_emp_orig,V,M,fs,n_classes,n_samples_transient = sGen.get_needed_values()

    final_evals = np.zeros((numSub,n_folds))

    #The choice of train and test data needs to be adapted, also I need folds
    for index,sub_i in tqdm(enumerate(emp_sub_list)):

        X_emp = X_emp_orig[[sub_i]]
        y_emp = y_emp_orig[[sub_i]]

        X_emp = np.concatenate([x for x in X_emp],axis=2)
        y_emp = np.concatenate([y for y in y_emp],axis=0)

        n_trials = X_emp.shape[2]

        folds = np.resize(np.arange(n_folds), n_trials)

        for i_fold in tqdm(range(n_folds)):
            X_train, y_train = X_emp[:, :,folds != i_fold], y_emp[folds != i_fold]
            X_test, y_test = X_emp[:, :,folds == i_fold], y_emp[folds == i_fold]
            
            #Add validation stuff
            n_trials_train = int((1-val_portion)*X_train.shape[2])
            n_trials_orig = X_train.shape[2]
            indices_train = np.random.choice(n_trials_orig,size=n_trials_train,replace=False)
            indices_val = [x for x in np.arange(n_trials_orig) if x not in indices_train]

            X_val,y_val = X_train[:,:,indices_val],y_train[indices_val]
            X_train,y_train = X_train[:,:,indices_train],y_train[indices_train]

            X_train = np.transpose(X_train,(2,0,1))
            X_train,y_train = splice_data(X_train,y_train,splices)
            y_train = y_train.astype(int)

            X_val = np.transpose(X_val,(2,0,1))
            X_val,y_val = splice_data(X_val,y_val,splices)
            y_val = y_val.astype(int)

            X_test = np.transpose(X_test,(2,0,1))
            X_test,y_test = splice_data(X_test,y_test,splices)
            y_test = y_test.astype(int)

            train_X_tensor = torch.tensor(X_train,dtype=torch.float32, device=device)
            valid_X_tensor = torch.tensor(X_val,dtype=torch.float32, device=device)
            test__X_tensor = torch.tensor(X_test,dtype=torch.float32, device=device)
            train_y_tensor = torch.tensor(y_train, device=device) 
            valid_y_tensor = torch.tensor(y_val, device=device) 
            test__y_tensor = torch.tensor(y_test, device=device)

            #What to do with the validation set?

            train_y_tensor = train_y_tensor.type(torch.LongTensor)
            valid_y_tensor = valid_y_tensor.type(torch.LongTensor)
            test__y_tensor = test__y_tensor.type(torch.LongTensor)

            train_dataloader = DataLoader(TensorDataset(train_X_tensor, train_y_tensor), batch_size=batch_size, drop_last=False)
            valid_dataloader = DataLoader(TensorDataset(valid_X_tensor, valid_y_tensor), batch_size=batch_size, drop_last=False)
            test__dataloader = DataLoader(TensorDataset(test__X_tensor, test__y_tensor), batch_size=batch_size, drop_last=False)
    
            net = EEGNet(20, F1=F1,D=D,F2=F2,drop_prob=drop,inp_length = X_train.shape[2],in_chans=in_chans).to(device)

            scores_df, trained_net = train_net_final(
                net, 
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader, 
                lr=lr, 
                max_epochs=epochs, 
                max_epochs_without_improvement=ep_noImp,
            )

            test_accuracy = test_net_final(trained_net,test__dataloader)
        
            final_evals[index,i_fold] = 100*test_accuracy

    return final_evals

#CCA for emp > emp
def do_CCA_emp_emp_final(sGen,emp_sub_list=[],n_folds=5,
            splices=1,numSub=30,val_portion=0.2,n_channels=1):
    
    X_emp_orig,y_emp_orig,V,M,fs,n_classes,n_samples_transient = sGen.get_needed_values()

    final_evals = np.zeros((numSub,n_folds))

    #The choice of train and test data needs to be adapted, also I need folds
    for index,sub_i in tqdm(enumerate(emp_sub_list)):

        X_emp = X_emp_orig[[sub_i]]
        y_emp = y_emp_orig[[sub_i]]

        X_emp = np.concatenate([x for x in X_emp],axis=2)
        y_emp = np.concatenate([y for y in y_emp],axis=0)

        trial_length = 2.1*(15/splices)
        n_trials = X_emp.shape[2]
        n_samples = int(trial_length * fs)

        folds = np.resize(np.arange(n_folds), n_trials)

        cca = CCA(n_components=1)

        for i_fold in tqdm(range(n_folds)):
            X_train, y_train = X_emp[:, :,folds != i_fold], y_emp[folds != i_fold]
            X_test, y_test = X_emp[:, :,folds == i_fold], y_emp[folds == i_fold]

            X_train = np.transpose(X_train,(2,0,1))
            X_train,y_train = splice_data(X_train,y_train,splices)
            y_train = y_train.astype(int)
            X_train = np.transpose(X_train,(1,2,0))

            X_test = np.transpose(X_test,(2,0,1))
            X_test,y_test = splice_data(X_test,y_test,splices)
            y_test = y_test.astype(int)
            X_test = np.transpose(X_test,(1,2,0))

            # -- TRAINING
            
            # Reshape the training data
            X_ = np.reshape(X_train, (n_channels, -1)).T
            M_ = np.tile(M, (int(np.ceil(n_samples/V.shape[0])), 1, 1))[:n_samples, :, :]
            M_ = M_[:, :, y_train]
            M_ = M_.transpose((1, 0, 2))
            M_ = np.reshape(M_, (2*n_samples_transient, -1)).T

            # Fit CCA using training data
            cca.fit(X_.astype("float32"), M_.astype("float32"))
            w = cca.x_weights_.flatten()
            r = cca.y_weights_.flatten()

            # Predict templates
            T = np.zeros((M.shape[0], n_classes))
            for i_class in range(n_classes):
                T[:, i_class] = np.dot(M[:, :, i_class], r)
            T = np.tile(T, (int(np.ceil(n_samples/V.shape[0])), 1))[:n_samples, :]

            # -- VALIDATION

            # Spatially filter validation data
            X_filtered = np.zeros((n_samples, y_test.size))
            for i_trial in range(y_test.size):
                X_filtered[:, i_trial] = np.dot(w, X_test[:, :, i_trial])

            # Template matching
            prediction = np.zeros(y_test.size)
            for i_trial in range(y_test.size):
                rho = np.corrcoef(X_filtered[:, i_trial], T.T)[0, 1:]
                prediction[i_trial] = np.argmax(rho)

            # Compute accuracy
            final_evals[index,i_fold] = 100*np.mean(prediction == y_test)

    return final_evals


######################################
# Last, the sim > sim functions for both EEGNet and CCA
######################################

def do_EEGNet_sim_sim_final(model_ind,device,sGen,nGen,gener,processor=None,sim_sub_list=[],sim_trials=[500,50,50],snr_base=0.68,
                 epochs=5,ep_noImp=10,F1=8,F2=16,D=2,drop=0.1,lr=5e-3,splices=1,batch_size=10,n_folds=5,in_chans=1):

    snr = snr_base
    snrs1 = [snr for i in range(sim_trials[0])]
    snrs2 = [snr for i in range(sim_trials[1])]

    argPink = {"sizeof":[1,1],"mask":[],"exponent":1}
    argGauss = {"sizeof":[1,1]}
    argSpike = {"sizeof":[1,1],"fs":120}
    argAlpha = {"sizeof":[1,1],"fs":120}

    noise_params = {"types":[nGen.genPink,nGen.genGauss,nGen.genFreqSpike,nGen.genAlpha],
                   "weights":[18/40,2/40,17/40,3/40],
                   "params":[argPink,argGauss,argSpike,argAlpha],
                   "channels":1}
    
    final_evals = np.zeros(n_folds)

    for i in tqdm(range(n_folds)):
        if model_ind == 1:
            codesT,targetsT,resultsT = gener.genN(sim_trials[0],snrs1,noise_params,maxRange=1,subjects=sim_sub_list,processor=processor)
            codesV,targetsV,resultsV = gener.genN(sim_trials[1],snrs2,noise_params,maxRange=1,subjects=sim_sub_list,processor=processor)
            codesTest,targetsTest,resultsTest = gener.genN(sim_trials[2],snrs2,noise_params,maxRange=1,subjects=sim_sub_list,processor=processor)
        elif model_ind == 2:
            codesT,targetsT,resultsT = gener.genNDrawn1(sim_trials[0],snrs1,subjects=sim_sub_list,processor=processor)
            codesV,targetsV,resultsV = gener.genNDrawn1(sim_trials[1],snrs2,subjects=sim_sub_list,processor=processor)
            codesTest,targetsTest,resultsTest = gener.genNDrawn1(sim_trials[2],snrs2,subjects=sim_sub_list,processor=processor)
        elif model_ind == 3:
            codesT,targetsT,resultsT = gener.genNDrawn2(sim_trials[0],snrs1,processor=processor)
            codesV,targetsV,resultsV = gener.genNDrawn2(sim_trials[1],snrs2,processor=processor)
            codesTest,targetsTest,resultsTest = gener.genNDrawn2(sim_trials[2],snrs2,processor=processor)
        elif model_ind == 4:
            codesT,targetsT,resultsT = gener.genNDrawn2_2(sim_trials[0],snrs1,processor=processor)
            codesV,targetsV,resultsV = gener.genNDrawn2_2(sim_trials[1],snrs2,processor=processor)
            codesTest,targetsTest,resultsTest = gener.genNDrawn2_2(sim_trials[2],snrs2,processor=processor)
        elif model_ind == 5:
            codesT,targetsT,resultsT = gener.genNDrawn2_3(sim_trials[0],snrs1,processor=processor)
            codesV,targetsV,resultsV = gener.genNDrawn2_3(sim_trials[1],snrs2,processor=processor)
            codesTest,targetsTest,resultsTest = gener.genNDrawn2_3(sim_trials[2],snrs2,processor=processor)
        

        resultsT = processor.process(resultsT)
        resultsT = np.transpose(resultsT,(2,0,1))
        resultsT = resultsT * 0.00002
        resultsT,targetsT = splice_data(resultsT,targetsT,splices)

        resultsV = processor.process(resultsV)
        resultsV = np.transpose(resultsV,(2,0,1))
        resultsV = resultsV * 0.00002
        resultsV,targetsV = splice_data(resultsV,targetsV,splices)

        resultsTest = processor.process(resultsTest)
        resultsTest = np.transpose(resultsTest,(2,0,1))
        resultsTest = resultsTest * 0.00002
        resultsTest,targetsTest = splice_data(resultsTest,targetsTest,splices)

        train_X_tensor = torch.tensor(resultsT,dtype=torch.float32, device=device)
        train_y_tensor = torch.tensor(targetsT, device=device) 
        valid_X_tensor = torch.tensor(resultsV,dtype=torch.float32, device=device)
        valid_y_tensor = torch.tensor(targetsV, device=device) 
        test_X_tensor = torch.tensor(resultsTest,dtype=torch.float32, device=device)
        test_y_tensor = torch.tensor(targetsTest, device=device) 

        train_y_tensor = train_y_tensor.type(torch.LongTensor)
        valid_y_tensor = valid_y_tensor.type(torch.LongTensor)
        test_y_tensor = test_y_tensor.type(torch.LongTensor)

        train_dataloader = DataLoader(TensorDataset(train_X_tensor, train_y_tensor), batch_size=batch_size, drop_last=False)
        valid_dataloader = DataLoader(TensorDataset(valid_X_tensor, valid_y_tensor), batch_size=batch_size, drop_last=False)
        test_dataloader = DataLoader(TensorDataset(test_X_tensor, test_y_tensor), batch_size=batch_size, drop_last=False)

        net = EEGNet(20, F1=F1,F2=F2,D=D,drop_prob=drop,inp_length = resultsT.shape[2],in_chans=in_chans).to(device)

        scores_df, trained_net = train_net_final(
            net, 
            train_dataloader=train_dataloader, 
            valid_dataloader=valid_dataloader,
            lr=lr, 
            max_epochs=epochs, 
            max_epochs_without_improvement=ep_noImp,
        )

        test_accuracy = test_net_final(trained_net,test_dataloader)
        final_evals[i] = 100*test_accuracy

    return scores_df,trained_net,final_evals 

def do_CCA_sim_sim_final(model_ind,sGen,nGen,gener,processor=None,sim_sub_list=[],sim_trials=[500,50],snr_base=0.68,
                 splices=1,n_folds=5,n_channels=1):

    snr = snr_base
    snrs1 = [snr for i in range(sim_trials[0])]
    snrs2 = [snr for i in range(sim_trials[1])]

    argPink = {"sizeof":[1,1],"mask":[],"exponent":1}
    argGauss = {"sizeof":[1,1]}
    argSpike = {"sizeof":[1,1],"fs":120}
    argAlpha = {"sizeof":[1,1],"fs":120}

    noise_params = {"types":[nGen.genPink,nGen.genGauss,nGen.genFreqSpike,nGen.genAlpha],
                   "weights":[18/40,2/40,17/40,3/40],
                   "params":[argPink,argGauss,argSpike,argAlpha],
                   "channels":1}
    
    final_evals = np.zeros(n_folds)

    for i in tqdm(range(n_folds)):
        if model_ind == 1:
            codesT,targetsT,resultsT = gener.genN(sim_trials[0],snrs1,noise_params,maxRange=1,subjects=sim_sub_list,processor=processor)
            codesTest,targetsTest,resultsTest = gener.genN(sim_trials[1],snrs2,noise_params,maxRange=1,subjects=sim_sub_list,processor=processor)
        elif model_ind == 2:
            codesT,targetsT,resultsT = gener.genNDrawn1(sim_trials[0],snrs1,subjects=sim_sub_list,processor=processor)
            codesTest,targetsTest,resultsTest = gener.genNDrawn1(sim_trials[1],snrs2,subjects=sim_sub_list,processor=processor)
        elif model_ind == 3:
            codesT,targetsT,resultsT = gener.genNDrawn2(sim_trials[0],snrs1,processor=processor)
            codesTest,targetsTest,resultsTest = gener.genNDrawn2(sim_trials[1],snrs2,processor=processor)
        elif model_ind == 4:
            codesT,targetsT,resultsT = gener.genNDrawn2_2(sim_trials[0],snrs1,processor=processor)
            codesTest,targetsTest,resultsTest = gener.genNDrawn2_2(sim_trials[1],snrs2,processor=processor)
        elif model_ind == 5:
            codesT,targetsT,resultsT = gener.genNDrawn2_3(sim_trials[0],snrs1,processor=processor)
            codesTest,targetsTest,resultsTest = gener.genNDrawn2_3(sim_trials[1],snrs2,processor=processor)
        

        resultsT = processor.process(resultsT)
        resultsT = np.transpose(resultsT,(2,0,1))
        resultsT = resultsT * 0.00002
        resultsT,targetsT = splice_data(resultsT,targetsT,splices)

        resultsTest = processor.process(resultsTest)
        resultsTest = np.transpose(resultsTest,(2,0,1))
        resultsTest = resultsTest * 0.00002
        resultsTest,targetsTest = splice_data(resultsTest,targetsTest,splices)

        y_train = targetsT.astype(int)
        y_test = targetsTest.astype(int)

        X_train = np.transpose(resultsT,(1,2,0))
        X_test = np.transpose(resultsTest,(1,2,0))

        X_emp_orig,y_emp_orig,V,M,fs,n_classes,n_samples_transient = sGen.get_needed_values()

        cca = CCA(n_components=1)

        trial_length = 2.1*(15/splices)
        n_trials = X_train.shape[2]
        n_samples = int(trial_length * fs)
                
        # Reshape the training data
        X_ = np.reshape(X_train, (n_channels, -1)).T
        M_ = np.tile(M, (int(np.ceil(n_samples/V.shape[0])), 1, 1))[:n_samples, :, :]
        M_ = M_[:, :, y_train]
        M_ = M_.transpose((1, 0, 2))
        M_ = np.reshape(M_, (2*n_samples_transient, -1)).T

        # Fit CCA using training data
        cca.fit(X_.astype("float32"), M_.astype("float32"))
        w = cca.x_weights_.flatten()
        r = cca.y_weights_.flatten()

        # Predict templates
        T = np.zeros((M.shape[0], n_classes))
        for i_class in range(n_classes):
            T[:, i_class] = np.dot(M[:, :, i_class], r)
        T = np.tile(T, (int(np.ceil(n_samples/V.shape[0])), 1))[:n_samples, :]

        #Testing:
        # Spatially filter validation data
        X_filtered = np.zeros((n_samples, y_test.size))
        for i_trial in range(y_test.size):
            X_filtered[:, i_trial] = np.dot(w, X_test[:, :, i_trial])

        # Template matching
        prediction = np.zeros(y_test.size)
        for i_trial in range(y_test.size):
            rho = np.corrcoef(X_filtered[:, i_trial], T.T)[0, 1:]
            prediction[i_trial] = np.argmax(rho)

        final_evals[i] = 100*np.mean(prediction == y_test)

    return final_evals 