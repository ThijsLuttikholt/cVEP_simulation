import os
import numpy as np
import random
import math

from copy import deepcopy

import glob
import h5py
import mne
import time

import scipy.io
from scipy.interpolate import griddata
from scipy import signal
from scipy.fftpack import fft,ifft
from scipy.signal import periodogram
from scipy.stats import truncnorm
from scipy.special import gamma as funGamma
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from matplotlib.patches import Circle, Ellipse, Polygon
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline, BSpline

import pandas as pd
import torch
from torch import nn
from torch.nn.functional import elu
from torch.utils.data import TensorDataset, DataLoader

from .extra_functions import splice_data

import colorednoise as cn

from scipy.stats import (
    norm, beta, expon, gamma, genextreme, logistic, lognorm, triang, uniform, fatiguelife,            
    gengamma, gennorm, dweibull, dgamma, gumbel_r, powernorm, rayleigh, weibull_max, weibull_min, 
    laplace, alpha, genexpon, bradford, betaprime, burr, fisk, genpareto, hypsecant, 
    halfnorm, halflogistic, invgauss, invgamma, levy, loglaplace, loggamma, maxwell, 
    mielke, ncx2, ncf, nct, nakagami, pareto, lomax, powerlognorm, powerlaw, rice, 
    semicircular, trapezoid, rice, invweibull, foldnorm, foldcauchy, cosine, exponpow, 
    exponweib, wald, wrapcauchy, truncexpon, truncnorm, t, rdist
    )


from .training import train_net,train_net2, train_net_highB
from .EEGNet import EEGNet

#Note the below uses genNDrawn in all cases. Might need to change that.
def do_EEGNet_sim(gener,device,processor=None,n_trials=[1000,100,100]):

    #NEED TO SAMPLES SNRS BASED ON A GIVEN RANGE HERE
    lower, higher = 0.5, 1.5
    mu, sigma = 1, 0.2
    drawer = truncnorm((lower-mu)/sigma,(higher-mu)/sigma, loc=mu, scale=sigma)
    #snrs = drawer.rvs(n_trials)
    
    snrs = [0.68 for i in range(n_trials[0])]
    
    #genNDrawn(self,n,snrs,subjects=[],code_times=15,cutoff=36,processor=None):
    
    codesT,targetsT,resultsT = gener.genNDrawn(n_trials[0],snrs,processor=processor)
    resultsT = processor.process(resultsT)
    resultsT = np.transpose(resultsT,(2,0,1))

    codesV,targetsV,resultsV = gener.genNDrawn(n_trials[1],snrs[:n_trials[1]],processor=processor)
    resultsV = processor.process(resultsV)
    resultsV = np.transpose(resultsV,(2,0,1))

    codesTest,targetsTest,resultsTest = gener.genNDrawn(n_trials[2],snrs[:n_trials[1]],processor=processor)
    resultsTest = processor.process(resultsTest)
    resultsTest = np.transpose(resultsTest,(2,0,1))
    
    resultsT = resultsT * 0.00002
    resultsV = resultsV * 0.00002
    resultsTest = resultsTest * 0.00002

    train_X_tensor = torch.tensor(resultsT,dtype=torch.float32, device=device)
    valid_X_tensor = torch.tensor(resultsV,dtype=torch.float32, device=device)
    test__X_tensor = torch.tensor(resultsTest,dtype=torch.float32, device=device)
    train_y_tensor = torch.tensor(targetsT, device=device)
    valid_y_tensor = torch.tensor(targetsV, device=device)
    test__y_tensor = torch.tensor(targetsTest, device=device)

    train_y_tensor = train_y_tensor.type(torch.LongTensor)
    valid_y_tensor = valid_y_tensor.type(torch.LongTensor)
    test__y_tensor = test__y_tensor.type(torch.LongTensor)

    batch_size=10
    train_dataloader = DataLoader(TensorDataset(train_X_tensor, train_y_tensor), batch_size=batch_size, drop_last=False)
    valid_dataloader = DataLoader(TensorDataset(valid_X_tensor, valid_y_tensor), batch_size=batch_size, drop_last=False)
    test__dataloader = DataLoader(TensorDataset(test__X_tensor, test__y_tensor), batch_size=batch_size, drop_last=False)
    
    net = EEGNet(20, drop_prob=0.1).to(device)

    scores_df, test_accuracy, net = train_net(
        net, 
        train_dataloader=train_dataloader, 
        valid_dataloader=valid_dataloader, 
        test__dataloader=test__dataloader, 
        lr=5e-3, 
        max_epochs=50, 
        max_epochs_without_improvement=50,
    )
    
    return test_accuracy

###########################################################################

###########################################################################

def do_EEGNet_emp_redraw(sGen,gener,device,processor=None,emp_sub_list=[],perRound=1000,numValid=200,snr_base=0.68):

    snr = 0.68
    snrs1 = [snr for i in range(perRound)]
    snrs2 = [snr for i in range(numValid)]

    codesV,targetsV,resultsV = gener.genNDrawn(numValid,snrs2,processor=processor)
    resultsV = processor.process(resultsV)
    resultsV = np.transpose(resultsV,(2,0,1))
    
    resultsV = resultsV * 0.00002

    ######################################
    #Add the different generation for the testing stuff
    ######################################
    
    X_emp,y_emp,V,M,fs,n_classes,n_samples_transient = sGen.get_needed_values()
    if len(emp_sub_list) != 0:
        X_emp = X_emp[emp_sub_list]
        y_emp = y_emp[emp_sub_list]

    X_emp = np.concatenate([x for x in X_emp],axis=2)
    y_emp = np.concatenate([y for y in y_emp],axis=0)

    X_emp = np.transpose(X_emp,(2,0,1))

    valid_X_tensor = torch.tensor(resultsV,dtype=torch.float32, device=device)
    test__X_tensor = torch.tensor(X_emp,dtype=torch.float32, device=device)
    valid_y_tensor = torch.tensor(targetsV, device=device)
    test__y_tensor = torch.tensor(y_emp, device=device)

    valid_y_tensor = valid_y_tensor.type(torch.LongTensor)
    test__y_tensor = test__y_tensor.type(torch.LongTensor)

    batch_size=10
    valid_dataloader = DataLoader(TensorDataset(valid_X_tensor, valid_y_tensor), batch_size=batch_size, drop_last=False)
    test__dataloader = DataLoader(TensorDataset(test__X_tensor, test__y_tensor), batch_size=batch_size, drop_last=False)
    
    net = EEGNet(20, drop_prob=0.1).to(device)

    scores_df, test_accuracy, net = train_net2(
        net, 
        gener,
        perRound,
        snrs1,
        batch_size,
        processor,
        valid_dataloader=valid_dataloader, 
        test__dataloader=test__dataloader, 
        lr=5e-3, 
        max_epochs=50, 
        max_epochs_without_improvement=50,
    )
    
    return scores_df,test_accuracy,net

###################################################

###################################################

#Model_ind: 1 = original, 2= noiseDraw, 3= bothDraw  (sub-parts dependent on sGen)
#ind 4 = second draw version with Gamma,  ind 5 = custom draw
def do_EEGNet_emp(model_ind,device,sGen,nGen,gener,processor=None,sim_sub_list=[],emp_sub_list=[],n_trials=[500,50],snr_base=0.68,
                 epochs=5,F1=8,drop=0.1,lr=5e-3,splices=1,tiv=False,batch_size=10):

    snr = 0.68
    snrs1 = [snr for i in range(n_trials[0])]
    snrs2 = [snr for i in range(n_trials[1])]

    argPink = {"sizeof":[1,1],"mask":[],"exponent":1}#1.7}
    argGauss = {"sizeof":[1,1]}
    argSpike = {"sizeof":[1,1],"fs":120}
    argAlpha = {"sizeof":[1,1],"fs":120}

    noise_params = {"types":[nGen.genPink,nGen.genGauss,nGen.genFreqSpike,nGen.genAlpha],
                   "weights":[18/40,2/40,17/40,3/40],
                   "params":[argPink,argGauss,argSpike,argAlpha],
                   "channels":1}
    
    if model_ind == 1:
        codesT,targetsT,resultsT = gener.genN(n_trials[0],snrs1,noise_params,maxRange=1,subjects=sim_sub_list,processor=processor)
        codesV,targetsV,resultsV = gener.genN(n_trials[1],snrs2,noise_params,maxRange=1,subjects=sim_sub_list,processor=processor)
    elif model_ind == 2:
        codesT,targetsT,resultsT = gener.genNDrawn1(n_trials[0],snrs1,subjects=sim_sub_list,processor=processor)
        codesV,targetsV,resultsV = gener.genNDrawn1(n_trials[1],snrs2,subjects=sim_sub_list,processor=processor)
    elif model_ind == 3:
        codesT,targetsT,resultsT = gener.genNDrawn2(n_trials[0],snrs1,processor=processor)
        codesV,targetsV,resultsV = gener.genNDrawn2(n_trials[1],snrs2,processor=processor)
    elif model_ind == 4:
        codesT,targetsT,resultsT = gener.genNDrawn2_2(n_trials[0],snrs1,processor=processor)
        codesV,targetsV,resultsV = gener.genNDrawn2_2(n_trials[1],snrs2,processor=processor)
    elif model_ind == 5:
        codesT,targetsT,resultsT = gener.genNDrawn2_3(n_trials[0],snrs1,processor=processor)
        codesV,targetsV,resultsV = gener.genNDrawn2_3(n_trials[1],snrs2,processor=processor)
    

    resultsT = processor.process(resultsT)
    resultsT = np.transpose(resultsT,(2,0,1))
    
    resultsT = resultsT * 0.00002


    resultsV = processor.process(resultsV)
    resultsV = np.transpose(resultsV,(2,0,1))
    
    resultsV = resultsV * 0.00002

    ######################################
    #Add the different generation for the testing stuff
    ######################################
    
    X_emp,y_emp,V,M,fs,n_classes,n_samples_transient = sGen.get_needed_values()
    if len(emp_sub_list) != 0:
        X_emp = X_emp[emp_sub_list]
        y_emp = y_emp[emp_sub_list]

    X_emp = np.concatenate([x for x in X_emp],axis=2)
    y_emp = np.concatenate([y for y in y_emp],axis=0)

    X_emp = np.transpose(X_emp,(2,0,1))

    resultsT,targetsT = splice_data(resultsT,targetsT,splices)
    resultsV,targetsV = splice_data(resultsV,targetsV,splices)
    X_emp,y_emp = splice_data(X_emp,y_emp,splices)
    
    if tiv:
        valid_X_tensor = torch.tensor(X_emp,dtype=torch.float32, device=device)
        valid_y_tensor= torch.tensor(y_emp,device=device)
    else:    
        valid_X_tensor = torch.tensor(resultsV,dtype=torch.float32, device=device)
        valid_y_tensor = torch.tensor(targetsV, device=device)
   
    train_X_tensor = torch.tensor(resultsT,dtype=torch.float32, device=device)
    test__X_tensor = torch.tensor(X_emp,dtype=torch.float32, device=device)
    train_y_tensor = torch.tensor(targetsT, device=device) 
    test__y_tensor = torch.tensor(y_emp, device=device)

    train_y_tensor = train_y_tensor.type(torch.LongTensor)
    valid_y_tensor = valid_y_tensor.type(torch.LongTensor)
    test__y_tensor = test__y_tensor.type(torch.LongTensor)

    train_dataloader = DataLoader(TensorDataset(train_X_tensor, train_y_tensor), batch_size=batch_size, drop_last=False)
    valid_dataloader = DataLoader(TensorDataset(valid_X_tensor, valid_y_tensor), batch_size=batch_size, drop_last=False)
    test__dataloader = DataLoader(TensorDataset(test__X_tensor, test__y_tensor), batch_size=batch_size, drop_last=False)
    
    net = EEGNet(20, F1=F1,drop_prob=drop,inp_length = X_emp.shape[2]).to(device)

    scores_df, test_accuracy, net = train_net(
        net, 
        train_dataloader=train_dataloader, 
        valid_dataloader=valid_dataloader, 
        test__dataloader=test__dataloader, 
        lr=lr, 
        max_epochs=epochs, 
        max_epochs_without_improvement=epochs,
    )
    
    return scores_df,test_accuracy,net

#Model_ind: 1 = original, 2= noiseDraw, 3= bothDraw  (sub-parts dependent on sGen)
#Model ind: 4 = new version of bothDraw
def do_EEGNet_emp_epSim(model_ind,device,sGen,nGen,gener,processor=None,sim_sub_list=[],emp_sub_list=[],n_trials=[500,50],snr_base=0.68,
                 epochs=5,F1=8,drop=0.1,lr=5e-3,splices=1,tiv=False,batch_size=10):

    snr = 0.68
    snrs1 = [snr for i in range(n_trials[0])]
    snrs2 = [snr for i in range(n_trials[1])]

    argPink = {"sizeof":[1,1],"mask":[],"exponent":1}#1.7}
    argGauss = {"sizeof":[1,1]}
    argSpike = {"sizeof":[1,1],"fs":120}
    argAlpha = {"sizeof":[1,1],"fs":120}

    noise_params = {"types":[nGen.genPink,nGen.genGauss,nGen.genFreqSpike,nGen.genAlpha],
                   "weights":[18/40,2/40,17/40,3/40],
                   "params":[argPink,argGauss,argSpike,argAlpha],
                   "channels":1}
    
    if model_ind == 1:
        codesV,targetsV,resultsV = gener.genN(n_trials[1],snrs2,noise_params,maxRange=1,subjects=sim_sub_list,processor=processor)
    elif model_ind == 2:
        codesV,targetsV,resultsV = gener.genNDrawn1(n_trials[1],snrs2,subjects=sim_sub_list,processor=processor)
    elif model_ind == 3:
        codesV,targetsV,resultsV = gener.genNDrawn2(n_trials[1],snrs2,processor=processor)
    elif model_ind == 4:
        codesV,targetsV,resultsV = gener.genNDrawn2_2(n_trials[1],snrs2,processor=processor)
    elif model_ind == 5:
        codesV,targetsV,resultsV = gener.genNDrawn2_3(n_trials[1],snrs2,processor=processor)

    resultsV = processor.process(resultsV)
    resultsV = np.transpose(resultsV,(2,0,1))
    
    resultsV = resultsV * 0.00002

    ######################################
    #Add the different generation for the testing stuff
    ######################################
    
    X_emp,y_emp,V,M,fs,n_classes,n_samples_transient = sGen.get_needed_values()
    if len(emp_sub_list) != 0:
        X_emp = X_emp[emp_sub_list]
        y_emp = y_emp[emp_sub_list]

    X_emp = np.concatenate([x for x in X_emp],axis=2)
    y_emp = np.concatenate([y for y in y_emp],axis=0)

    X_emp = np.transpose(X_emp,(2,0,1))

    
    resultsV,targetsV = splice_data(resultsV,targetsV,splices)
    X_emp,y_emp = splice_data(X_emp,y_emp,splices)
    
    if tiv:
        valid_X_tensor = torch.tensor(X_emp,dtype=torch.float32, device=device)
        valid_y_tensor= torch.tensor(y_emp,device=device)
    else:    
        valid_X_tensor = torch.tensor(resultsV,dtype=torch.float32, device=device)
        valid_y_tensor = torch.tensor(targetsV, device=device)

    test__X_tensor = torch.tensor(X_emp,dtype=torch.float32, device=device)
    test__y_tensor = torch.tensor(y_emp, device=device)

    valid_y_tensor = valid_y_tensor.type(torch.LongTensor)
    test__y_tensor = test__y_tensor.type(torch.LongTensor)

    batch_size=10
    valid_dataloader = DataLoader(TensorDataset(valid_X_tensor, valid_y_tensor), batch_size=batch_size, drop_last=False)
    test__dataloader = DataLoader(TensorDataset(test__X_tensor, test__y_tensor), batch_size=batch_size, drop_last=False)
    
    net = EEGNet(20, F1=F1,drop_prob=drop,inp_length = X_emp.shape[2],in_chans=X_emp.shape[1]).to(device)

    scores_df, test_accuracy, net = train_net2(
        net, 
        device,
        model_ind,
        nGen,
        gener,
        n_trials[0],
        sim_sub_list,
        splices,
        batch_size,
        processor,
        valid_dataloader=valid_dataloader, 
        test__dataloader=test__dataloader, 
        lr=lr, 
        max_epochs=epochs, 
        max_epochs_without_improvement=epochs,
    )

    return scores_df,test_accuracy,net


###############################################

###############################################

def do_EEGNet_2emp(device,sGen,emp_sub_list=[],n_folds = 10,epochs=5,F1=8,D=2,
        F2=16,drop=0.1,lr=5e-3,splices=5,allEps=True,batch_size=10):
    
    X_emp,y_emp,V,M,fs,n_classes,n_samples_transient = sGen.get_needed_values()
    if len(emp_sub_list) != 0:
        X_emp = X_emp[emp_sub_list]
        y_emp = y_emp[emp_sub_list]

    X_emp = np.concatenate([x for x in X_emp],axis=2)
    y_emp = np.concatenate([y for y in y_emp],axis=0)

    #Participants X Channels X Samples X trials
    X_emp = np.transpose(X_emp,(2,0,1)) #X_emp becomes trials*participants X channels X samples
    
    X_emp,y_emp = splice_data(X_emp,y_emp,splices)
    
    n_trials = len(X_emp)
    
    #folds = np.repeat(np.arange(n_folds), n_trials / n_folds)
    folds = np.resize(np.arange(n_folds), n_trials)

    # Loop folds
    accuracy = np.zeros(n_folds)
    score_dfs = []

    if allEps:
        toEp = n_folds
    else:
        toEp = 1

    #Need to switch this to only doing the first fold, for testing.
    for i_fold in tqdm(range(toEp)):#tqdm(range(n_folds)):
        
        train_X = X_emp[folds != i_fold,:,:]
        valid_X = X_emp[folds == i_fold,:,:]
        test_X = X_emp[folds == i_fold,:,:]
        train_y = y_emp[folds != i_fold]
        valid_y = y_emp[folds == i_fold]
        test_y = y_emp[folds == i_fold]

        #test__X_tensor = torch.tensor(X_emp,dtype=torch.float32, device=device)
        train_X_tensor = torch.tensor(train_X, dtype=torch.float32, device=device) 
        test_X_tensor = torch.tensor(test_X, dtype=torch.float32, device=device) 

        train_y_tensor = torch.tensor(train_y, device=device) 
        test_y_tensor =  torch.tensor(test_y, device=device) 

        train_y_tensor = train_y_tensor.type(torch.LongTensor)
        test_y_tensor = test_y_tensor.type(torch.LongTensor)

        train_dataloader = DataLoader(TensorDataset(train_X_tensor, train_y_tensor), batch_size=batch_size, drop_last=False)
        valid_dataloader = DataLoader(TensorDataset(test_X_tensor, test_y_tensor), batch_size=batch_size, drop_last=False)
        test__dataloader = DataLoader(TensorDataset(test_X_tensor, test_y_tensor), batch_size=batch_size, drop_last=False)

        net = EEGNet(20,F1=F1,F2=F2,D=D, drop_prob=drop,inp_length = X_emp.shape[2],in_chans=X_emp.shape[1]).to(device)#0.1).to(device)

        scores_df, test_accuracy, net  = train_net(
            net, 
            train_dataloader=train_dataloader, 
            valid_dataloader=valid_dataloader,
            test__dataloader=test__dataloader, 
            lr=lr, 
            max_epochs=epochs, 
            max_epochs_without_improvement=epochs,
        )
        
        accuracy[i_fold] = test_accuracy
        score_dfs.append(scores_df)
        
    #full_acc = np.mean(accuracy)
    return score_dfs, accuracy, net

##########################################
#The version where train&test is determined BEFORE splicing
##########################################

def do_EEGNet_2emp_2(device,sGen,emp_sub_list=[],n_folds = 10,epochs=5,F1=8,D=2,
        F2=16,drop=0.1,lr=5e-3,splices=5,allEps=True,batch_size=10):
    
    X_emp,y_emp,V,M,fs,n_classes,n_samples_transient = sGen.get_needed_values()
    if len(emp_sub_list) != 0:
        X_emp = X_emp[emp_sub_list]
        y_emp = y_emp[emp_sub_list]

    X_emp = np.concatenate([x for x in X_emp],axis=2)
    y_emp = np.concatenate([y for y in y_emp],axis=0)

    #X_emp,y_emp = splice_data(X_emp,y_emp,splices)

    #Participants X Channels X Samples X trials
    X_emp = np.transpose(X_emp,(2,0,1)) #X_emp becomes trials*participants X channels X samples
    
    n_trials = len(X_emp)
    
    #folds = np.repeat(np.arange(n_folds), n_trials / n_folds)
    folds = np.resize(np.arange(n_folds), n_trials)

    # Loop folds
    accuracy = np.zeros(n_folds)
    score_dfs = []

    if allEps:
        toEp = n_folds
    else:
        toEp = 1

    #Need to switch this to only doing the first fold, for testing.
    for i_fold in tqdm(range(toEp)):#tqdm(range(n_folds)):
        
        train_X = X_emp[folds != i_fold,:,:]
        valid_X = X_emp[folds == i_fold,:,:]
        test_X = X_emp[folds == i_fold,:,:]
        train_y = y_emp[folds != i_fold]
        valid_y = y_emp[folds == i_fold]
        test_y = y_emp[folds == i_fold]

        train_X,train_y = splice_data(train_X,train_y,splices)
        valid_X,valid_y = splice_data(valid_X,valid_y,splices)
        test_X,test_y = splice_data(test_X,test_y,splices)

        #test__X_tensor = torch.tensor(X_emp,dtype=torch.float32, device=device)
        train_X_tensor = torch.tensor(train_X, dtype=torch.float32, device=device) 
        test_X_tensor = torch.tensor(test_X, dtype=torch.float32, device=device) 

        train_y_tensor = torch.tensor(train_y, device=device) 
        test_y_tensor =  torch.tensor(test_y, device=device) 

        train_y_tensor = train_y_tensor.type(torch.LongTensor)
        test_y_tensor = test_y_tensor.type(torch.LongTensor)

        train_dataloader = DataLoader(TensorDataset(train_X_tensor, train_y_tensor), batch_size=batch_size, drop_last=False)
        valid_dataloader = DataLoader(TensorDataset(test_X_tensor, test_y_tensor), batch_size=batch_size, drop_last=False)
        test__dataloader = DataLoader(TensorDataset(test_X_tensor, test_y_tensor), batch_size=batch_size, drop_last=False)

        net = EEGNet(20,F1=F1,F2=F2,D=D, drop_prob=drop,inp_length = train_X.shape[2],in_chans=train_X.shape[1]).to(device)#0.1).to(device)

        scores_df, test_accuracy, net  = train_net(
            net, 
            train_dataloader=train_dataloader, 
            valid_dataloader=valid_dataloader,
            test__dataloader=test__dataloader, 
            lr=lr, 
            max_epochs=epochs, 
            max_epochs_without_improvement=epochs,
        )
        
        accuracy[i_fold] = test_accuracy
        score_dfs.append(scores_df)
        
    #full_acc = np.mean(accuracy)
    return score_dfs, accuracy, net

###############################################

###############################################

def get_ind_accuracies(net,indices,sGen,device):
    ind_accuracies = np.zeros(len(indices))
    X_emp,y_emp,V,M,fs,n_classes,n_samples_transient = sGen.get_needed_values()
    counter=0
    for cur_ind in indices:
        X_curTest = X_emp[[cur_ind]]
        y_curTest = y_emp[[cur_ind]]

        X_curTest = np.concatenate([x for x in X_curTest],axis=2)
        y_curTest = np.concatenate([y for y in y_curTest],axis=0)

        X_curTest = np.transpose(X_curTest,(2,0,1))
        
        test__X_tensor = torch.tensor(X_curTest,dtype=torch.float32, device=device)
        test__y_tensor = torch.tensor(y_curTest, device=device)
        test__y_tensor = test__y_tensor.type(torch.LongTensor)
        
        batch_size=10
        
        test__dataloader = DataLoader(TensorDataset(test__X_tensor, test__y_tensor), batch_size=batch_size, drop_last=False)
        
        # test step :
        net.eval()
        accuracy_accumulator = 0
        count = 0
        with torch.no_grad():
            for batch in test__dataloader:
                X,y = batch
                y_pred = net(X)
                accuracy_accumulator += torch.sum(y==y_pred.argmax(dim=1)).detach().item()
                count += y.shape[0]
        test_accuracy = accuracy_accumulator/count
        
        ind_accuracies[counter] = test_accuracy
        counter = counter+1
    return ind_accuracies

