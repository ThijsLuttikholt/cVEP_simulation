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

import colorednoise as cn

from .extra_functions import splice_data

from scipy.stats import (
    norm, beta, expon, gamma, genextreme, logistic, lognorm, triang, uniform, fatiguelife,            
    gengamma, gennorm, dweibull, dgamma, gumbel_r, powernorm, rayleigh, weibull_max, weibull_min, 
    laplace, alpha, genexpon, bradford, betaprime, burr, fisk, genpareto, hypsecant, 
    halfnorm, halflogistic, invgauss, invgamma, levy, loglaplace, loggamma, maxwell, 
    mielke, ncx2, ncf, nct, nakagami, pareto, lomax, powerlognorm, powerlaw, rice, 
    semicircular, trapezoid, rice, invweibull, foldnorm, foldcauchy, cosine, exponpow, 
    exponweib, wald, wrapcauchy, truncexpon, truncnorm, t, rdist
    )

def train_net(
    net, # the network being trained
    train_dataloader, # the training data
    valid_dataloader, # the validation data
    test__dataloader, # the test data
    lr, # the learning rate
    max_epochs=100, # maximum number of pass through the whole training dataset
    max_epochs_without_improvement = 10, # parameter used by the early stopping mechanism
):
    #optimizer = torch.optim.SGD(params=net.parameters(), lr=lr)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    scores = []
    num_activations = 0
    best_valid_accuracy = None
    best_net_params = None
    best_epoch = -1
    valid_accs = np.zeros(max_epochs)
    for epoch in tqdm(range(max_epochs), leave=False, desc='epochs'): 
        # training for one full epoch :
        #net.train()
        accuracy_accumulator = 0
        count = 0
        loss_accumulator = 0.
        for batch in train_dataloader:
            X,y = batch
            optimizer.zero_grad() #Why is this not done earlier?
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
        #net.eval()
        accuracy_accumulator = 0
        count = 0
        loss_accumulator = 0.
        with torch.no_grad():
            #for batch in tqdm(valid_dataloader, leave=False, desc='valid batches'):
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
            num_activations += 1
            best_valid_accuracy = last_valid_accuracy
            best_epoch = epoch
            best_net_params = deepcopy(net.state_dict())
        stop_condition = epoch-best_epoch>max_epochs_without_improvement
        if stop_condition:
            break

    # test step :
    net.load_state_dict(best_net_params)
    #net.eval()
    accuracy_accumulator = 0
    count = 0
    with torch.no_grad():
        for batch in test__dataloader:
            X,y = batch
            y_pred = net(X)
            accuracy_accumulator += torch.sum(y==y_pred.argmax(dim=1)).detach().item()
            count += y.shape[0]
    test_accuracy = accuracy_accumulator/count
    
    # prepare scores DataFrame :
    scores_df = pd.DataFrame(scores)
    scores_df['F1'] = net.F1
    scores_df['D'] = net.D
    scores_df['F2'] = net.F2
    scores_df['drop_prob'] = net.drop_prob
    scores_df['lr'] = lr
    scores_df['max_epochs'] = max_epochs
    scores_df['max_epochs_without_improvement'] = max_epochs_without_improvement
    scores_df['test_accuracy'] = test_accuracy

    return scores_df, test_accuracy, net



def train_net2(
    net, # the network being trained
    device,
    model_ind,
    nGen,
    gener,
    perRound,
    sim_sub_list,
    splices,
    batch_size,
    processor,
    valid_dataloader, # the validation data
    test__dataloader, # the test data
    lr, # the learning rate
    max_epochs=100, # maximum number of pass through the whole training dataset
    max_epochs_without_improvement = 10, # parameter used by the early stopping mechanism
):
    snr = 0.68
    snrs1 = [snr for i in range(perRound)]

    argPink = {"sizeof":[1,1],"mask":[],"exponent":1}#1.7}
    argGauss = {"sizeof":[1,1]}
    argSpike = {"sizeof":[1,1],"fs":120}
    argAlpha = {"sizeof":[1,1],"fs":120}

    noise_params = {"types":[nGen.genPink,nGen.genGauss,nGen.genFreqSpike,nGen.genAlpha],
                   "weights":[18/40,2/40,17/40,3/40],
                   "params":[argPink,argGauss,argSpike,argAlpha],
                   "channels":1}

    #optimizer = torch.optim.SGD(params=net.parameters(), lr=lr)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    scores = []
    best_valid_accuracy = None
    best_net_params = None
    best_epoch = -1
    for epoch in tqdm(range(max_epochs), leave=False, desc='epochs'): 
        #Getting new samples
        if model_ind == 1:
            codesT,targetsT,resultsT = gener.genN(perRound,snrs1,noise_params,maxRange=1,subjects=sim_sub_list,processor=processor)
        elif model_ind == 2:
            codesT,targetsT,resultsT = gener.genNDrawn1(perRound,snrs1,subjects=sim_sub_list,processor=processor)
        elif model_ind == 3:
            codesT,targetsT,resultsT = gener.genNDrawn2(perRound,snrs1,processor=processor)
        elif model_ind == 4:
            codesT,targetsT,resultsT = gener.genNDrawn2_2(perRound,snrs1,processor=processor)
        elif model_ind == 5:
            codesT,targetsT,resultsT = gener.genNDrawn2_3(perRound,snrs1,processor=processor)

        resultsT = processor.process(resultsT)
        resultsT = np.transpose(resultsT,(2,0,1))
        resultsT = resultsT * 0.00002
        
        resultsT,targetsT = splice_data(resultsT,targetsT,splices)

        train_X_tensor = torch.tensor(resultsT,dtype=torch.float32, device=device)
        train_y_tensor = torch.tensor(targetsT, device=device)
        train_y_tensor = train_y_tensor.type(torch.LongTensor)
        train_dataloader = DataLoader(TensorDataset(train_X_tensor, train_y_tensor), batch_size=batch_size, drop_last=False)
        
        
        # training for one full epoch :
        #net.train()
        accuracy_accumulator = 0
        count = 0
        loss_accumulator = 0.
        for batch in train_dataloader:
            X,y = batch
            y_pred = net(X)
            loss = loss_func(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            accuracy_accumulator += torch.sum(y==y_pred.argmax(dim=1)).detach().item()
            loss_accumulator += loss.detach().item()
            count += y.shape[0]
        scores.append(dict(epoch=epoch, phase='train', metric='accuracy', value=accuracy_accumulator/count))
        scores.append(dict(epoch=epoch, phase='train', metric='loss',     value=loss_accumulator/count))
        
        # validation step :
        #net.eval()
        accuracy_accumulator = 0
        count = 0
        loss_accumulator = 0.
        with torch.no_grad():
            #for batch in tqdm(valid_dataloader, leave=False, desc='valid batches'):
            for batch in valid_dataloader:
                X,y = batch
                y_pred = net(X)
                loss = loss_func(y_pred, y)
                accuracy_accumulator += torch.sum(y==y_pred.argmax(dim=1)).detach().item()
                loss_accumulator += loss.detach().item()
                count += y.shape[0]
        last_valid_accuracy = accuracy_accumulator/count
        scores.append(dict(epoch=epoch, phase='valid', metric='accuracy', value=last_valid_accuracy))
        scores.append(dict(epoch=epoch, phase='valid', metric='loss',     value=loss_accumulator/count))
        
        # early stopping step :
        update_condition = (best_valid_accuracy==None or last_valid_accuracy>best_valid_accuracy)# <to be completed in 3.5>
        if update_condition:
            best_valid_accuracy = last_valid_accuracy
            best_epoch = epoch
            best_net_params = deepcopy(net.state_dict())
        stop_condition = epoch-best_epoch>max_epochs_without_improvement# <to be completed in 3.5>
        if stop_condition:
            break

    # test step :
    net.load_state_dict(best_net_params)
    #net.eval()
    accuracy_accumulator = 0
    count = 0
    with torch.no_grad():
        for batch in test__dataloader:
            X,y = batch
            y_pred = net(X)
            accuracy_accumulator += torch.sum(y==y_pred.argmax(dim=1)).detach().item()
            count += y.shape[0]
    test_accuracy = accuracy_accumulator/count
    
    # prepare scores DataFrame :
    scores_df = pd.DataFrame(scores)
    scores_df['F1'] = net.F1
    scores_df['D'] = net.D
    scores_df['F2'] = net.F2
    scores_df['drop_prob'] = net.drop_prob
    scores_df['lr'] = lr
    scores_df['max_epochs'] = max_epochs
    scores_df['max_epochs_without_improvement'] = max_epochs_without_improvement
    scores_df['test_accuracy'] = test_accuracy

    return scores_df, test_accuracy, net


############################################################
# Training the network with a higher batch size
############################################################

def train_net_highB(
    net, # the network being trained
    train_dataloader, # the training data
    valid_dataloader, # the validation data
    test__dataloader, # the test data
    lr, # the learning rate
    max_epochs=100, # maximum number of pass through the whole training dataset
    max_epochs_without_improvement = 10, # parameter used by the early stopping mechanism
):
    #optimizer = torch.optim.SGD(params=net.parameters(), lr=lr)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    scores = []
    num_activations = 0
    best_valid_accuracy = None
    best_net_params = None
    best_epoch = -1
    valid_accs = np.zeros(max_epochs)
    for epoch in tqdm(range(max_epochs), leave=False, desc='epochs'): 
        # training for one full epoch :
        net.train()
        accuracy_accumulator = 0
        count = 0
        loss_accumulator = 0.
        for batch in train_dataloader:
            X,y = batch
            optimizer.zero_grad() #Why is this not done earlier?
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
        net.eval()
        accuracy_accumulator = 0
        count = 0
        loss_accumulator = 0.
        with torch.no_grad():
            #for batch in tqdm(valid_dataloader, leave=False, desc='valid batches'):
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
            num_activations += 1
            best_valid_accuracy = last_valid_accuracy
            best_epoch = epoch
            best_net_params = deepcopy(net.state_dict())
        stop_condition = epoch-best_epoch>max_epochs_without_improvement
        if stop_condition:
            break

    # test step :
    net.load_state_dict(best_net_params)
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
    
    # prepare scores DataFrame :
    scores_df = pd.DataFrame(scores)
    scores_df['F1'] = net.F1
    scores_df['D'] = net.D
    scores_df['F2'] = net.F2
    scores_df['drop_prob'] = net.drop_prob
    scores_df['lr'] = lr
    scores_df['max_epochs'] = max_epochs
    scores_df['max_epochs_without_improvement'] = max_epochs_without_improvement
    scores_df['test_accuracy'] = test_accuracy

    return scores_df, test_accuracy, net


############################################################
# Doing the splice-to-full classification
############################################################

def eval_split(device,sGen,net,emp_sub_list,splices,batch_size):
    X_emp,y_emp,V,M,fs,n_classes,n_samples_transient = sGen.get_needed_values()
    if len(emp_sub_list) != 0:
        X_emp = X_emp[emp_sub_list]
        y_emp = y_emp[emp_sub_list]

    X_emp = np.concatenate([x for x in X_emp],axis=2)
    y_emp = np.concatenate([y for y in y_emp],axis=0)

    X_emp = np.transpose(X_emp,(2,0,1))
    
    X_emp2,y_emp2 = splice_data(X_emp,y_emp,splices)
    
    test__X_tensor = torch.tensor(X_emp2,dtype=torch.float32, device=device)
    test__y_tensor = torch.tensor(y_emp2, device=device)
    test__y_tensor = test__y_tensor.type(torch.LongTensor)

    test__dataloader = DataLoader(TensorDataset(test__X_tensor, test__y_tensor), batch_size=batch_size, drop_last=False)
    
    all_predictions = np.zeros(X_emp2.shape[0])
    count=0
    with torch.no_grad():
        for batch in test__dataloader:
            X,y = batch
            y_pred = net(X)
            predictions = y_pred.argmax(dim=1).detach().numpy()
            #predictions = predictions.type(torch.LongTensor)
            all_predictions[count:count+y.shape[0]] = predictions
            count += y.shape[0]
    
    return X_emp,y_emp,all_predictions

def get_scores(X_emp,y_emp,all_predictions,splices):
    accuracies = np.zeros(X_emp.shape[0])
    full_preds = np.zeros(X_emp.shape[0])
    all_predictions = all_predictions.astype(int)
    for i in range(X_emp.shape[0]):
        cur_preds = all_predictions[i*splices:i*splices+splices]
        pred = np.bincount(cur_preds).argmax()
        full_preds[i] = pred
        if pred == y_emp[i]:
            accuracies[i] = 1
        else:
            accuracies[i] = 0
    return accuracies,full_preds

def get_full_accs(device,sGen,net,emp_sub_list,splices,batch_size=10):
    X_emp,y_emp,all_predictions = eval_split(device,sGen,net,emp_sub_list,splices,batch_size)
    accuracies,full_preds = get_scores(X_emp,y_emp,all_predictions,splices)
    return accuracies,full_preds