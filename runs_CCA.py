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

from .extra_functions import splice_data

import pandas as pd
import torch
from torch import nn
from torch.nn.functional import elu
from torch.utils.data import TensorDataset, DataLoader

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


def do_CCA_sim(sGen,nGen,gener,processor=None,processor2=None,sim_sub_list=[],n_trials=1000):
    X_emp,y_emp,V,M,fs,n_classes,n_samples_transient = sGen.get_needed_values()
    

    argPink = {"sizeof":[1,1],"mask":[],"exponent":1}#1.7}
    argGauss = {"sizeof":[1,1]}
    argSpike = {"sizeof":[1,1],"fs":120}
    argAlpha = {"sizeof":[1,1],"fs":120}

    noise_params = {"types":[nGen.genPink,nGen.genGauss,nGen.genFreqSpike,nGen.genAlpha],
                   "weights":[18/40,2/40,17/40,3/40],
                   "params":[argPink,argGauss,argSpike,argAlpha],
                   "channels":1}
    #argPink = {"sizeof":[1,1],"mask":[],"exponent":1.7}
    #argGauss = {"sizeof":[1,1]}
    #argSpike = {"sizeof":[1,1],"fs":120}

    #noise_params = {"types":[testNgen.genPink,testNgen.genGauss,testNgen.genFreqSpike],
     #              "weights":[1/3,1/3,1/3],
      #             "params":[argPink,argGauss,argSpike],
       #            "channels":1}

    n_channels = 1
    n_samples = int(2.1 * fs)

    #NEED TO SAMPLES SNRS BASED ON A GIVEN RANGE HERE
    lower, higher = 0.5, 1.5
    mu, sigma = 1, 0.2
    drawer = truncnorm((lower-mu)/sigma,(higher-mu)/sigma, loc=mu, scale=sigma)
    snrs = drawer.rvs(n_trials)
    
    codes_sim,y_sim,X_sim = gener.genN(n_trials,snrs,noise_params,maxRange=0.05,subjects=sim_sub_list,processor=processor2)
    if processor != None:
        X_sim = processor.process(X_sim)

    n_folds = 10

    cca = CCA(n_components=1)

    # Chronological folding
    folds = np.repeat(np.arange(n_folds), n_trials / n_folds)

    accuracy = np.zeros(n_folds)
    spatial_filters = np.zeros((n_folds, n_channels)) 
    temporal_filters = np.zeros((n_folds, 2*n_samples_transient))

    for i_fold in range(n_folds):#tqdm(range(n_folds)):

        # Split data to train and valid set
        X_train, y_train = X_sim[:, :n_samples, folds != i_fold], y_sim[folds != i_fold] 
        X_valid, y_valid = X_sim[:,:n_samples, folds != i_fold], y_sim[folds != i_fold]

        # -- TRAINING

        # Reshape the training data
        X_ = np.reshape(X_train, (n_channels, -1)).T
        M_ = np.tile(M, (np.ceil(n_samples/V.shape[0]), 1, 1))[:n_samples, :, :]
        M_ = M_[:, :, y_train]
        M_ = M_.transpose((1, 0, 2))
        M_ = np.reshape(M_, (2*n_samples_transient, -1)).T

        # Fit CCA using training data
        cca.fit(X_.astype("float32"), M_.astype("float32"))
        w = cca.x_weights_.flatten()
        r = cca.y_weights_.flatten()
        spatial_filters[i_fold, :] = w
        temporal_filters[i_fold, :] = r

        # Predict templates
        T = np.zeros((M.shape[0], n_classes))
        for i_class in range(n_classes):
            T[:, i_class] = np.dot(M[:, :, i_class], r)
        T = np.tile(T, (np.ceil(n_samples/V.shape[0]), 1))[:n_samples, :]

        # -- VALIDATION

        # Spatially filter validation data
        X_filtered = np.zeros((n_samples, y_valid.size))
        for i_trial in range(y_valid.size):
            X_filtered[:, i_trial] = np.dot(w, X_valid[:, :, i_trial])

        # Template matching
        prediction = np.zeros(y_valid.size)
        for i_trial in range(y_valid.size):
            rho = np.corrcoef(X_filtered[:, i_trial], T.T)[0, 1:]
            prediction[i_trial] = np.argmax(rho)

        # Compute accuracy
        accuracy[i_fold] = 100*np.mean(prediction == y_valid)

    #print("Trial: {:.1f} accuracy with a standard deviation of {:.2f}".format(accuracy.mean(), accuracy.std()))
    return accuracy.mean(),accuracy.std()


#######################################################

#######################################################

def do_CCA(sGen,nGen,gener,processor=None,processor2=None,emp_sub_list=[],sim_sub_list=[],n_trials=1000,splices=15):
    X_emp,y_emp,V,M,fs,n_classes,n_samples_transient = sGen.get_needed_values()

    if len(emp_sub_list) != 0:
        X_emp = X_emp[emp_sub_list]
        y_emp = y_emp[emp_sub_list]

    X_emp = np.concatenate([x for x in X_emp],axis=2)
    y_emp = np.concatenate([y for y in y_emp],axis=0)

    #X_emp = np.transpose(X_emp,(2,0,1))
    
    argPink = {"sizeof":[1,1],"mask":[],"exponent":1}#1.7}
    argGauss = {"sizeof":[1,1]}
    argSpike = {"sizeof":[1,1],"fs":120}
    argAlpha = {"sizeof":[1,1],"fs":120}

    noise_params = {"types":[nGen.genPink,nGen.genGauss,nGen.genFreqSpike,nGen.genAlpha],
                   "weights":[18/40,2/40,17/40,3/40],
                   "params":[argPink,argGauss,argSpike,argAlpha],
                   "channels":1}
    
    #argPink = {"sizeof":[1,1],"mask":[],"exponent":1.7}
    #argGauss = {"sizeof":[1,1]}
    #argSpike = {"sizeof":[1,1],"fs":120}

    #noise_params = {"types":[testNgen.genPink,testNgen.genGauss,testNgen.genFreqSpike],
    #               "weights":[1/3,1/3,1/3],
    #               "params":[argPink,argGauss,argSpike],
    #               "channels":1}

    #n_trials =200
    trial_length = 2.1*(15/splices)
    n_channels = 1
    n_samples = int(trial_length * fs)
    snr = 0.68
    
    #NEED TO SAMPLES SNRS BASED ON A GIVEN RANGE HERE
    lower, higher = 3,5#0.5, 1.5
    mu, sigma = 4, 0.7#0.2
    drawer = truncnorm((lower-mu)/sigma,(higher-mu)/sigma, loc=mu, scale=sigma)
    #snrs = drawer.rvs(n_trials)
    snrs = [snr for i in range(n_trials)]
    
    codes_sim,y_sim,X_sim = gener.genN(n_trials,snrs,noise_params,maxRange=0.05,subjects=sim_sub_list,processor=processor2)
    if processor != None:
        X_sim = processor.process(X_sim)

    X_sim = X_sim * 0.00002
    
    n_folds = 1

    cca = CCA(n_components=1)

    # Chronological folding
    folds = np.repeat(np.arange(n_folds), n_trials / n_folds)

    accuracy = np.zeros(n_folds)
    spatial_filters = np.zeros((n_folds, n_channels)) 
    temporal_filters = np.zeros((n_folds, 2*n_samples_transient))

    for i_fold in range(n_folds):

        # Split data to train and valid set
        X_train, y_train = X_sim[:, :n_samples, :], y_sim 
        X_valid, y_valid = X_emp[:,:n_samples,:], y_emp

        # -- TRAINING

        # Reshape the training data
        X_ = np.reshape(X_train, (n_channels, -1)).T
        M_ = np.tile(M, (np.ceil(n_samples/V.shape[0]), 1, 1))[:n_samples, :, :]
        M_ = M_[:, :, y_train]
        M_ = M_.transpose((1, 0, 2))
        M_ = np.reshape(M_, (2*n_samples_transient, -1)).T

        # Fit CCA using training data
        cca.fit(X_.astype("float32"), M_.astype("float32"))
        w = cca.x_weights_.flatten()
        r = cca.y_weights_.flatten()
        spatial_filters[i_fold, :] = w
        temporal_filters[i_fold, :] = r

        # Predict templates
        T = np.zeros((M.shape[0], n_classes))
        for i_class in range(n_classes):
            T[:, i_class] = np.dot(M[:, :, i_class], r)
        T = np.tile(T, (np.ceil(n_samples/V.shape[0]), 1))[:n_samples, :]

        # -- VALIDATION

        # Spatially filter validation data
        X_filtered = np.zeros((n_samples, y_valid.size))
        for i_trial in range(y_valid.size):
            X_filtered[:, i_trial] = np.dot(w, X_valid[:, :, i_trial])

        # Template matching
        prediction = np.zeros(y_valid.size)
        for i_trial in range(y_valid.size):
            rho = np.corrcoef(X_filtered[:, i_trial], T.T)[0, 1:]
            prediction[i_trial] = np.argmax(rho)

        # Compute accuracy
        accuracy[i_fold] = 100*np.mean(prediction == y_valid)

    #print("Trial: {:.1f} accuracy with a standard deviation of {:.2f}".format(accuracy.mean(), accuracy.std()))
    return accuracy.mean(),accuracy.std()

##########################################################

##########################################################

def do_CCA_empOnly(sGen,emp_sub_list=[],n_channels=1,n_folds=10,splices=1):
    X_emp,y_emp,V,M,fs,n_classes,n_samples_transient = sGen.get_needed_values()

    if len(emp_sub_list) != 0:
        X_emp = X_emp[emp_sub_list]
        y_emp = y_emp[emp_sub_list]

    X_emp = np.concatenate([x for x in X_emp],axis=2) #This is channels X samples X trials*participants
    y_emp = np.concatenate([y for y in y_emp],axis=0)
    y_emp = y_emp.astype(int)
    

    X_emp = np.transpose(X_emp,(2,0,1))

    X_emp,y_emp = splice_data(X_emp,y_emp,splices)

    y_emp = y_emp.astype(int)
    X_emp = np.transpose(X_emp,(1,2,0))

    trial_length = 2.1*(15/splices)
    n_trials = X_emp.shape[2]
    n_samples = int(trial_length * fs)

    cca = CCA(n_components=1)

    # Chronological folding
    #folds = np.repeat(np.arange(n_folds), n_trials / n_folds)
    folds = np.resize(np.arange(n_folds), n_trials)

    accuracy = np.zeros(n_folds)
    spatial_filters = np.zeros((n_folds, n_channels)) 
    temporal_filters = np.zeros((n_folds, 2*n_samples_transient))

    for i_fold in tqdm(range(n_folds)):

        # Split data to train and valid set
        X_train, y_train = X_emp[:, :n_samples, folds != i_fold], y_emp[folds != i_fold]
        X_valid, y_valid = X_emp[:, :n_samples, folds == i_fold], y_emp[folds == i_fold]

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
        spatial_filters[i_fold, :] = w
        temporal_filters[i_fold, :] = r

        # Predict templates
        T = np.zeros((M.shape[0], n_classes))
        for i_class in range(n_classes):
            T[:, i_class] = np.dot(M[:, :, i_class], r)
        T = np.tile(T, (int(np.ceil(n_samples/V.shape[0])), 1))[:n_samples, :]

        # -- VALIDATION

        # Spatially filter validation data
        X_filtered = np.zeros((n_samples, y_valid.size))
        for i_trial in range(y_valid.size):
            X_filtered[:, i_trial] = np.dot(w, X_valid[:, :, i_trial])

        # Template matching
        prediction = np.zeros(y_valid.size)
        for i_trial in range(y_valid.size):
            rho = np.corrcoef(X_filtered[:, i_trial], T.T)[0, 1:]
            prediction[i_trial] = np.argmax(rho)

        # Compute accuracy
        accuracy[i_fold] = 100*np.mean(prediction == y_valid)

    #print("Trial: {:.1f} accuracy with a standard deviation of {:.2f}".format(accuracy.mean(), accuracy.std()))
    return accuracy


#############################################

#############################################

