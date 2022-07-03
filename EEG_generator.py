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

from scipy.stats import (
    norm, beta, expon, gamma, genextreme, logistic, lognorm, triang, uniform, fatiguelife,            
    gengamma, gennorm, dweibull, dgamma, gumbel_r, powernorm, rayleigh, weibull_max, weibull_min, 
    laplace, alpha, genexpon, bradford, betaprime, burr, fisk, genpareto, hypsecant, 
    halfnorm, halflogistic, invgauss, invgamma, levy, loglaplace, loggamma, maxwell, 
    mielke, ncx2, ncf, nct, nakagami, pareto, lomax, powerlognorm, powerlaw, rice, 
    semicircular, trapezoid, rice, invweibull, foldnorm, foldcauchy, cosine, exponpow, 
    exponweib, wald, wrapcauchy, truncexpon, truncnorm, t, rdist
    )


class EEG_generator():
    def __init__(self,nGen,sGen):
        self.nGen = nGen
        self.sGen = sGen
        self.codes = sGen.getCodes()
        self.durations = sGen.getDurations()
        
    #noise_params keywords: types;weights;params  which are functions, weights, and function parameters respectively
    #function parameter keywords: At least the "sizeof" attribute, as a list with 1+ element
    def genN(self,n,snrs,noise_params, maxRange=1,subjects=[],code_times=15, cutOff=36,processor=None):
        channels=1
        signals = self.sGen.getN(n,subjects) #Randomly get N template pairs
        #code_indices = random.choices([i for i in range(len(self.codes[0]))],k=n)
        code_indices = np.resize(np.arange(len(self.codes[0])),n)
        np.random.shuffle(code_indices)
        
        codes = np.array([np.tile(self.codes[:,i],code_times) for i in code_indices])
        #Need to repeat codes 15 times and cut off signal length at the end if needed
        intermediate = np.array([self.overlay2(signals[i],codes[i]) for i in range(n)])
        
        responses = np.array([x[:len(x)-cutOff] for x in intermediate]) #This cuts off values at the end.
        
        tlength = len(responses[0])
        
        noise_params["noiseLength"] = tlength
        noise_params["n"] = n
        
        for item in noise_params["params"]:
            item["sizeof"] = [item["sizeof"][0],tlength]
        
        noises = self.nGen.genNoiseN(**noise_params)
        
        if processor != None:
            noises = np.transpose(noises,(1,2,0))
            noises = processor.process(noises)
            noises = np.transpose(noises,(2,0,1))
        
        combinations = [noises[i]/np.std(noises[i])+responses[i]/np.std(responses[i])*snrs[i] for i in range(n)]
        
        combinations = np.transpose(combinations,(1,2,0))
        combinations = combinations * 0.00002
        return codes,np.array(code_indices),np.array(combinations)
    
    #Drawing only the noise
    def genNDrawn1(self,n,snrs,subjects=[],code_times=15,cutOff=36,processor=None):
        channels=1
        signals = self.sGen.getN(n,subjects) #Randomly get N template pairs
        #code_indices = random.choices([i for i in range(len(self.codes[0]))],k=n)
        code_indices = np.resize(np.arange(len(self.codes[0])),n)
        np.random.shuffle(code_indices)
        
        codes = np.array([np.tile(self.codes[:,i],code_times) for i in code_indices])
        #Need to repeat codes 15 times and cut off signal length at the end if needed
        intermediate = np.array([self.overlay2(signals[i],codes[i]) for i in range(n)])
        
        responses = np.array([x[:len(x)-cutOff] for x in intermediate]) #This cuts off values at the end.
        
        tlength = len(responses[0])
        
        
        noises = self.nGen.drawN(n,tlength)
        
        if processor != None:
            noises = np.transpose(noises,(1,2,0))
            noises = processor.process(noises)
            noises = np.transpose(noises,(2,0,1))
        
        combinations = [noises[i]/np.std(noises[i])+responses[i]/np.std(responses[i])*snrs[i] for i in range(n)]
        
        combinations = np.transpose(combinations,(1,2,0))
        combinations = combinations * 0.00002
        return codes,np.array(code_indices),np.array(combinations)

    #Drawing both the signal and the noise, using original double Gamma
    def genNDrawn2(self,n,snrs,code_times=15,cutOff=36,processor=None):
        channels=1
        #signals = self.sGen.getN(n,subjects) #Randomly get N template pairs
        signals = self.sGen.drawN(n)
        
        #code_indices = random.choices([i for i in range(len(self.codes[0]))],k=n)
        code_indices = np.resize(np.arange(len(self.codes[0])),n)
        np.random.shuffle(code_indices)
        
        codes = np.array([np.tile(self.codes[:,i],code_times) for i in code_indices])
        #Need to repeat codes 15 times and cut off signal length at the end if needed
        intermediate = np.array([self.overlay2(signals[i],codes[i]) for i in range(n)])
        
        responses = np.array([x[:len(x)-cutOff] for x in intermediate]) #This cuts off values at the end.
        
        tlength = len(responses[0])
        
        
        noises = self.nGen.drawN(n,tlength)
        
        if processor != None:
            noises = np.transpose(noises,(1,2,0))
            noises = processor.process(noises)
            noises = np.transpose(noises,(2,0,1))
        
        combinations = [noises[i]/np.std(noises[i])+responses[i]/np.std(responses[i])*snrs[i] for i in range(n)]
        
        combinations = np.transpose(combinations,(1,2,0))
        combinations = combinations * 0.00002
        return codes,np.array(code_indices),np.array(combinations)

    #Drawing both the signal and the noise, using relative double Gamma
    def genNDrawn2_2(self,n,snrs,code_times=15,cutOff=36,processor=None):
        channels=1
        #signals = self.sGen.getN(n,subjects) #Randomly get N template pairs
        signals = self.sGen.drawN2(n)
        
        #code_indices = random.choices([i for i in range(len(self.codes[0]))],k=n)
        code_indices = np.resize(np.arange(len(self.codes[0])),n)
        np.random.shuffle(code_indices)
        
        codes = np.array([np.tile(self.codes[:,i],code_times) for i in code_indices])
        #Need to repeat codes 15 times and cut off signal length at the end if needed
        intermediate = np.array([self.overlay2(signals[i],codes[i]) for i in range(n)])
        
        responses = np.array([x[:len(x)-cutOff] for x in intermediate]) #This cuts off values at the end.
        
        tlength = len(responses[0])
        
        
        noises = self.nGen.drawN(n,tlength)
        
        if processor != None:
            noises = np.transpose(noises,(1,2,0))
            noises = processor.process(noises)
            noises = np.transpose(noises,(2,0,1))
        
        combinations = [noises[i]/np.std(noises[i])+responses[i]/np.std(responses[i])*snrs[i] for i in range(n)]
        
        combinations = np.transpose(combinations,(1,2,0))
        combinations = combinations * 0.00002
        return codes,np.array(code_indices),np.array(combinations)

    #Drawing both the signal and the noise, using the custom signal representation
    def genNDrawn2_3(self,n,snrs,code_times=15,cutOff=36,processor=None):
        channels=1
        #signals = self.sGen.getN(n,subjects) #Randomly get N template pairs
        signals = self.sGen.drawN3(n)
        
        #code_indices = random.choices([i for i in range(len(self.codes[0]))],k=n)
        code_indices = np.resize(np.arange(len(self.codes[0])),n)
        np.random.shuffle(code_indices)
        
        codes = np.array([np.tile(self.codes[:,i],code_times) for i in code_indices])
        #Need to repeat codes 15 times and cut off signal length at the end if needed
        intermediate = np.array([self.overlay2(signals[i],codes[i]) for i in range(n)])
        
        responses = np.array([x[:len(x)-cutOff] for x in intermediate]) #This cuts off values at the end.
        
        tlength = len(responses[0])
        
        
        noises = self.nGen.drawN(n,tlength)
        
        if processor != None:
            noises = np.transpose(noises,(1,2,0))
            noises = processor.process(noises)
            noises = np.transpose(noises,(2,0,1))
        
        combinations = [noises[i]/np.std(noises[i])+responses[i]/np.std(responses[i])*snrs[i] for i in range(n)]
        
        combinations = np.transpose(combinations,(1,2,0))
        combinations = combinations * 0.00002
        return codes,np.array(code_indices),np.array(combinations)

    #Drawing both the signal and the noise, using the custom signal representation
    def genNDrawn2_3_rel(self,n,snrs,code_times=15,cutOff=36,processor=None):
        channels=1
        #signals = self.sGen.getN(n,subjects) #Randomly get N template pairs
        signals = self.sGen.drawN3_rel(n)
        
        #code_indices = random.choices([i for i in range(len(self.codes[0]))],k=n)
        code_indices = np.resize(np.arange(len(self.codes[0])),n)
        np.random.shuffle(code_indices)
        
        codes = np.array([np.tile(self.codes[:,i],code_times) for i in code_indices])
        #Need to repeat codes 15 times and cut off signal length at the end if needed
        intermediate = np.array([self.overlay2(signals[i],codes[i]) for i in range(n)])
        
        responses = np.array([x[:len(x)-cutOff] for x in intermediate]) #This cuts off values at the end.
        
        tlength = len(responses[0])
        
        
        noises = self.nGen.drawN(n,tlength)
        
        if processor != None:
            noises = np.transpose(noises,(1,2,0))
            noises = processor.process(noises)
            noises = np.transpose(noises,(2,0,1))
        
        combinations = [noises[i]/np.std(noises[i])+responses[i]/np.std(responses[i])*snrs[i] for i in range(n)]
        
        combinations = np.transpose(combinations,(1,2,0))
        combinations = combinations * 0.00002
        return codes,np.array(code_indices),np.array(combinations)
    
    def overlay2(self,signals,code):
        result = np.zeros(len(code)+len(signals[0]))
        for ind,val in enumerate(code):
            if ind==0 and val>0:
                duration = self.find_duration_idx(code,ind)
                result[ind:ind+len(signals[duration])] += signals[duration]
                continue
            elif ind==0 and val==0:
                continue

            if val>0 and code[ind-1]==0:
                duration = self.find_duration_idx(code,ind)
                result[ind:ind+len(signals[duration])] += signals[duration]
        return result
    
    def find_duration_idx(self,code,index):
        count = 0
        for i in range(index,len(code)):
            if code[i]>0:
                count += 1
            else:
                return np.where(self.durations==count)[0][0]
        return np.where(self.durations==count)[0][0]