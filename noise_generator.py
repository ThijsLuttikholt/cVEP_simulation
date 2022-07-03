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

#Make more generic, ensure that generating noise can be done, and that new functions of noise generation can be added
#Also add option to get the frequency domain of the noise
class noiseGen():

    def __init__(self,meansName='noiseMeans_exp1.npy',stdsName='noiseStds_exp1.npy'):
        self.paramMeans = np.load(meansName)
        self.paramStds = np.load(stdsName)
        argPink = {"sizeof":[1,1],"mask":[],"exponent":1}#1.7}
        argGauss = {"sizeof":[1,1]}
        argSpike = {"sizeof":[1,1],"fs":120}
        argAlpha = {"sizeof":[1,1],"fs":120}

        self.noise_params = {"types":[self.genPink,self.genGauss,self.genFreqSpike,self.genAlpha],
                   "weights":[18/40,2/40,17/40,3/40],
                   "params":[argPink,argGauss,argSpike,argAlpha],
                   "channels":1}

    def genPink(self,sizeof,mask=[],exponent=1):#1.7):
        
        noise = cn.powerlaw_psd_gaussian(exponent, [sizeof[0],sizeof[1]]) #This contains random parts
        
        if len(mask) > 0:
            noise = noise * np.tile(mask, (sizeof[0],sizeof[1]))

        return noise #Need to adapt this, only works with 1 channel now (used to be noise[0])

    def genGauss(self,sizeof,mask=[],mean=[],var=1):
        
        noise = np.random.randn(sizeof[0],sizeof[1]) * math.sqrt(var) #Need to adapt this, WHY?
        
        if len(mean) > 0:
            if len(mean[0]) == 1:
                noise = noise + np.tile(mean, sizeof)
            else:
                noise = noise + np.tile(mean, [sizeof[0],1])
        
        if len(mask) > 0:
            noise = noise * np.tile(mask, (1,sizeof[1]))

        return noise
    
    def genAlpha(self,sizeof,mask=[],band=[8.5,12],fs=120):
        gauss = self.genGauss(sizeof)
        b,a = signal.butter(3,band,'bandpass',fs=fs)
        noise = signal.lfilter(b,a,gauss)
        return noise

    def genFreqSpike(self,sizeof, mask=[], peakFreq=50, harmonics=False, fs=120): #What does harmonics indicate?
        timeElapsed = np.array(range(sizeof[1])).T / fs
        
        frequencySpike = [math.sin(2 * math.pi * time * peakFreq ) for time in timeElapsed]
        if harmonics:
            power = 0.1
            count = 2
            while peakFreq * count <= fs/2:
                frequencySpike = frequencySpike+power * np.array(
                    [math.sin(2 * math.pi * time * peakFreq * count) for time in timeElapsed])
                power = power**2
                count = count+1
        
        if len(mask)>0:
            frequencySpike = mask * frequencySpike
        
        noise = np.tile(frequencySpike, (1, sizeof[0]))
        return noise

    def convSame(self,vec1,vec2):
        npad = len(vec2) - 1
        full = np.convolve(vec1, vec2, 'full')
        first = npad - npad//2
        result = full[first:first+len(vec1)]
        return result

    #Adapt this function
    def genBlink(self,sizeof,dataset=[],cOrient=True,mask=[10,30,50],eyeblinkStd1=0.020,eyeblinkStd2=0.031,EBamp=5,fs=120):
        #blink standard deviations are in seconds
        noisePower = np.zeros((sizeof[0],1))
        maxStd = max(eyeblinkStd1,eyeblinkStd2)
        sampleStep = 1/fs
        lp = np.arange(-10*maxStd,10*maxStd+sampleStep,sampleStep)
        blinkIndexes = np.argwhere(np.array(mask)>0)#find(mask)
        blinkMaxPower = np.zeros((len(blinkIndexes),1))
        for i in range(len(blinkIndexes)):
            pattern = (lp+np.random.randn(1)*eyeblinkStd1)* np.exp(-(lp*lp)/(2*eyeblinkStd2**2))
            thisBlink = np.zeros((sizeof[0],1)) #This seems to be wrong too
            thisBlink[blinkIndexes[i]] = 1 
            print(noisePower.shape)
            intermediate = self.convSame(thisBlink[i], pattern)#This goes wrong
            print(intermediate.shape)
            noisePower += intermediate
            blinkMaxPower[i] = max(np.absolute(pattern))
            
        avgBlinkMaxPower = np.mean(blinkMaxPower)
        if avgBlinkMaxPower>0:
            noisePower = noisePower * (EBamp / avgBlinkMaxPower)
        
        if cOrient:
            noiseMultiplier = np.ones((1,sizeof[1]))
        else:
            #% If we're generating for a free dipole orientation case, weight directions differently
            assert sizeof[1]%3 == 0
            noiseMultiplier = np.tile([1,0,10],[1,sizeof[1]/3])
        
        noise = np.random.randn(sizeof[0],sizeof[1]) * np.tile(noisePower, [1,sizeof[1]]) * np.tile(noiseMultiplier , [sizeof[0], 1])
        return noise[0]

    def genEyeMov(self,sizeof, dataset=[],cOrient=True,physMod='core_head',mask=[10],movAmp=1): #Need to adapt the mask
        if cOrient:
            noiseMultiplier = np.ones((1,sizeof[1]))
        else:
            assert sizeof[1]%3 == 0
            noiseMultiplier = np.tile([1,0,10],[1,sizeof[1]/3])
        
        noise = np.random.randn(sizeof[0],sizeof[1]) * np.tile(
            mask,[1,sizeof[1]]) * np.tile(noiseMultiplier, [sizeof[0],1])
        return noise[0]

    def genLogU(self,sizeof,mask=[]):
        noise = np.sign(np.random.rand(sizeof[0],sizeof[1])-0.5) * np.log(np.random.rand(sizeof[0],sizeof[1]))
        if len(mask)>0:
            noise = noise*np.tile(mask,[1,sizeof[1]])
        return noise


    #This function needs to be made more generic, should accept any noise type
    #Should also take into account individual noise weights and normalize
    def genNoise(self,types, weights,params, noiseLength,channels=1,mult=2e-5):
        assert len(types)==len(weights)
        assert len(types)==len(params)
        #assert np.sum(weights) == 1
        
        fullNoise = np.zeros((channels, noiseLength))
        
        for ind,ntype in enumerate(types):
            currentNoise = ntype(**params[ind])
            stDev = np.std(currentNoise)
            
            #Already has the correct relative weighting, but why sqrt(1/weight)  ?
            #currentNoise = currentNoise * (math.sqrt(1/snrs[ind])/stDev) 
            currentNoise = currentNoise /stDev * weights[ind]
            
            #Should we merely sum or average them? Summing pbb, weights should count up to 1
            fullNoise += currentNoise
            
        #Maybe the fullNoise should still be averaged or something in the end?
        fullNoise = fullNoise * mult
        
        return fullNoise
    
    def genNoiseN(self,n,types,weights,params,noiseLength,channels=1):
        noises = np.array([self.genNoise(types,weights,params,noiseLength,channels) for i in range(n)])
        return noises
    
    def drawN(self,n,tlength,channels=1):
        noises=np.zeros((n,channels,tlength))
        draws = self.drawNoiseParams(n)
        thisNoisePar = self.noise_params
        thisNoisePar["noiseLength"] = tlength
        #thisNoisePar["n"] = n
        
        for item in thisNoisePar["params"]:
            item["sizeof"] = [item["sizeof"][0],tlength]
        
        for i in range(n):
            curNoisePar = thisNoisePar
            curNoisePar["params"][0]["exponent"] = draws[0,i]
            curNoisePar["weights"] = draws[1:5,i]
            
            curNoise = self.genNoise(**curNoisePar)
            curNoise *= draws[5,i]
            #print(curNoise.shape)
            noises[i,:] = curNoise
            
        return noises
    
    def drawNoiseParams(self,n):
        values = np.zeros((6,n))
        for i in range(6):
            values[i,:] = norm.rvs(size=n,loc=self.paramMeans[i],scale=self.paramStds[i])
        return values