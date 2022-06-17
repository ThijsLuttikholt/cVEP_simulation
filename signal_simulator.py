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

from scipy.optimize import minimize,fmin
from scipy.special import gamma

from scipy.stats import truncnorm

from scipy.special import expit as logistic


class signal_simulator():
    def __init__(self,shortParsPath,longParsPath):
        self.short_dists,self.long_dists = self.make_distributions(shortParsPath,longParsPath)

    def make_s(self,x,L,k,x0): #Acts weirdly when x values are low
        out = np.array(L/(1+np.exp(-k*(x-x0))))
        return out

    def find_closest_after(self,stepsize,peak):
        mySum=0
        i=0
        while mySum<peak:
            mySum+= stepsize
            i+=1
        return i

    def find_closest_prior(self,stepsize,peak):
        mySum=0
        i=0
        while mySum<peak:
            mySum+= stepsize
            i+=1
        return i-1

    def fit_s2(self,peak_points,amplitudes,fs,length,k,x0):
        stepSize = (length/fs)/length #Is 36 correct or should it be 35?
        begin = self.find_closest_after(stepSize,peak_points[0]) #Seems to be fine
        end = self.find_closest_prior(stepSize,peak_points[1])#Seems to be fine
        alt_x = np.arange(begin,end+1,1)
        L = amplitudes[1]-amplitudes[0]
        out = self.make_s(alt_x,L,k,x0/stepSize)
        return out

    def func_4parts2(self,peak_points,amplitudes,ks,x0s,fs,length):
        maxPoint = length-1
        stepSizer = (maxPoint/120)/maxPoint
        overlap_points = [x*stepSizer for x in range(length)]
        out = np.zeros(length*6)
        startInd=0
        curSum=0
        minuser = 0
        for i in range(1,5):
            newPeaks = [curSum,curSum+peak_points[i]]
            newX0 = newPeaks[0]+((newPeaks[1]-newPeaks[0])*x0s[i-1])
            part = self.fit_s2(newPeaks,amplitudes[i-1:i+1],fs,maxPoint,ks[i-1],newX0)
            partLen = len(part)
            part_shifted = part+amplitudes[i-1]
            if newPeaks[0] in overlap_points and i>1:
                minuser = 1
            #Need some kind of overlap handling
            #out[startInd-minuser:startInd-minuser+partLen] = part_shifted #Old version with means
            if minuser>0:
                out[startInd:startInd-minuser+partLen] = part_shifted[minuser:]#out[startInd-minuser]/2
            else:
                out[startInd:startInd+partLen] = part_shifted
            startInd -= minuser
            startInd += partLen
            curSum = newPeaks[1]
            minuser=0
        return out[:length]

    #Requires a list of 5 peak points, 5 amplitudes, 4 ks, and 4 x0s
    #First peak point should be 0, next 4 should all be compared to previous
    #x0s are relative, the percentage of the difference. 
    def intermediate(self,values,fs=120,length=36):
        return self.func_4parts2(values[:5],values[5:10],values[10:14],values[14:18],fs,length)

    ######################################################################################

    ######################################################################################

    def draw_wide_gaussian(self,inp):
        #Use scipy.stats.truncnorm
        #Calculate sigma based on distance to nearest bound somehow?
        mu = inp[0]
        bounds = np.array(inp[1:])
        bounds = np.array(bounds)
        sigma = np.max(np.abs(bounds-mu))/2
        lower = bounds[0]
        upper = bounds[1]
        distr = truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        return distr

    def make_distributions(self,shortParsPath,longParsPath):
        shortParams = np.load(shortParsPath)
        longParams = np.load(longParsPath)
        short_dists = self.help_dist(shortParams)
        long_dists = self.help_dist(longParams)
        return short_dists,long_dists

    def help_dist(self,params):
        out = [self.draw_wide_gaussian(params[i,:]) for i in range(params.shape[0])]
        return out


    def drawN(self,n):
        shortPars,longPars = self.drawPars(n)
        xs = np.array([[self.intermediate(shortPars[i,:]),self.intermediate(longPars[i,:])] for i in range(n)])
        return xs

    def drawPars(self,n):
        short_pars = self.drawHalfPars(n,self.short_dists)
        long_pars = self.drawHalfPars(n,self.long_dists)
        return short_pars,long_pars

    def drawHalfPars(self,n,dists):
        all_params = np.zeros((n,18))
        all_params[:,:5] = self.drawTimes(n,dists[0:3])
        all_params[:,5:10] = self.drawAmps(n,dists[3:6])
        all_params[:,10:14] = np.ones((n,4))
        all_params[:,14:18] = self.drawX0s(n,dists[6:10])
        return all_params

    def drawTimes(self,n,dists):
        times = np.zeros((n,5))
        for i in range(3):
            times[:,i+1] = dists[i].rvs(n)
            times = self.fixTimes(n,times,dists)
        for i in range(n):
            times[i,4] = 36/120-0.00001 - np.sum(times[i,:4])
        return times
        
    def fixTimes(self,n,times,dists):
        for i in range(n):
            if np.sum(times[i,:]) >= 36/120-0.00001 or np.sum(times[i,:])<0.15:
                times[i,1:4] = self.draw1Times(dists)
        return times
                
    def draw1Times(self,dists):
        notReady = True
        newTimes = np.zeros(3)
        while notReady:
            for i in range(3):
                newTimes[i] = dists[i].rvs(1)
            if np.sum(newTimes) < 36/120-0.00001 and np.sum(newTimes)>0.15:
                notReady=False
        return newTimes

    def drawAmps(self,n,dists):
        amps = np.zeros((n,5))
        for i in range(3):
            amps[:,i+1] = dists[i].rvs(n)
        return amps

    def drawX0s(self,n,dists):
        x0s = np.zeros((n,4))
        for i in range(4):
            x0s[:,i] = dists[i].rvs(n)
        return x0s