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


class preprocessor():
    def __init__(self,fs=120,band_pass=[2.0,30.0],tmin=0.0,tmax=31.5):
        # Configuration for the preprocessing
        self.fs = 120  # target sampling frequency in Hz for resampling
        self.band_pass = [2.0, 30.0]  # bandpass cutoffs in Hz for spectral filtering
        self.tmin = 0.0  # trial onset in seconds for slicing
        self.tmax = 31.5  # trial end in seconds for slicing
    
        self.FR = 60
           
    def process(self,inp):  #Inp has shape channels * measurements * trials  = (1,3780,N)
        
        """
        I should instead be doing only the spectral band-pass and downsampling pbb
        Pbb loop over the first and last dimension of the simulated data? 
        Then band-pass and downsample
        """
        mne.set_log_level(verbose=False)
        outp = np.zeros(inp.shape)
        
        for i1 in range(inp.shape[0]):
            for i2 in range(inp.shape[2]):
                current = inp[i1,:,i2]
                current = mne.filter.filter_data(current,sfreq=self.fs,l_freq=self.band_pass[0],h_freq=self.band_pass[1])
                #current = mne.filter.resample(current,)   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
                outp[i1,:,i2] = current
        return outp