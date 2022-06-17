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

class Ensure4d(torch.nn.Module):
    def forward(self, x):
        while(len(x.shape) < 4):
            x = x.unsqueeze(-1)
        return x

class Expression(torch.nn.Module):
    """Compute given expression on forward pass.

    Parameters
    ----------
    expression_fn : callable
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
            self.__class__.__name__ +
            "(expression=%s) " % expression_str
        )

def squeeze_final_output(x):
    """Removes empty dimension at end and potentially removes empty time
     dimension. It does  not just use squeeze as we never want to remove
     first dimension.

    Returns
    -------
    x: torch.Tensor
        squeezed tensor
    """

    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)

def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)
def _transpose_1_0(x):
    return x.permute(0, 1, 3, 2)


class EEGNet(nn.Sequential):
    def __init__(
        self,
        n_classes,
        F1=8,
        D=1,#2,
        F2=16, #Maybe this should be adapted?
        drop_prob=0.25,
        inp_length=3780,
        in_chans=1,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = inp_length
        self.final_conv_length = "auto"
        self.pool_mode = "mean"
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = 60
        self.third_kernel_size = (8, 4)
        self.drop_prob = drop_prob

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        self.add_module("ensuredims", Ensure4d())
        # b c 0 1
        # now to b 1 0 c
        self.add_module("dimshuffle", Expression(_transpose_to_b_1_c_0))

        self.add_module(
            "conv_temporal",
            nn.Conv2d(
                1,
                self.F1,
                (1, self.kernel_length),
                stride=1,
                bias=False,
                padding=(0, self.kernel_length // 2),
            ),
        )
        self.add_module(
            "bnorm_temporal",
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
        )
        self.add_module(
            "conv_spatial",
            Conv2dWithConstraint(
                self.F1,
                self.F1 * self.D,
                (self.in_chans, 1),
                max_norm=1,
                stride=1,
                bias=False,
                groups=self.F1,
                padding=(0, 0),
            ),
        )
        """
        self.add_module(
            "conv_combi",
            nn.Conv2d(
                1,
                self.F1,
                (self.in_chans, self.kernel_length),
                #max_norm=1,
                stride=(1,1),
                bias=False,
                groups=1,
                padding=(0, self.kernel_length // 2),
                dilation=(1,1),
            ),
        )
        """

        self.add_module(
            "bnorm_1",
            nn.BatchNorm2d(
                self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3
            ),
        )
        self.add_module("elu_1", Expression(elu))

        self.add_module("pool_1", pool_class(kernel_size=(1, 4), stride=(1, 4)))
        self.add_module("drop_1", nn.Dropout(p=self.drop_prob))

        # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7
        self.add_module(
            "conv_separable_depth",
            nn.Conv2d(
                self.F1 * self.D,
                self.F1 * self.D,
                (1, 16),
                stride=1,
                bias=False,
                groups=self.F1 * self.D,
                padding=(0, 16 // 2),
            ),
        )
        self.add_module(
            "conv_separable_point",
            nn.Conv2d(
                self.F1 * self.D,
                self.F2,
                (1, 1),
                stride=1,
                bias=False,
                padding=(0, 0),
            ),
        )

        self.add_module(
            "bnorm_2",
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
        )
        self.add_module("elu_2", Expression(elu))
        self.add_module("pool_2", pool_class(kernel_size=(1, 8), stride=(1, 8)))
        self.add_module("drop_2", nn.Dropout(p=self.drop_prob))

        out = self(
            torch.ones(
                (1, self.in_chans, self.input_window_samples, 1),
                dtype=torch.float32
            )
        )
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]

        if self.final_conv_length == "auto":
            n_out_time = out.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time

        self.add_module(
            "conv_classifier",
            nn.Conv2d(
                self.F2,
                self.n_classes,
                (n_out_virtual_chans, self.final_conv_length),
                bias=True,
            ),
        )
        # Transpose back to the the logic of braindecode,
        # so time in third dimension (axis=2)
        self.add_module("permute_back", Expression(_transpose_1_0))
        self.add_module("squeeze", Expression(squeeze_final_output))

        #_glorot_weight_zero_bias(self)