import numpy as np

from scipy.signal import periodogram
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from scipy.stats import truncnorm


def get_freq_dom(result, fs=120):
    if len(np.shape(result))<2:
        result = np.reshape(result, (1,-1))
    freq, spectral = periodogram(result,fs,axis=1)
    return freq,spectral

def plot_accuracy(scores_df, height=5, aspect=1, **facet_kwargs):    
    if isinstance(scores_df, list):
        scores_df = pd.concat(scores_df)
    data = scores_df[scores_df['metric']=='accuracy']
    g = sns.FacetGrid(data=data, height=height, aspect=aspect, **facet_kwargs)
    g.map_dataframe(sns.lineplot, x='epoch', y='value', style='phase', color='b')
    def aux(data, **kwargs):
        valid_acc = data[data.phase=='valid'].set_index('epoch', inplace=False).value
        ax = plt.gca()
        mx = valid_acc.max()
        ax.axvline(valid_acc.argmax(), color='red')
        ax.axhline(mx, color='red')
        ax.text(0, mx, "valid: {:.2f}".format(mx), va='bottom', color='red')
        test_acc = data['test_accuracy'].unique()
        assert len(test_acc)==1
        test_acc = test_acc[0]
        ax.axhline(test_acc, color='green')
        ax.text(0, test_acc, "test: {:.2f}".format(test_acc), va='bottom', color='green')     
    g.map_dataframe(aux)
    g.set_xlabels('number of epochs')
    g.set_ylabels('accuracy [a.u.]', color='b')
    g.add_legend()

def plot_loss(scores_df, height=5, aspect=1, **facet_kwargs):
    if isinstance(scores_df, list):
        scores_df = pd.concat(scores_df)
    data = scores_df[scores_df['metric']=='loss']
    g = sns.FacetGrid(data=data, height=height, aspect=aspect, **facet_kwargs)
    g.map_dataframe(sns.lineplot, x='epoch', y='value', style='phase', color='orange')
    g.set(yscale='log')
    g.set_ylabels('loss [a.u.]', color='orange')
    g.set_xlabels('number of epochs')
    g.add_legend()

def splice_data(X_emp,y_emp,splices):
    assert X_emp.shape[2]%splices == 0
    assert 15%splices == 0    #Due to 15 being code repetitions in this dataset
    new_X_emp = np.zeros((X_emp.shape[0]*splices,X_emp.shape[1],int(X_emp.shape[2]/splices)))
    new_y_emp = np.zeros(y_emp.shape[0]*splices)
    for i,val in enumerate(y_emp):
        X_split = X_emp[i]
        for i2 in range(splices):
            i3 = i2+1
            spliceLength = X_split.shape[1]/splices
            new_X_emp[i*splices+i2] = X_split[:,int(i2*spliceLength):int(i3*spliceLength)]
            new_y_emp[i*splices+i2] = val
    return new_X_emp,new_y_emp
        
    
def split_one(X_split,splices):
    new_X = np.zeros((splices,X_split.shape[0],X_split.shape[1]/5))
    for i in range(splices):
        spliceLength = X_split.shape[1]/splices
        new_X[i] = X_split[:,i*spliceLength:(i+1)*spliceLength]
    return new_X

def gaussian_drawer(inp):
        mu = inp[0]
        bounds = np.array(inp[1:])
        bounds = np.array(bounds)
        sigma = np.max(np.abs(bounds-mu))/2
        lower = bounds[0]
        upper = bounds[1]
        distr = truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        return distr
