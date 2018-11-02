#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 17:19:02 2018

@author: neelkanth
"""
"""Importing Libraries"""
# Importing standard libraries
import sys
import time
import re
import json
import os
from pathlib import PurePath, Path
from collections import OrderedDict as od

# import python scientific libraries
import numpy as np
import pandas as pd
import pandas_datareader as web
pd.set_option('display.max_rows', 100)
import scipy.stats as stats
import statsmodels.api as sm
import math
import ffn
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from multiprocessing import cpu_count
from numba import jit

# import visual tools
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

plt.style.use('seaborn-talk')
plt.style.use('bmh')
plt.rcParams['font.size'] = 9.5
plt.rcParams['font.weight'] = 'medium'
plt.rcParams['figure.figsize'] = 10, 7
blue, green, red, purple, gold, teal = sns.color_palette('colorblind', 6)

# Import util libs
import warnings
warnings.filterwarnings("ignore")
import missingno as msno
import src.features.bars as brs
import src.features.snippets as snp
from src.utils.utils import *
from tqdm import tqdm, tqdm_notebook

RANDOM_STATE = 777

"""loading dataframe"""
dataset = pd.read_csv('./data/dataset.csv', index_col=0)

"""Symmetric CUMSUM Filter"""
# Symmetric CUMSUM Filter
def getTEvents(gRaw, h):
    tEvents, sPos, sNeg = [], 0, 0
    diff = np.log(gRaw).diff().dropna()
    for i in tqdm(diff.index[1:]):
        try:
            pos, neg = float(sPos+diff.loc[i]), float(sNeg+diff.loc[i])
        except Exception as e:
            print(e)
            print(sPos+diff.loc[i], type(sPos+diff.loc[i]))
            print(sNeg+diff.loc[i], type(sNeg+diff.loc[i]))
            break
        sPos, sNeg = max(0., pos), min(0., neg)
        if sNeg < -h:
            sNeg=0; tEvents.append(i)
        elif sPos>h:
            sPos=0; tEvents.append(i)
    return pd.DatetimeIndex(tEvents)

# daily Volume Estimator
def getDailyVol(close, span0=100):
    # daily vol reindexed to close
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0>0]
    df0 = (pd.Series(close.index[df0 - 1], index=close.index[close.shape[0]-df0.shape[0]:]))
    try:
        df0 = close.loc[df0.index]/ close.loc[df0.values].values - 1 # daily rets
    except Exception as e:
        print(f"error: {e}\nplease confirm no duplicate indices")
    df0=df0.ewm(span=span0).std().rename('dailyVol')
    return df0

# Triple-Barrier Labelling Method
def applyPtSlOnT1(close, events, ptSl, molecule):
    # apply stop/ loss profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    if ptSl[0]>0:
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index) #NaNs
    if ptSl[1]>0:
        sl = -ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index) #NaNs
    for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1] #path prices
        df0 = (df0/ close[loc] -1)* events_.at[loc, 'side'] # path returns
        out.loc[loc, 'sl'] = df0[df0<sl[loc]].index.min() # earliest stop loss
        out.loc[loc, 'pt'] = df0[df0>pt[loc]].index.min() # earliest profit taking
    return out

        