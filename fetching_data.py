#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:57:13 2018

@author: neelkanth
"""

"""Importing all the libraries"""

# Importing standard libraries
import sys
import time
import re
import os
import json
from pathlib import PurePath, Path

os.environ['THEANO_FLAGS'] = 'device=cpu, floatX=float32'

# Importing python scientific libraries
import numpy as np
import pandas as pd
import pandas_datareader as web
pd.set_option('display.max_rows', 100)
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
import scipy.stats as stats
import statsmodels.api as sm
from numba import jit
import math
import pymc3 as pm
from theano import shared, theano as tt

# importing visual libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

plt.style.use('seaborn-talk')
plt.style.use('bmh')
plt.rcParams['font.weight'] = 'medium'

blue, grean, red, purple, gold, teal = sns.color_palette('colorblind', 6)

# Import util libs
import pyarrow as palind
import pyarrow.parquet as pq
from tqdm import tqdm, tqdm_notebook
import warnings
warnings.filterwarnings("ignore")
import missingno as msno

from src.utils.utils import *
from src.features.bars import get_imbalance
import src.features.bars as brs
import src.features.snippets as snp

RANDOM_STATE = 777


"""Reading and Cleaning data"""

def read_kibot_ticks(fp):
    cols = list(map(str.lower, ['Date', 'Time', 'Price', 'Bid', 'Ask', 'Size']))
    df = (pd.read_csv(fp, header=None)
    .rename(columns=dict(zip(range(len(cols)), cols)))
    .assign(dates=lambda df: (pd.to_datetime(df['date']+df['time'], format='%m/%d/%Y%H:%M:%S')))
    .assign(v = lambda df: df['size']) # volume
    .assign(dv= lambda df: df['price']*df['size']) # dollar volume
    .drop(['date', 'time'], axis=1)
    .set_index('dates')
    .drop_duplicates())
    return df

infp = PurePath('./data/IVE_tickbidask.txt')

# Creating a tick dataframe
df = read_kibot_ticks(infp)

# Printing the output fataframe
outfp = PurePath('./data/IVE_tickbidask.parq')
df.to_parquet(outfp)

# missing numbers
msno.matrix(df)


"""Eliminating outliers from the dataset"""
def mad_outlier(y, thresh=3.):
    '''
    compute outliers based on mad
    # args
        y: assumed to be array with shape (N,1)
        thresh: float()
    # returns
        array index of outliers
    '''
    median = np.median(y)
    diff = np.sum((y-median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    
    modified_z_score = 0.6745 * diff/ med_abs_deviation
    return modified_z_score > thresh

# defining variable mad
mad = mad_outlier(df.price.values.reshape(-1,1))

# removing outliers
df = df.loc[~mad]

"""Exporting the processed data to file"""
outfp = PurePath('./data/clean_IVE_fut_prices.parq')
df.to_parquet(outfp)


"""Creating Dollar Bars"""
def dollar_bars(df, dv_column, m):
    '''
    compute dollar bars
    
    # args
        df: pd.DataFrame()
        dv_column: name of dollar volume bar
        m: int(), threshold value for dollars
    # returns
        idx: list of indices
    '''
    t = df[dv_column]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += x
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
    return idx

def dollar_bar_df(df, dv_column, m):
    idx = dollar_bars(df, dv_column, m)
    return df.iloc[idx].drop_duplicates()


# loading data - if not already loaded
df = pd.read_parquet('./data/clean_IVE_fut_prices.parq')
cprint(df)

# executing the function:
dollar_M = 1_000_000  # arbitrary
print(f"dollar threshold: {dollar_M:,}")
dv_bar_df = dollar_bar_df(df,'dv', dollar_M)
cprint(dv_bar_df)

# Resampling
def select_sample_data(ref, sub, price_col, date):
    '''
    select a sample of data based on date, assumes datetimeindex
    
    # args
        ref: pd.DataFrame containing all ticks
        sub: subordinated pd.DataFrame of prices
        price_col: str(), price column
        date: str(), date to select
    # returns
        xdf: ref pd.Series
        xtdf: subordinated pd.Series
    '''
    xdf = ref[price_col].loc[date]
    xtdf= sub[price_col].loc[date]
    return xdf, xtdf

xDate = '2009-10-01' #'2017-10-4'
xdf, xtdf = select_sample_data(df, dv_bar_df, 'price', xDate)

"""Exporting the Dollar Bar"""
dv_bar_df.to_csv('./data/dataset.csv', sep=',', header=True)
