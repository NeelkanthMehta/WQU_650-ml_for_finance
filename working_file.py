#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 21:14:16 2018

@author: neelkanth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ta

# Loading dataset
'''
the loaded dataset is processed and cleaned dollar bar dataset for an illustrative index
'''
dataset = pd.read_csv('./data/dataset.csv', index_col=0)
dataset['dv'] = dataset['dv'].astype(np.int64)
dataset.index = pd.to_datetime(dataset.index)

# Creating ohlc data
price = dataset['price'].resample('10Min').ohlc().fillna(method='ffill').dropna(how='all')

df = price.merge(dataset[['v','dv']], how='outer', left_index=True, right_index=True).fillna(method='ffill').dropna(how='all')
df['v'][0], df['dv'][0] = 0, 0

df.info()

"""Generating Features"""
'''For details, ref: https://bit.ly/2CZp7OB'''
# Awsome Oscilator
ao = ta.momentum.ao(high=df['high'], low=df['low'], fillna=True) # Awsome Oscillator
rsi = ta.momentum.rsi(close=df['close'], fillna=True) # Relative Strength Index
tsi = ta.momentum.tsi(close=df['close'], fillna=True) # True Strength Index
macd = ta.trend.macd(close=df['close'], fillna=True) # MACD
r_pct = ta.momentum.wr(high=df['high'], low=df['low'], close=df['close'], fillna=True) # William's %R
EoM = ta.volume.ease_of_movement(high=df['high'], low=df['low'], close=df['close'], volume=df['v'],fillna=True) # East of Movement
cmf = ta.volume.chaikin_money_flow(high=df['high'], low=df['low'], close=df['close'], volume=df['v'], fillna=True) # Chaikin Money Flow
force_index = ta.volume.force_index(close=df['close'], volume=df['v'], fillna=True) # Force Index

# eight features in all
features = pd.concat((ao, rsi, tsi, macd, r_pct, EoM, cmf, force_index), axis=1)
cols = ['ao', 'rsi', 'tsi', 'macd', 'r_pct', 'EoM', 'cmf', 'force_index']

"""Conducting feature selection"""
'''We'll first require to scale all the features'''
from sklearn.preprocessing import StandardScaler

# Instantiating StandardScaler and fitting it to features
sc = StandardScaler()
features = sc.fit_transform(features)

# Principal COmponent Analysis
from sklearn.decomposition import PCA

# Instantiating and fitting PCA
pca = PCA(n_components=None, svd_solver='full')
processed = pca.fit_transform(features)

pca.explained_variance_ratio_
pca.explained_variance_
principal_components = pd.DataFrame(pca.components_)
principal_components.iloc[0,:].sum()
cov = pd.DataFrame(pca.get_covariance())

#principal_components[0].abs().sum()/ principal_components.abs().sum()
#[principal_components.iloc[i,:].abs().sum() for i in range(0,8)]