#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 19:49:12 2018

@author: neelkanth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tick_ = pd.read_csv('saved_data/tick_bars.csv').set_index('date')
vol_  = pd.read_csv('saved_data/volume_bars.csv').set_index('date')
time_ = pd.read_csv('saved_data/time_bars.csv').set_index('date')
dollar= pd.read_csv('saved_data/dollar_bars.csv').set_index('date')

tick_.index  = pd.to_datetime(tick_.index)
vol_.index   = pd.to_datetime(vol_.index)
time_.index  = pd.to_datetime(time_.index)
dollar.index = pd.to_datetime(dollar.index)

"""What type produces the most stable weekly count"""
tick_weekly = tick_['close'].resample('W')
vol__weekly =  vol_['close'].resample('W')
time_weekly = time_['close'].resample('W')
dollaweekly =dollar['close'].resample('W')

count_df = pd.concat((tick_weekly.count(), vol__weekly.count(), time_weekly.count(), dollaweekly.count()), axis=1)
count_df.columns = ['tick', 'vol', 'time','dollar']

# Number of bars over time
count_df.plot(kind='bar', figsize=[10, 5], color=('darkred', 'darkblue', 'green', 'darkcyan'))
plt.title('Numbers of bars (obs) over time', loc='center', fontsize='bold', fontname='Times New Roman')
plt.show()

"""Checking Serial Correlation among the data"""
from statsmodels.graphics.tsaplots import plot_acf

time_returns = np.log(time_['close']).diff().dropna()
tick_returns = np.log(tick_['close']).diff().dropna()
vol__returns = np.log(vol_['close']).diff().dropna()
dollarweekly = np.log(dollar['close']).diff().dropna()

# Autocorrelation plot
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)

plot_acf(time_returns, lags=10)  # Testing Autocorrelation in time bars
ax.set_title('Time Bars AutoCorrelation')
#plt.show()

plot_acf(tick_returns, lags=10)  # Testing Autocorrelation in tick bars
ax.set_title('Tick Bars Autocorrelation')
#plt.show()

plot_acf(vol__returns, lags=10)  # Testing Autocorrelation in volume bars
ax.set_title('Volume Bars Autocorrelation')
#plt.show()

plot_acf(dollarweekly, lags=10)  # Testing Autocorrelation in dollar bars
ax.set_title('Dollar Bars Autocorrelation')
plt.show()

# Standardized Time-Series plot
plt.plot(time_['close']/ time_['close'][0])
plt.plot(tick_['close']/ tick_['close'][0])
plt.plot(vol_['close']/ vol_['close'][0])
plt.plot(dollar['close']/ dollar['close'][0])
plt.xticks(rotation=90)
plt.show()

pd.plotting.autocorrelation_plot(time_returns)
pd.plotting.autocorrelation_plot(tick_returns)
pd.plotting.autocorrelation_plot(vol__returns)
pd.plotting.autocorrelation_plot(dollarweekly)