#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd # pandas for data wrangling
import numpy as np # numpy for standard deviation
import matplotlib.pyplot as plt # plt from matplotlib
plt.style.use('ggplot') # define ggplot as plotstyle 

# load data
train = pd.read_csv("./data/train.gz",usecols=['click','hour']) # due to memory reasons load only necessary columns
train.head(20)

# Wrangle train data and create datetime column
dateHour = pd.to_datetime(train['hour'].astype(str), format='%y%m%d%H') # convert column 'hour' to date time object dateHour
train.insert(loc=0, column='dateHour', value=dateHour) # insert dateHour as first column

# CTR over time
# .... CTR = clicks / impressions
# .... clicks = sum of 'click' after 1-hour grouping
# .... impressions = count of entries in 'click' after 1-hour grouping

# train_grouped_hourly_old = train.groupby(pd.Grouper(key='dateHour',freq='H')).agg({"click": np.sum, "id": pd.Series.nunique}) # data grouped by hour and clicks per ad
train_grouped_hourly = train.groupby(pd.Grouper(key='dateHour',freq='H') # groupby hour
                                     )['click'].agg(['sum','count'] # sum and count 'click'
                                         ).rename(columns={'sum':'clicks','count':'impressions'}) # rename sum and count to clicks & impressions

train_grouped_hourly['ctr'] = train_grouped_hourly['clicks']/train_grouped_hourly['impressions'] # calculate CTR in new column 'ctr'
del train # delete original data train

# PLOT
fig, ax = plt.subplots(figsize=(20,10)) # create figure & axes

ax.plot(train_grouped_hourly.index,train_grouped_hourly['ctr'], label = "CTR", color="black", linewidth=2)  # plot 'ctr'
ax.legend(loc='best')

ax.set(xlabel='Date', title='CTR - Time Series') # set labels, etc

fig.savefig("ctr_ts.png")
# plt.show()

## hourly ticks ##
# https://stackoverflow.com/questions/48790378/how-to-get-ticks-every-hour



# Outlier Detection
# .... Build a simple outlier detection algorithm based on a “moving average”. A data point is identified as an outlier, if it is more than 1.5 standard
# .... deviations apart from its calculated moving average (for simplicity’s sake, we will assume a Gaussian distribution here). 
# .... The outcome of this task is a plot, that highlights all found outliers. 

# 12 hour moving average & according standard deviation #
train_grouped_hourly['MA_12h'] = train_grouped_hourly['ctr'].rolling(12).mean() # 12h moving average
train_grouped_hourly['std_12h'] = train_grouped_hourly['ctr'].rolling(12).std() # according std to 12h moving average
# 24 hour moving average #
train_grouped_hourly['MA_24h'] = train_grouped_hourly['ctr'].rolling(24).mean() # 24h moving average
train_grouped_hourly['std_24h'] = train_grouped_hourly['ctr'].rolling(24).std() # according std to 24h moving average
# 24 hour moving average #
train_grouped_hourly['MA_48h'] = train_grouped_hourly['ctr'].rolling(48).mean() # 48h moving average
train_grouped_hourly['std_48h'] = train_grouped_hourly['ctr'].rolling(48).std() # according std to 48h moving average

## calculate standard deviation for whole series
stdev = np.std(train_grouped_hourly['ctr'])

# calculate ranges according to definition of outlier = moving average +- 1.5 std
train_grouped_hourly['upper_12'] = train_grouped_hourly['MA_12h'] + 1.5 * train_grouped_hourly['std_12h'] # upper range for MA_12h
train_grouped_hourly['upper_12_full'] = train_grouped_hourly['MA_12h'] + 1.5 * stdev # upper range for MA_12h with full stdev
train_grouped_hourly['lower_12'] = train_grouped_hourly['MA_12h'] - 1.5 * train_grouped_hourly['std_12h'] # lower range for MA_12h
train_grouped_hourly['lower_12_full'] = train_grouped_hourly['MA_12h'] - 1.5 * stdev # lower range for MA_12h with full stdev
train_grouped_hourly['upper_24'] = train_grouped_hourly['MA_24h'] + 1.5 * train_grouped_hourly['std_24h'] # upper range for MA_24h
train_grouped_hourly['upper_24_full'] = train_grouped_hourly['MA_24h'] + 1.5 * stdev # upper range for MA_24h with full stdev
train_grouped_hourly['lower_24'] = train_grouped_hourly['MA_24h'] - 1.5 * train_grouped_hourly['std_24h'] # lower range for MA_24h
train_grouped_hourly['lower_24_full'] = train_grouped_hourly['MA_24h'] - 1.5 * stdev # lower range for MA_24h with full stdev
train_grouped_hourly['upper_48'] = train_grouped_hourly['MA_48h'] + 1.5 * train_grouped_hourly['std_48h'] # upper range for MA_48h
train_grouped_hourly['upper_48_full'] = train_grouped_hourly['MA_48h'] + 1.5 * stdev # upper range for MA_48h
train_grouped_hourly['lower_48'] = train_grouped_hourly['MA_48h'] - 1.5 * train_grouped_hourly['std_48h'] # lower range for MA_48h
train_grouped_hourly['lower_48_full'] = train_grouped_hourly['MA_48h'] - 1.5 * stdev # lower range for MA_48h
# for plotting: assign ctr value to outliers according to rule: bigger than upper range OR smaller than lower range #
train_grouped_hourly.loc[(train_grouped_hourly['ctr'] > train_grouped_hourly['upper_12']) | (train_grouped_hourly['ctr'] < train_grouped_hourly['lower_12']), 'outlier_12h'] = train_grouped_hourly['ctr']
train_grouped_hourly.loc[(train_grouped_hourly['ctr'] > train_grouped_hourly['upper_12_full']) | (train_grouped_hourly['ctr'] < train_grouped_hourly['lower_12_full']), 'outlier_12h_full'] = train_grouped_hourly['ctr']


train_grouped_hourly.loc[(train_grouped_hourly['ctr'] > train_grouped_hourly['upper_24']) | (train_grouped_hourly['ctr'] < train_grouped_hourly['lower_24']), 'outlier_24h'] = train_grouped_hourly['ctr']
train_grouped_hourly.loc[(train_grouped_hourly['ctr'] > train_grouped_hourly['upper_24_full']) | (train_grouped_hourly['ctr'] < train_grouped_hourly['lower_24_full']), 'outlier_24h_full'] = train_grouped_hourly['ctr']


train_grouped_hourly.loc[(train_grouped_hourly['ctr'] > train_grouped_hourly['upper_48']) | (train_grouped_hourly['ctr'] < train_grouped_hourly['lower_48']), 'outlier_48h'] = train_grouped_hourly['ctr']
train_grouped_hourly.loc[(train_grouped_hourly['ctr'] > train_grouped_hourly['upper_48_full']) | (train_grouped_hourly['ctr'] < train_grouped_hourly['lower_48_full']), 'outlier_48h_full'] = train_grouped_hourly['ctr']



# PLOT
fig1, ax = plt.subplots(figsize=(20,10))

# plot data
ax.plot(train_grouped_hourly.index,train_grouped_hourly['ctr'], label = "CTR", color="black", linewidth=2)
ax.plot(train_grouped_hourly.index,train_grouped_hourly['MA_12h'], label = "MA_12h", color="green")
ax.plot(train_grouped_hourly.index,train_grouped_hourly['MA_24h'], label = "MA_24h", color="darkblue")
# ax.plot(train_grouped_hourly.index,train_grouped_hourly['outlier_48h'], label = "outlier_12h", color="purple", marker='o',markersize=16)
ax.plot(train_grouped_hourly.index,train_grouped_hourly['outlier_24h'], label = "outlier_24h", color="darkred", marker='o', linestyle = 'None', markersize=12)
ax.plot(train_grouped_hourly.index,train_grouped_hourly['outlier_24h_full'], label = "outlier_24h_full", color="black", marker='x', linestyle = 'None', markersize=12)

ax.plot(train_grouped_hourly.index,train_grouped_hourly['outlier_12h'], label = "outlier_12h", color="red", marker='o', linestyle = 'None')
ax.plot(train_grouped_hourly.index,train_grouped_hourly['outlier_12h_full'], label = "outlier_12h_full", color="black", marker='+', linestyle = 'None', markersize=12)


ax.set(xlabel='Date',
       title='Time Series of CTR with MA_12h, MA_24H, MA_48h')

ax.legend(loc='best')


# ax.grid()

fig1.savefig("ctr_ts_ma_outlier.png")
#plt.show() # show plot


