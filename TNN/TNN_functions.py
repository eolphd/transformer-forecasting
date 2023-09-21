#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import scipy.stats as st


# In[ ]:


def timestamp(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data


def table2lags(table, max_lag, min_lag=0, separator='_'):
    """ Given a dataframe, return a dataframe with different lags of all its columns """
    values=[]
#     for i in range(min_lag, max_lag + 1):
#         values.append(table.shift(i).copy())
#         values[-1].columns = [c + separator + str(i) for c in table.columns]
    for i in range(min_lag+1, max_lag + 1):
        values.append(table.shift(-i).copy())
        values[-1].columns = [c + separator + str(i) + 'fut' for c in table.columns]

    return pd.concat(values, axis=1)

def table2lags_GSA(table, max_lag, max_past, min_lag=0, separator='_'):
    """ Given a dataframe, return a dataframe with different lags of all its columns """
    values=[]
    for i in range(min_lag, max_past):
        values.append(table.shift(i).copy())
        values[-1].columns = [c + separator + str(i) + 'past' for c in table.columns]
    for i in range(min_lag+1, max_lag + 1):
        values.append(table.shift(-i).copy())
        values[-1].columns = [c + separator + str(i) + 'fut' for c in table.columns]

    return pd.concat(values, axis=1)

def table2lags_GSA_past(table, max_lag, max_past, min_lag=0, separator='_'):
    """ Given a dataframe, return a dataframe with different lags of all its columns """
    values=[]
    for i in range(min_lag, max_past):
        values.append(table.shift(i).copy())
        values[-1].columns = [c + separator + str(i) + 'past' for c in table.columns]
#     for i in range(min_lag+1, max_lag + 1):
#         values.append(table.shift(-i).copy())
#         values[-1].columns = [c + separator + str(i) + 'fut' for c in table.columns]

    return pd.concat(values, axis=1)

def table2lags_GSA_fut(table, max_lag, max_past, min_lag=0, separator='_'):
    """ Given a dataframe, return a dataframe with different lags of all its columns """
    values=[]
#     for i in range(min_lag, max_past):
#         values.append(table.shift(i).copy())
#         values[-1].columns = [c + separator + str(i) + 'past' for c in table.columns]
    for i in range(min_lag+1, max_lag + 1):
        values.append(table.shift(-i).copy())
        values[-1].columns = [c + separator + str(i) + 'fut' for c in table.columns]

    return pd.concat(values, axis=1)

#=========================================================================================================================#

"""truncate to predict one day n steps ahead"""
def truncate(x, feature_cols, target_cols, timestamp_cols, train_len, test_len):
    in_, out_, outn_, timestamp_, timestampn_= [], [], [], [], []
    for i in range(len(x)-train_len-test_len+1):
        in_.append(x[i:(i+train_len), feature_cols].tolist())
        out_.append(x[(i+train_len-1):(i+train_len), target_cols].tolist()) 
        outn_.append(x[(i+train_len + test_len-1):(i+train_len + test_len), target_cols].tolist())
        timestamp_.append(x[(i+train_len-1):(i+train_len), timestamp_cols].tolist())
        timestampn_.append(x[(i+train_len + test_len-1):(i+train_len+test_len), timestamp_cols].tolist())

    return np.array(in_), np.array(out_), np.array(outn_), np.array(timestamp_), np.array(timestampn_)

# """truncate to predict n steps ahead"""
def truncate_multistep(x, feature_cols, target_cols, timestamp_cols, train_len, test_len):
    in_, out_, timestamp_= [], [], []
    for i in range(len(x)-train_len-test_len+1):
        in_.append(x[i:(i+train_len), feature_cols].tolist())
        out_.append(x[(i+train_len-1):(i+train_len+test_len), target_cols].tolist()) 
        timestamp_.append(x[(i+train_len-1):(i+train_len+test_len), timestamp_cols].tolist())
    return np.array(in_), np.array(out_), np.array(timestamp_)


# """truncate to predict n steps ahead"""
def truncate_informer(x, feature_cols, target_cols, timestamp_cols, train_len, test_len):
    in_, out_, timestamp_= [], [], []
    for i in range(len(x)-train_len-test_len+1):
        in_.append(x[i:(i+train_len), feature_cols].tolist())
#         out_.append(x[(i+train_len-test_len):(i+train_len+test_len), target_cols].tolist()) 
#         timestamp_.append(x[(i+train_len-test_len):(i+train_len+test_len), timestamp_cols].tolist())
        out_.append(x[i:(i+train_len+test_len), target_cols].tolist()) 
        timestamp_.append(x[i:(i+train_len+test_len), timestamp_cols].tolist())
    return np.array(in_), np.array(out_), np.array(timestamp_)

def truncate_informer3(x, feature_cols, target_cols, timestamp_cols, tback1, tback2, steps):
    in_, out_, timestamp_= [], [], []
    if tback1 >= tback2:
        for i in range(len(x)-tback1-steps+1):
            in_.append(x[i:(i+tback1), feature_cols].tolist())
            out_.append(x[(i+tback1-tback2):(i+tback1+steps), target_cols].tolist()) 
            timestamp_.append(x[(i+tback1-tback2):(i+tback1+steps), timestamp_cols].tolist())
    else:
        for i in range(len(x)-tback2-steps+1):
            in_.append(x[i+tback2-tback1:(i+tback2), feature_cols].tolist())
            out_.append(x[i:(i+tback2+steps), target_cols].tolist()) 
            timestamp_.append(x[i:(i+tback2+steps), timestamp_cols].tolist())
    return np.array(in_), np.array(out_), np.array(timestamp_)

#===========================================================================================================================#

def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=42).permutation(len(a))
    return a[p], b[p]

def stats(pred, confidence):
    a = 1.0 * np.array(pred)
    m,n = (a.shape)
    m, se, median = np.mean(a, axis = 1), st.stats.tstd(a, axis = 1), np.median(a, axis = 1)
    h = 1.96*se
    return m, m-h, m+h, median, se

