#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import TNN_functions
import tensorflow as tf


# In[ ]:


def preprocess(df, df_rain, df_targets, steps, col_pred, col_pred_targets, t_lag,t_lag2, n_test, batch_size, dev_ratio, itera):
    """Truncate time series into (samples, time steps, features)"""
    X = df.copy().iloc[:, :]
#     print('df:', df.shape)
    print('X:',X.columns)

#     rain = df_rain.copy().iloc[:,:]
#     print('rain:',df_rain.columns)

    rain = df_rain.copy().iloc[:,:-8]
    T2V = df_rain.copy().iloc[:,-8:]

    rain_lag = TNN_functions.table2lags(rain, steps, min_lag =0)#-1) # 0 to use only forecasts, -1 to include past values
    print('rain_for:',rain_lag.columns)

    X = pd.concat([X, rain_lag, T2V], axis=1)
    X = X.dropna(axis=0)
#     print('X_dropna:',X.shape)
    Xnp = X.to_numpy()
    
    X_targets = df_targets.copy().iloc[:, :]
#     print('df_targets:', df_targets.shape)

    X_targets = X_targets.iloc[:-steps,:]
    X_targets = pd.concat([X_targets, df_rain.iloc[:-steps, -8:]], axis=1)
#     print('X_targets:', X_targets.shape)
    Xnp_targets = X_targets.to_numpy()


    #========================================================================================================================#
#     """truncate to predict one day n steps ahead"""
#     X_in, X_out, X_outn, timestamp_, timestampn_ = TNN_functions.truncate(Xnp, 
#                           feature_cols=range(Xnp.shape[-1]), 
#                           target_cols=range(col_pred-1, col_pred), # set the target_cols as a range in order to obtain a 3D output array
#                         timestamp_cols=range(Xnp.shape[-1]-8, Xnp.shape[-1]),
#                           train_len=t_lag, 
#                           test_len=steps) 
#     X_out =  np.concatenate((X_out, X_outn), axis=1)
#     timestamp = np.concatenate((timestamp_, timestampn_), axis=1)
# #     print(X_out.shape)
# #     print(timestamp.shape)
#     X_out = np.concatenate((X_out, timestamp), axis=-1)
#     print('X_out:',X_out.shape)


#     """truncate TARGETS to predict one day n steps ahead"""
#     X_in_targets, X_out_targets, X_outn_targets, timestamp_, timestampn_ = TNN_functions.truncate(Xnp_targets, 
#                           feature_cols=range(Xnp_targets.shape[-1]), 
#                           target_cols=range(col_pred_targets-1, col_pred_targets), # set the target_cols as a range in order to obtain a 3D output array
#                           timestamp_cols=range(Xnp_targets.shape[-1]-8, Xnp_targets.shape[-1]),
#                           train_len=t_lag, 
#                           test_len=steps) 
#     X_out_targets =  np.concatenate((X_out_targets, X_outn_targets), axis=1)
#     timestamp = np.concatenate((timestamp_, timestampn_), axis=1)
#     X_out_targets = np.concatenate((X_out_targets, timestamp), axis=-1)
# #     print('X_out_targets:', X_out_targets.shape)

#     """truncate to predict n steps ahead"""
    X_in, X_out, timestamp_out= TNN_functions.truncate_informer3(Xnp, 
                          feature_cols=range(Xnp.shape[-1]), 
                          target_cols=range(col_pred-1, col_pred), # set the target_cols as a range in order to obtain a 3D output array
                          timestamp_cols=range(Xnp.shape[-1]-8, Xnp.shape[-1]),
                          tback1=t_lag,
                          tback2=t_lag2,                                       
                          steps=steps) 
    print('X_out', X_out.shape)
    print(timestamp_out.shape)
    print('X_in', X_in.shape)
    X_out =  np.concatenate((X_out, timestamp_out), axis=-1)
#     X_out[:,-steps:,:1]=0

        
#             """truncate TARGETS to predict one day n steps ahead"""
    X_in_targets, X_out_targets, timestamp = TNN_functions.truncate_informer3(Xnp_targets, 
                          feature_cols=range(Xnp_targets.shape[-1]), 
                          target_cols=range(col_pred_targets-1, col_pred_targets), # set the target_cols as a range in order to obtain a 3D output array
                          timestamp_cols=range(Xnp_targets.shape[-1]-8, Xnp_targets.shape[-1]),
                          tback1=t_lag,
                          tback2=t_lag2,                                       
                          steps=steps) 
    X_out_targets = np.concatenate((X_out_targets, timestamp), axis=-1)
#     X_out_targets[:,-steps:,:1]=0

    print('X_out_targets:', X_out_targets.shape)
        #========================================================================================================================#

    # Split into training and testing sets

    """first split the test subset to reserve the last n_test time steps of the time series"""
    test_X = X_in[-n_test:, :, :]
    test_y = X_out[-n_test:, :, :]
    test_y_targets = X_out_targets[-n_test:, :, :]
#     print('test_y_targets:', test_y_targets.shape)
    X_in = X_in[:-n_test, :, :]
    X_out = X_out[:-n_test, :, :]

    """next, shuffle the remaining time series"""
    X_in_shuffled, X_out_shuffled = TNN_functions.shuffle(X_in, X_out)

    """finally, split the training and validation subsets"""
    """1) mask for inverse indexing the dev subset that will be used to perform K-fold cross-validation"""

    n_dev = int(dev_ratio* len(X_in_shuffled))
    n_train = len(X_in_shuffled) - n_dev

    mask = np.ones(len(X_in_shuffled), bool)
    mask[itera * n_dev : (itera+1) * n_dev] = 0
    train_X = X_in_shuffled[mask]
    train_y = X_out_shuffled[mask]

    """2) standardize training, development, and testing subsets using mean and std from training subset"""
    train_X_mean = (train_X.mean(axis=(0,1)))
    train_X_std = (train_X.std(axis=(0,1)))
    train_y_mean = (train_y.mean(axis=(0,1)))
    train_y_std = (train_y.std(axis=(0,1)))

    train_X = (train_X - train_X_mean)/train_X_std
    train_y = (train_y - train_y_mean)/train_y_std


    dev_X = (X_in_shuffled[itera * n_dev : (itera+1) * n_dev, :, :] - train_X_mean)/train_X_std
    dev_y = (X_out_shuffled[itera * n_dev : (itera+1) * n_dev, :, :] - train_y_mean)/train_y_std

    test_X = (test_X - train_X_mean)/train_X_std
    test_y = (test_y - train_y_mean)/train_y_std
    test_y_targets = (test_y_targets - train_y_mean)/train_y_std

    # Prepare the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    train_dataset = train_dataset.batch(batch_size)

    # prepare the development dataset
    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_X, dev_y))
    dev_dataset = dev_dataset.batch(batch_size)

    # prepare the test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    test_dataset = test_dataset.batch(batch_size)
    
    return train_dataset, dev_dataset, train_X, train_y, test_X, test_y_targets, train_y_mean, train_y_std, dev_X, dev_y, X, train_X_mean, train_X_std

