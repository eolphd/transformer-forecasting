#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from pandas import read_csv

import TNN_functions
# import Transformer_addwgt


# In[ ]:


def test_MC(transformer, MC_iterations, test_y_targets, n_test, test_X, train_y_mean, train_y_std):
    dist = np.ones(shape=(MC_iterations, (test_y_targets.shape[1]-1)*3))
    dist_ts = pd.DataFrame(np.ones(shape=(n_test, MC_iterations)))

    for MC in range(MC_iterations):
        ta=tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True)
        tu = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        ta=ta.write(0, (test_y_targets[:,:1,:]) )
        ta=tf.transpose(ta.stack(), perm=[1,0,2,3])
        ta=ta.reshape(ta.shape[0], ta.shape[1], ta.shape[-1])
        for i in range(0, test_y_targets.shape[1]-1):
            ta, attention_weights = transformer(((test_X), ta), True, False, True, False)
            tu = tu.write(i, ta)
        test_output = tf.transpose(tu.stack(), perm=[1,0,2,3])
        # Inverse normalize
        df_train_mean_y=((train_y_mean))
        df_train_std_y=((train_y_std))
        y_hat = K.eval(test_output) # to transform from tensor to numpy array
        print('y_hat:', y_hat.shape)
        y_hat = (y_hat.reshape(y_hat.shape[0], y_hat.shape[1]))
        y_test = K.eval(tf.cast(test_y_targets[:,1:,:1], dtype=tf.float32))
        print('y_test:', y_test.shape)
        y_test = (y_test.reshape(y_test.shape[0], y_test.shape[1]))
        y_hat_inv_scaled = pd.DataFrame(y_hat * df_train_std_y + df_train_mean_y)
        y_test_inv_scaled = pd.DataFrame(y_test * df_train_std_y + df_train_mean_y)

        for i in range(test_y_targets.shape[1]-1):
            dist_ts.iloc[:,MC] = y_hat_inv_scaled.iloc[:,i]
        print('iteration:', MC)

#     dist_ts.to_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/95PPU_test.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads))


#     for i in range(test_y_targets.shape[1]-1):
#         y_test_inv_scaled.iloc[:,i].to_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/y_test_%i.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads, i))

#     for i in range((test_y_targets.shape[1]-1)):
#         pred = pd.read_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/95PPU_test.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads)).iloc[:,1:]#*0.028316847 #(cfs to cms)
#         data = pd.read_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/y_test_%i.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads, i)).iloc[:,1:]#*0.028316847 #(cfs to cms)


#         mean, lb, ub, median, se = TNN_functions.stats(pred, 0.95)
#     mean = pd.DataFrame(mean)
#     #     se = pd.DataFrame(se)

#     #         mean = np.reshape(mean, (-1, 1))
#     #         se = np.reshape(se, (-1, 1))
#     mean.to_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/mean_%i.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads,itera))
#     #     se.to_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/sd_%i.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads,itera))

    return dist_ts, y_test_inv_scaled, y_test_inv_scaled

def test(transformer, test_y_targets, MC_iterations, n_test, test_X, train_y_mean, train_y_std):
    # Dataframe to keep results
    dist = np.ones(shape=(MC_iterations, (test_y_targets.shape[1]-1)*3))
    dist_ts = pd.DataFrame(np.ones(shape=(n_test, test_y_targets.shape[1]-1)))
    ta=tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True)
    tu = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    ta=ta.write(0, (test_y_targets[:,:1,:]) )
    ta=tf.transpose(ta.stack(), perm=[1,0,2,3])
    ta=ta.reshape(ta.shape[0], ta.shape[1], ta.shape[-1])
    print('ta.shape', ta.shape)

    for i in range(0, test_y_targets.shape[1]-1):
        print('i',i)
        ta, attention_weights = transformer(((test_X), ta), False, False, True, False)
        tu = tu.write(i, ta[:,:,:1])
        if i < test_y_targets.shape[1]-2:
            ta = tf.concat([ta,  test_y_targets[:,i+1:i+2,1:]], -1)        
        print('ta.shape', ta.shape)
    test_output = tf.transpose(tu.stack(), perm=[1,0,2,3])
        # Inverse normalize
    df_train_mean_y=((train_y_mean))
    df_train_std_y=((train_y_std))
    y_hat = K.eval(test_output) # to transform from tensor to numpy array
    print('y_hat:', y_hat.shape)
    y_hat = (y_hat.reshape(y_hat.shape[0], y_hat.shape[1]))
    y_test = K.eval(tf.cast(test_y_targets[:,1:,:1], dtype=tf.float32))
    print('y_test:', y_test.shape)
    y_test = (y_test.reshape(y_test.shape[0], y_test.shape[1]))
    y_hat_inv_scaled = pd.DataFrame(y_hat * df_train_std_y[0] + df_train_mean_y[0])
    y_test_inv_scaled = pd.DataFrame(y_test * df_train_std_y[0] + df_train_mean_y[0])
    print('shape!!',y_hat_inv_scaled.shape)

    for i in range(test_y_targets.shape[1]-1):
        dist_ts.iloc[:,i] = y_hat_inv_scaled.iloc[:,i]#     dist_ts.to_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/95PPU_test.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads))

#     for i in range(test_y_targets.shape[1]-1):
#         y_test_inv_scaled.iloc[:,i].to_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/y_test_%i.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads, i))

#     for i in range((test_y_targets.shape[1]-1)):
#         pred = pd.read_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/95PPU_test.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads)).iloc[:,1:]#*0.028316847 #(cfs to cms)
#         data = pd.read_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/y_test_%i.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads, i)).iloc[:,1:]#*0.028316847 #(cfs to cms)


#         mean, lb, ub, median, se = TNN_functions.stats(pred, 0.95)
#     mean = pd.DataFrame(mean)
#     #     se = pd.DataFrame(se)

#     #         mean = np.reshape(mean, (-1, 1))
#     #         se = np.reshape(se, (-1, 1))
#     mean.to_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/mean_%i.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads,itera))
#     #     se.to_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/sd_%i.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads,itera))

    return dist_ts, y_hat_inv_scaled, y_test_inv_scaled


def test_informer(transformer, test_y_targets, MC_iterations, n_test, test_X, train_y_mean, train_y_std, steps):
    # Dataframe to keep results
    dist = np.ones(shape=(MC_iterations, (test_y_targets.shape[1]-steps)*3))
    dist_ts = pd.DataFrame(np.ones(shape=(n_test, test_y_targets.shape[1]-steps)))
#     ta=tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True)
#     tu = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    ta=(test_y_targets[:,:,:].copy()) 
    ta[:,-steps:,:1] = 0
    ta = tf.convert_to_tensor(ta)
#     ta=tf.transpose(ta.stack(), perm=[1,0,2,3])
#     ta=ta.reshape(ta.shape[0], ta.shape[1], ta.shape[-1])
#     print('ta.shape', ta.shape)

    ta, attention_weights = transformer(((test_X), ta), False, False, True, False,steps)
#     tu = tu.write(i, ta[:,-steps:,:1])
    test_output = ta#[:,-steps:,:1]
#     print('test:', test_output)
#     test_output = tf.transpose(tu.stack(), perm=[1,0,2,3])
        # Inverse normalize
    df_train_mean_y=((train_y_mean))
    df_train_std_y=((train_y_std))
    y_hat = K.eval(test_output) # to transform from tensor to numpy array
#     print('y_hat:', y_hat.shape)
    y_hat = (y_hat.reshape(y_hat.shape[0], y_hat.shape[1]))
    y_test = test_y_targets[:,-steps:,:1].copy()
    y_test = (y_test.reshape(y_test.shape[0], y_test.shape[1]))
#     print(y_test)
    y_hat_inv_scaled = pd.DataFrame(y_hat * df_train_std_y[0] + df_train_mean_y[0])
    y_test_inv_scaled = pd.DataFrame(y_test * df_train_std_y[0] + df_train_mean_y[0])
#     print('shape!!',y_hat_inv_scaled.shape)

#     for i in range(test_y_targets.shape[1]-steps):
    dist_ts= y_hat_inv_scaled
    #     dist_ts.to_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/95PPU_test.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads))

#     for i in range(test_y_targets.shape[1]-1):
#         y_test_inv_scaled.iloc[:,i].to_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/y_test_%i.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads, i))

#     for i in range((test_y_targets.shape[1]-1)):
#         pred = pd.read_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/95PPU_test.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads)).iloc[:,1:]#*0.028316847 #(cfs to cms)
#         data = pd.read_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/y_test_%i.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads, i)).iloc[:,1:]#*0.028316847 #(cfs to cms)


#         mean, lb, ub, median, se = TNN_functions.stats(pred, 0.95)
#     mean = pd.DataFrame(mean)
#     #     se = pd.DataFrame(se)

#     #         mean = np.reshape(mean, (-1, 1))
#     #         se = np.reshape(se, (-1, 1))
#     mean.to_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/mean_%i.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads,itera))
#     #     se.to_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/sd_%i.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads,itera))

    return dist_ts, y_hat_inv_scaled, y_test_inv_scaled

def test_hybrid_informer(transformer, test_y_targets, MC_iterations, n_test, test_X, train_y_mean, train_y_std, steps):
    # Dataframe to keep results
    dist = np.ones(shape=(MC_iterations, (test_y_targets.shape[1]-steps)*3))
    dist_ts = pd.DataFrame(np.ones(shape=(n_test, test_y_targets.shape[1]-steps)))
#     ta=tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True)
#     tu = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    ta=(test_y_targets[:,:-1,:].copy()) 
    ta[:,-steps+1:,:1] = 0
    
#     ta=tf.transpose(ta.stack(), perm=[1,0,2,3])
#     ta=ta.reshape(ta.shape[0], ta.shape[1], ta.shape[-1])
#     print('ta.shape', ta.shape)

    ta, attention_weights = transformer(((test_X), ta), False, False, True, False,steps)
#     tu = tu.write(i, ta[:,-steps:,:1])
    test_output = ta#[:,-steps:,:1]
#     print('test:', test_output)
#     test_output = tf.transpose(tu.stack(), perm=[1,0,2,3])
        # Inverse normalize
    df_train_mean_y=((train_y_mean))
    df_train_std_y=((train_y_std))
    y_hat = K.eval(test_output) # to transform from tensor to numpy array
    print('y_hat:', y_hat.shape)
    y_hat = (y_hat.reshape(y_hat.shape[0], y_hat.shape[1]))
    y_test = test_y_targets[:,-steps:,:1].copy()
    y_test = (y_test.reshape(y_test.shape[0], y_test.shape[1]))
#     print(y_test)
    y_hat_inv_scaled = pd.DataFrame(y_hat * df_train_std_y[0] + df_train_mean_y[0])
    y_test_inv_scaled = pd.DataFrame(y_test * df_train_std_y[0] + df_train_mean_y[0])
    print('shape!!',y_hat_inv_scaled.shape)

#     for i in range(test_y_targets.shape[1]-steps):
    dist_ts= y_hat_inv_scaled
    #     dist_ts.to_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/95PPU_test.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads))

#     for i in range(test_y_targets.shape[1]-1):
#         y_test_inv_scaled.iloc[:,i].to_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/y_test_%i.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads, i))

#     for i in range((test_y_targets.shape[1]-1)):
#         pred = pd.read_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/95PPU_test.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads)).iloc[:,1:]#*0.028316847 #(cfs to cms)
#         data = pd.read_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/y_test_%i.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads, i)).iloc[:,1:]#*0.028316847 #(cfs to cms)


#         mean, lb, ub, median, se = TNN_functions.stats(pred, 0.95)
#     mean = pd.DataFrame(mean)
#     #     se = pd.DataFrame(se)

#     #         mean = np.reshape(mean, (-1, 1))
#     #         se = np.reshape(se, (-1, 1))
#     mean.to_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/mean_%i.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads,itera))
#     #     se.to_csv(path_to_the_directory+variable+'_'+station+'_%isteps_%ilag_95PPU_%iemb_%ibatch_%idff_%ilayers_%iheads/sd_%i.csv' %(steps, t_lag, emb_dim_enc, batch_size, dff, num_layers, num_heads,itera))

    return dist_ts, y_hat_inv_scaled, y_test_inv_scaled
