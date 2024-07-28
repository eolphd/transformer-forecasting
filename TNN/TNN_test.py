import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from pandas import read_csv

import TNN_functions

def test_informer(transformer, test_y_targets, MC_iterations, n_test, test_X, train_y_mean, train_y_std, steps):
    # Dataframe to keep results
    dist = np.ones(shape=(MC_iterations, (test_y_targets.shape[1]-steps)*3))
    dist_ts = pd.DataFrame(np.ones(shape=(n_test, test_y_targets.shape[1]-steps)))
    ta=(test_y_targets[:,:,:].copy()) 
    ta[:,-steps:,:1] = 0
    ta = tf.convert_to_tensor(ta)

    ta, attention_weights = transformer(((test_X), ta), False, False, True, False,steps)
    test_output = ta#[:,-steps:,:1]

        # Inverse normalize
    df_train_mean_y=((train_y_mean))
    df_train_std_y=((train_y_std))
    y_hat = K.eval(test_output) 
    y_hat = (y_hat.reshape(y_hat.shape[0], y_hat.shape[1]))
    y_test = test_y_targets[:,-steps:,:1].copy()
    y_test = (y_test.reshape(y_test.shape[0], y_test.shape[1]))
    y_hat_inv_scaled = pd.DataFrame(y_hat * df_train_std_y[0] + df_train_mean_y[0])
    y_test_inv_scaled = pd.DataFrame(y_test * df_train_std_y[0] + df_train_mean_y[0])

    dist_ts= y_hat_inv_scaled

    return dist_ts, y_hat_inv_scaled, y_test_inv_scaled
