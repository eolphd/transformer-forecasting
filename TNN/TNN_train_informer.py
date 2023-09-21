#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import time

import TNN_functions
import Transformer_addwgt


# In[ ]:


METRIC_MSE='mean_squared_error'
train_MSE_metric = tf.keras.metrics.MeanSquaredError('loss', dtype=tf.float32)
val_MSE_metric = tf.keras.metrics.MeanSquaredError()
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


# In[ ]:


def exponential_decay_fn(lr0, s, EPOCHS):
    return lr0 * 0.1 ** (EPOCHS / s)

def loss(y_real, y_hat):
    return tf.reduce_mean(tf.square(y_real - y_hat))

# @tf.function
def train_step(x_batch_train, tar_inp, tar_real, transformer, optimizer,steps):
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:
            
            # run the forward pass of the layer.
            # the operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
        
        output, attention_weights_train = transformer((x_batch_train, tar_inp), True, False, True, False, steps) # output for this minibatch
        # Compute the loss value for this minibatch
#         print('target real:',tar_real[:,-steps:,:1])
#         print('target simulated:', output[:,-steps:,:1])
        loss_value = loss(tf.cast(tar_real, dtype=tf.float32), output)
#         print('train results:' ,output[:,-steps:,:1])
    # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, transformer.trainable_weights)
        
    # Run one step of gradient descent (or optimizer used) by updating the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, transformer.trainable_weights))
            
    # Update training metric.
    train_MSE_metric.update_state(tar_real, output)
    return loss_value, attention_weights_train

# @tf.function
def test_step(x_batch_dev, tar_inp_dev, tar_real_dev, transformer,steps):
    dev_output, _ = transformer((x_batch_dev, tar_inp_dev), False, False, True, False, steps)
    
    
#     print('target real dev:',tar_real_dev[:,-steps:,:1])
#     print('target simulated dev:', dev_output[:,-steps:,:1])

    # Compute the loss value for this epoch
    loss_value_dev = loss(tf.cast(tar_real_dev, dtype=tf.float32), dev_output)
#     print('train results:' ,dev_output[:,-steps:,:1])
    # Update val metrics
    val_MSE_metric.update_state(tar_real_dev, dev_output)




# In[ ]:


def train_val(transformer, num_layers, emb_dim_enc, emb_dim_dec, num_heads, dff, train_X, train_y, dropout_rate, EPOCHS, train_dataset, dev_dataset, steps):
#     transformer = Transformer_addwgt.Transformer(
#                     num_layers=num_layers,
#                     d_model=emb_dim_enc,
#                     d_model_out=emb_dim_dec,
#                     num_heads=num_heads,
#                     dff=dff,
#                     input_vocab_size = train_X.shape[1],
#                     target_vocab_size=train_y.shape[1],
#     #                     features=X_out.shape[-1],
#                     rate=dropout_rate
#                     )
    learning_rate = exponential_decay_fn(lr0=0.01, s=20, EPOCHS=EPOCHS) #(lr0, s)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

        # Training and Validation 
    for epoch in range(EPOCHS):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
#             print('y_batch_train:',y_batch_train)
            tar_inp = y_batch_train[:,:,:]
            tinp = tar_inp.numpy()
            tinp[:,-steps:, :1]=0
            tar_inp = tf.convert_to_tensor(tinp)
            tar_real = y_batch_train[:,-steps:, :1]
#             print('tar real 1:', tar_real)

            loss_value = train_step(x_batch_train, tar_inp, tar_real, transformer, optimizer, steps)

        # Display metrics at the end of each epoch.
        train_MSE = train_MSE_metric.result()
        print("Training MSE: %.4f" % (float(train_MSE),))
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_MSE_metric.result(), step=epoch)

        # Reset training metrics at the end of each epoch
        train_MSE_metric.reset_states()

        # Run a validation/development loop at the end of each epoch.
        for x_batch_dev, y_batch_dev in dev_dataset:
            tar_inp_dev = y_batch_dev[:,:,:]
            tinp = tar_inp_dev.numpy()
            tinp[:,-steps:, :1]=0
            tar_inp_dev = tf.convert_to_tensor(tinp)   
            tar_real_dev = y_batch_dev[:,-steps:, :1]

            dev_output = test_step(x_batch_dev, tar_inp_dev, tar_real_dev, transformer,steps)

        val_MSE = val_MSE_metric.result()
        val_MSE_metric.reset_states()
        print("Development MSE: %.4f" % (float(val_MSE),))
        print("Time taken: %.2fs" % (time.time() - start_time)) 

