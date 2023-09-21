#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import logging
import os
import pathlib
import re
import sys
import time
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import scipy.stats as st
from scipy import stats
import math
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

# MASKING
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


# SCALED DOT PRODUCT ATTENTION
def scaled_dot_product_attention(q, k, v, mask):
    """
    Self-Attention.
    Scaled dot product attention takes in a query Q, a key K, a value V, and a mask as inputs to return representations of the sequence
    
    Attention(Q,K,V) = softmax(QK.T/sqrt(dk) + M)V
    
    Q: matrix of queries
    K: matrix of keys
    V: matrix of values
    M: optional mask
    dk: dimension of the keys, used to scale everything down so the softmax does not explode.
    """
    
    """  
    Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """
    
    # Q * K.T
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    """
    As the softmax normalization is done on K, its values decide the amount of importance given to Q.
    The output represents the multiplication of the attention weights and the V (value) vector. 
    This ensures that the words you want to focus on are kept as is and the irrelevant words are flushed out.'''   
    """   
    
    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    # multiply by the matrix of values V
    output = tf.matmul(attention_weights, v, transpose_b=False)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention consists of four parts:
    
    1. Linear layers and split into heads
    2. Scaled dot-product attention
    3. Concatenation of heads
    4. Final linear layer
    
    Each multi-head attention block gets three inputs: Q, K, V. These are put through linear (Dense) layers and split up into multiple heads
    
    the scaled_dot_product_attention defined above is applied to each head (broadcasted for efficiency). The attention output for each head
    is then concatenated and put through a final Dense layer.
    
    Instead of one single attention head, Q, K, and V are split into multiple heads because it allows the model to jointly attend
    to information from different representation subspaces at different positions.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth)) # Split the last dimension into (num_heads, depth).
        return tf.transpose(x, perm=[0, 2, 1, 3]) # Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)

    def split_heads_queries(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth)) # Split the last dimension into (num_heads, depth).
        return tf.transpose(x, perm=[0, 2, 1, 3]) # Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    
    def call(self, v, k, q, mask):
        
        batch_size = K.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads_queries(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)    
        
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # output.shape (batch_size, seq_len_q, d_model)
                                               # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        return output, attention_weights

# POINT WISE FEED FORWARD NETWORK
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

# ENCODER
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate):
        super(EncoderLayer, self).__init__()
        """
        Each encoder layer consists of sublayers:
        
        1. Multi-head attention (with padding mask)
        2. Point wise feed forward networks.
        
        Each of these sublayers has a residual connection around it followed by a layer normalization. Residual connections
        help in avoiding the vanishing gradient problem in deep networks.
        
        The output of each sublayer is LayerNorm(x + sublayer(x)). The normalization is done on the d_model (last) axis. 
        There are N encoder layers in the transformer.
        
        Arguments:
        dff - fully connected dimension
        """

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        """
        Forward pass for the Encoder layer
        
        Arguments:
        x - tensor of shape (batch_size, input_seq_len, d_model)
        training - boolean set to true to activate the training mode for dropout layers
        mask - boolean mask to ensure that the padding is not treated as part of the input
        
        Returns:
        out2 - tensor of shape (batch_size, input_seq_len, d_model)
        """
        # 1. calculate self-attention using multi-head attention
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        
        # 2. apply dropout to the self-attention output
        attn_output = self.dropout1(attn_output, training=training)
        
        # 3. apply layer normalization on the sum of the input and the attention output (residual connection) to get the output
        # of the multi-head attention layer
        out1 = self.layernorm1(x * attn_output)  # (batch_size, input_seq_len, d_model)
        
        # 4. pass the output of the mha layer through a ffn
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        
        # 5. apply dropout layer to ffn output
        ffn_output = self.dropout2(ffn_output, training=training)
        
        # 6. apply layer normalization on sum of the output from mha and ffn output to get the  output of the encoder layer
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
               vocab_size, rate): 
        super(Encoder, self).__init__()
        
        """
#         Encoder
#         The Encoder consists of:

#         1. Input Embedding
#         2. Positional Encoding 
#         3. N encoder layers
        
#         The output of the encoder is the input to the decoder.
#         """

        self.d_model = d_model
        self.num_layers = num_layers
        
#         self.embedding = tf.keras.layers.Embedding(vocab_size, self.d_model)

        self.pos_encoding = positional_encoding(vocab_size, self.d_model)
#         self.time_encoding = Time2Vector(vocab_size)
#         self.concat = tf.keras.layers.concatenate()

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask):
        """
        Forward pass for the encoder
        
        Arguments:
        x - tensor of shape (batch_size, input_seq_len)
        training - boolean set to true to activate the training model for dropout layers
        mask - boolean mask to ensure that the padding is not treated as part of the input
        
        Returns:
        out2 - tensor of shape (batch_size, input_seq_len, d_model)
        """
        
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
#         x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        
        # scale embedding by multiplying it by the square root of the embedding dimension
#         x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # add the positional encoding to embedding
        x += self.pos_encoding[:, :seq_len, :]
#         x = tf.keras.layers.concatenate([x, t], axis=-1)
#         x = tf.keras.layers.Dense(x)(x.shape)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)
    
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate): #rate=0.1):
        super(DecoderLayer, self).__init__()
        
        """
        Decoder layer
        is composed by two multi-head attention blocks, one that takes the new input and uses self-attention, and the other one that
        combines it with the output of the encoder, followed by a fully connected block.
        
        Each decoder layer consists of sublayers:

        1. Masked multi-head attention (with look ahead mask and padding mask).
        2. Multi-head attention (with padding mask). V (value) and K (key) receive the encoder output as inputs. 
            Q (query) receives the output from the masked multi-head attention sublayer.
        3. Point wise feed forward networks.
        
        Each of these sublayers has a residual connection around it followed by a layer normalization. 
        The output of each sublayer is LayerNorm(x + Sublayer(x)). The normalization is done on the d_model (last) axis.

        There are N decoder layers in the transformer.

        As Q receives the output from decoder's first attention block, and K receives the encoder output, 
        the attention weights represent the importance given to the decoder's input based on the encoder's output. 
        In other words, the decoder predicts the next word by looking at the encoder output and self-attending to its own output. 
        See the demonstration above in the scaled dot product attention section.
        
        Arguments:
        dff - fully connected dimension
        """

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        """
        Forward pass for the decoder layer.
        
        Arguments:
        x - tensor of shape (batch_size, target_seq_len, d_model)
        enc_output - tensor of shape (batch_size, input_seq_len, d_model)
        training - boolean, set to true to activate the training mode for dropout layers
        look_ahead_mask - boolean mask for the target_input
        padding_mask - boolean mask for the second multihead attention layer
        
        Returns:
        out3 - tensor of shape (batch_size, target_seq_len, d_model)
        attn_weights_block1 - tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        attn_weights_block2 - tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """
        # BLOCK 1
        # 1. calculate self-attention and return attention scores as attn_weights_block1
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        
        # 2. apply dropout layer on the attention output
        attn1 = self.dropout1(attn1, training=training)
        
        # 3. apply layer normalization to the sum of the attention output (residual connection) and the input
        out1 = self.layernorm1(attn1 * x)

        # BLOCK 2
        # 4. calculate self-attention using the Q from the first block and K and V from the encoder output
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        
        # 5. apply dropout layer on the attention output
        attn2 = self.dropout2(attn2, training=training)
        
        # 6. apply layer normalization to the sum of the attention output and the output of the first block (residual connection)
        out2 = self.layernorm2(attn2 * out1)  # (batch_size, target_seq_len, d_model)
        
        # BLOCK 3
        # 7. pass the output of the second block through a ffn
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        
        # 8. apply a dropout layer to the ffn output
        ffn_output = self.dropout3(ffn_output, training=training)
        
        # 9. apply layer normalization to the sum of the ffn output and the output of the second block
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
               vocab_size, rate):
        super(Decoder, self).__init__()
        
        """
#         Decoder
#         The Decoder consists of:

#         Output Embedding (not included here due to the numerical nature of the time series)
#         Positional Encoding
#         N decoder layers
#         The target is put through an embedding which is summed with the positional encoding. 
#         The output of this summation is the input to the decoder layers. 
#         The output of the decoder is the input to the final linear layer.'''
#         """

        self.d_model = d_model
        
        self.num_layers = num_layers

#         self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
    
        self.pos_encoding = positional_encoding(vocab_size, d_model)

#         self.time_encoding = Time2Vector(vocab_size)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                               for _ in range(self.num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):
        """
        Forward pass for the decoder
        
        Arguments:
        x - tensor of shape (batch_size, target_seq_len, d_model)
        enc_output - tensor of shape (batch_size, input_seq_len, d_model)
        training - boolean set to true to activate the training mode for dropout layers
        look_ahead_mask - boolean mask for the target_input
        padding_mask - boolean mask for the second multihead attention layer
        
        Returns:
        x - tensor of shape (batch_size, target_seq_len, d_model)
        attention_weights - dictionary of tensors containing all the attention weights 
            each of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """

        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        # create word embeddings
#         x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        
        # scale embeddings by multiplying it by the square root of their dimension
#         x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        # calculate positional encoding and add to time series embedding
        x += self.pos_encoding[:, :seq_len, :]
#         x += self.time_encoding(x)
#         x += self.concatenate( [x, t], axis = -1)

        # apply a dropout layer to x
        x = self.dropout(x, training=training)

        # use a for loop to pass x through a stack of decoder layers and update attention weights
        for i in range(self.num_layers):
            # pass x and the encoder output through a stack of decoder layers
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)

            # update attention_weights dictionary with the attention weights of block 1 and block 2
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights
    
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, d_model_out, num_heads, dff, input_vocab_size, target_vocab_size, rate, steps): #input_vocab_size,
               #target_vocab_size, rate=0.1): #pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
            
        """
        Transformer
        Transformer consists of the encoder, decoder and a final linear layer. 
        The output of the decoder is the input to the linear layer and its output is returned.
        
        Flow of data through Transformer:
        1. input passes through an encoder, which is just repeated encoder layers (multi-head attention of input, ffn to help detect features)
        2. encoder output passes through a decoder, consisting of the decoder layers (multi-head attention on generated output, multi-head attention with the Q from the first multi-head attention layer and the K and V from the encoder)
        3. After the Nth decoder layer, two dense layers and a softmax are applied to generate prediction for the next output in the sequence.
        """
        self._embedding = tf.keras.layers.Dense(d_model)
        self._embedding_dec = tf.keras.layers.Dense(d_model_out)


        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, rate)

        self.final_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, training, enc_padding_mask,
           look_ahead_mask, dec_padding_mask,steps):
        """
        Forward pass for the entire Transformer.
        Arguments:
        inp - input tensor of shape (batch_size, input_seq_len, fully_connected_dim)
        tar - target tensor of shape (batch_size, input_seq_len, fully_connected_dim)
        training - boolean set to true to activate the training mode for dropout layers
        look_ahead_mask - boolean mask for the target
        dec_padding_mask - boolean mask for the second multihead attention layer
        attention_weights - dictionary of tensors containing all the attention weights for the decoder, each of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """
        inp, tar = inputs

        # embedding layer
        encoding = self._embedding(inp)
        
        decoding = self._embedding_dec(tar)

        enc_output = self.encoder(encoding,
                                  #inp, 
                                  training, 
                                  enc_padding_mask)  # (batch_size, inp_seq_len, d_model=fully_connected_dim=num_features)

        dec_output, attention_weights = self.decoder(
            decoding, 
            enc_output, 
            training,
            look_ahead_mask, 
            dec_padding_mask)  # dec_output.shape == (batch_size, tar_seq_len, d_model)

        final_output = self.final_layer(dec_output[:,-steps:,:1])  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

