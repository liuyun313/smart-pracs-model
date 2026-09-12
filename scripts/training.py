# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 11:10:21 2025

@author: 58211
"""


import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

from keras.models import Sequential
from keras.layers import Embedding
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable() 
def spearman_correlation(y_true, y_pred):
    # Compute the rank of each sample
    def compute_rank(x):
        # Get sorting indices (ascending order)
        sorted_indices = tf.argsort(x, axis=-1)
        # Get the sorted position of the original indices
        ranks = tf.argsort(sorted_indices, axis=-1) + 1  # +1 to convert to 1-based ranks
        return tf.cast(ranks, tf.float32)

    # Handle inputs of different shapes
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    # Compute ranks
    rank_true = compute_rank(y_true)
    rank_pred = compute_rank(y_pred)

    # Compute the Pearson correlation coefficient of the ranks
    mx = tf.reduce_mean(rank_true)
    my = tf.reduce_mean(rank_pred)
    xm = rank_true - mx
    ym = rank_pred - my
    r_num = tf.reduce_sum(xm * ym)
    r_den = tf.sqrt(tf.reduce_sum(xm**2) * tf.reduce_sum(ym**2))
    r = r_num / (r_den + tf.keras.backend.epsilon())
    return r

@register_keras_serializable()
class PositionEmbedding(layers.Layer):
    """Learnable position embedding layer"""
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_emb(positions)
        return x + positions
    
    # Ensure the serialization method includes all parameters
    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "d_model": self.d_model
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)    
    

   
# Differentiable ranking approximation (using neural network ranking tricks)
def differentiable_rank(x, alpha=0.1):
    """
    Sort using a Sigmoid approximation based on pairwise comparison
    Args:
        x : input tensor (batch_size, 1)
        alpha : temperature coefficient (controls the degree of approximation)
    """
    x = tf.reshape(x, (-1, 1))  # Ensure the shape is (batch_size, 1)
    pairwise_diff = x - tf.transpose(x)  # Compute pairwise differences
    pairwise_rank = tf.sigmoid(pairwise_diff / alpha)  # Approximate step function
    return tf.reduce_sum(pairwise_rank, axis=1)  # Compute the approximate rank of each element

# Differentiable Spearman loss function

@register_keras_serializable()  # <-- legacy decorator
def spearman_loss(y_true, y_pred, alpha=0.1):
    """ 
    Differentiable Spearman correlation coefficient loss
    Args:
        alpha : hyperparameter controlling the ranking approximation (recommended 0.1-1.0)
    """
    # Flatten
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    
    # Compute approximate ranks
    rank_true = differentiable_rank(y_true, alpha)
    rank_pred = differentiable_rank(y_pred, alpha)
    
    # Standardize ranks
    rank_true = (rank_true - tf.reduce_mean(rank_true)) / tf.math.reduce_std(rank_true)
    rank_pred = (rank_pred - tf.reduce_mean(rank_pred)) / tf.math.reduce_std(rank_pred)
    
    # Compute covariance
    cov = tf.reduce_mean(rank_true * rank_pred)
    
    # Spearman correlation coefficient
    spearman = cov / (tf.math.reduce_std(rank_true) * tf.math.reduce_std(rank_pred) + 1e-8)
    
    # Return the negative correlation coefficient (minimizing the loss)
    return 1.0 - spearman



def build_regression_model(seq_len, vocab_size, embedding_dim=128, max_len=50):
    """
    Build a regression model
    Args:
        seq_len: input sequence length
        vocab_size: vocabulary size
        embedding_dim: embedding dimension
        max_len: maximum position encoding length (must be >= seq_len)
    """
    dropout_rate = 0.3
    # Input layer
    inputs = layers.Input(shape=(seq_len,))
    
    # Embedding layer + position encoding
    x = layers.Embedding(input_dim=vocab_size+1, output_dim=embedding_dim)(inputs)
    x = PositionEmbedding(max_len, embedding_dim)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Convolutional layer
    x = layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)    
        
    # Self-attention layer
    x = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)  # use multi-head attention
    x = layers.Dropout(dropout_rate)(x)
        
    # Global average pooling
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout_rate)(x) 
    
    # Fully connected layer
    outputs = layers.Dense(1, activation='relu')(x)  # regression output layer
    
    # Build the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=100,
        decay_rate=0.9)  
            
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_scheduler)    
    
    model.compile(optimizer=optimizer, 
                  loss=spearman_loss,
                  metrics=[spearman_correlation]
                  )
    
    return model

# Example usage
if __name__ == "__main__":
    
    directory = os.getcwd()

    f = '../data/encoded_known.csv'

    data = pd.read_csv(f)
            
    X, target = data.iloc[:,1:-1], data.iloc[:,-1] 
     
    
    # Parameter settings
    seq_len = 13     # input sequence length
    embedding_dim = 16  # embedding dimension
    max_len = 13     # maximum position encoding length (must be >= seq_len)
    
    vocab_size = 100
    spearman_v = 0.0
    
    
    # Build the model
    model = build_regression_model(seq_len, vocab_size, embedding_dim, max_len)
    
    # View the model structure
    model.summary()
    
    # Add the early stopping callback
    early_stopping = callbacks.EarlyStopping(
        monitor='val_spearman_correlation',  #spearman_correlation
        mode='max',                  # maximize the correlation coefficient
        patience=20,                  # increase the patience value
        min_delta=0.001,             # set the minimum improvement threshold
        restore_best_weights=True
    )
    
    # Create the callback list (extensible to add other callbacks)
    callbacks_list = [early_stopping]
    
    # Train the model (add the validation_split parameter)
    X_train, X_test, y_train, y_test = train_test_split(X, target, 
                                                        test_size=0.2, 
                                                        random_state=0)  

    history = model.fit(
        X_train, y_train,
        epochs=500,           # set a large number of epochs
        batch_size=32,
        callbacks=callbacks_list,  # add callbacks
        validation_split=0.15  # use 20% of the data as the validation set
        # 
    )
       
        
# In[]
    y_hat = model.predict(X_train)
    # y_hat = (y_hat - min(y_hat))/(max(y_hat) - min(y_hat))
    y_hat = np.array(y_hat)
    y_train = np.array(y_train)
    plt.figure(figsize=(12, 4))
    plt.title("MSE: {:.4f}, R2: {:.4f}".format(mean_squared_error(y_hat, y_train), r2_score(y_train, y_hat)))
    plt.plot(y_train, marker='.', color='red', label="Real value")
    plt.plot(y_hat, marker='.', color='blue', label="Predicted value")
    plt.legend()
    plt.show()
    
    import scipy.stats as stats
    
    spearman_v, p_value = stats.spearmanr(y_hat, y_train)
    
    # Output the correlation coefficient and p-value
    print("Spearman correlation coefficient:", spearman_v)
    print("p-value:", p_value)

# In[]
    y_hat = model.predict(X_test)
    # y_hat = (y_hat - min(y_hat))/(max(y_hat) - min(y_hat))
    y_hat = np.array(y_hat)
    y_test = np.array(y_test)
    plt.figure(figsize=(12, 4))
    plt.title("MSE: {:.4f}, R2: {:.4f}".format(mean_squared_error(y_hat, y_test), r2_score(y_test, y_hat)))
    plt.plot(y_test, marker='.', color='red', label="Real value")
    plt.plot(y_hat, marker='.', color='blue', label="Predicted value")
    plt.legend()
    plt.show()
    
    import scipy.stats as stats
    
    spearman_v, p_value = stats.spearmanr(y_hat, y_test)
    
    # Output the correlation coefficient and p-value
    print("Spearman correlation coefficient:", spearman_v)
    print("p-value:", p_value)
    
# In[] save model

    # model_name = 'best_model'.keras'
    # model.save(model_name)
    
    
    
    