# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:20:46 2025

@author: 58211
"""
import pandas as pd
import os
import numpy as np
from keras.models import load_model
from training import spearman_loss, spearman_correlation, PositionEmbedding
from sklearn.model_selection import train_test_split
import scipy.stats as stats


# In[] load trained model
directory = os.getcwd()

file = os.path.join(directory, "../models/best_model.keras")

model = load_model(file,
                    custom_objects={'spearman_loss': spearman_loss,
                                    'spearman_correlation': spearman_correlation, 
                                    'CustomLayer': PositionEmbedding}
                    ) 



# In[]
f = 'encoded_unknown.csv'
X_derep = pd.read_csv(f)
X = X_derep.iloc[:,1:14]
y_score = model.predict(X)
