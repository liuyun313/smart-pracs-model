# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:33:12 2025

@author: 58211
"""


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, LSTM
from keras.optimizers import SGD
import tensorflow as tf
from joblib import dump


# In[] read data        
directory = os.getcwd()

data = pd.read_excel('../data/data.xlsx', sheet_name='data')

X, target = data.iloc[:,1:-1], data.iloc[:,-1] 


target = (target - np.mean(target)) / np.std(target)
target = (target - min(target))/(max(target) - min(target))

X_train, X_test, y_train, y_test = train_test_split(X, target, 
                                                    test_size=0.2, 
                                                    random_state=0)  



# In[] regression model CNN
'''
from keras.callbacks import EarlyStopping, ModelCheckpoint

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)  # split into training and test sets

nb_features = X.shape[1]

model = Sequential()
n_filters = 16
model.add(Convolution1D(n_filters, 7, input_shape=(nb_features, 1)))
model.add(Activation('relu'))
model.add(Convolution1D(n_filters, 5, input_shape=(nb_features - 3 + 1 , 1)))
model.add(Activation('relu'))
model.add(Convolution1D(n_filters, 3, input_shape=(nb_features - 6 + 1 , 1)))
#model.add(Activation('relu'))
model.add(Flatten())
#model.add(Dropout(0.8))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10,
        decay_rate=0.9)

optimizer = tf.keras.optimizers.Adam(learning_rate = lr_scheduler)    

model.compile(loss='MeanSquaredError', optimizer=optimizer, metrics=['MeanSquaredError'])

early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=50, 
        verbose=0, 
        mode='auto',
        restore_best_weights = True
        )

nb_epoch = 2000


model.fit(X_train, y_train, epochs = nb_epoch, 
          batch_size=8, 
          callbacks=[early_stopping], 
          validation_data=(X_val, y_val))#   sample_weight = sample_weights)


y_hat = model.predict(X_test)
model.save('cnn.keras')
'''

# In[] regression model svr
'''
from sklearn.svm import SVR

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': [0.01, 0.1, 1]
}
grid_search = GridSearchCV(SVR(), param_grid, cv=5)
grid_search.fit(X_train, y_train)#, sample_weight = sample_weights

# Get the best model
best_svr = grid_search.best_estimator_

y_hat = best_svr.predict(X_test)

# save model
#dump(best_svr, 'model_svr.joblib')
'''
# In[] regression model gbr
'''
from sklearn.ensemble import GradientBoostingRegressor

param_grid = {
    'n_estimators': range(5, 50, 5),
    'max_depth':range(3,14,2), 
    'min_samples_split':range(2,20,2)
}

grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=5)
grid_search.fit(X_train, y_train)#, sample_weight = sample_weights)

# Get the best model
best_svr = grid_search.best_estimator_

y_hat = best_svr.predict(X_test)

# save model
dump(best_svr, 'model_gbr.joblib')
'''
# In[] regression model rf
'''
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    'n_estimators': range(5, 30, 5),
    'max_depth':range(3,14,2), 
    'min_samples_split':range(2,20,2),
    'max_features':range(1,13,2)
}

grid_search = GridSearchCV(RandomForestRegressor(oob_score=True), param_grid, cv=5)
grid_search.fit(X_train, y_train)#, sample_weight = sample_weights)

# Get the best model
best_svr = grid_search.best_estimator_

feature_imp = best_svr.feature_importances_

y_hat = best_svr.predict(X_test)

# save model
dump(best_svr, 'model_rf.joblib')
'''

# In[] LSTM
'''
from keras.callbacks import EarlyStopping, ModelCheckpoint

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)  # split into training and test sets

nb_features = X.shape[1]

model = Sequential()

model.add(LSTM(128, input_shape=(13, 1), return_sequences=True))
model.add(LSTM(128, return_sequences=False))

model.add(Flatten())
#model.add(Dropout(0.8))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10,
        decay_rate=0.9)

optimizer = tf.keras.optimizers.Adam(learning_rate = lr_scheduler)    

model.compile(loss='MeanSquaredError', optimizer=optimizer, metrics=['MeanSquaredError'])

early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=50, 
        verbose=0, 
        mode='auto',
        restore_best_weights = True
        )

nb_epoch = 2000


model.fit(X_train, y_train, epochs = nb_epoch, 
          batch_size=8, 
          callbacks=[early_stopping], 
          validation_data=(X_val, y_val))#   sample_weight = sample_weights)


y_hat = model.predict(X_test)
model.save('lstm.keras')
'''

# In[] evaluation

y_hat = np.array(y_hat)
y_test = np.array(y_test)
plt.figure(figsize=(12, 4))
plt.title("MSE: {:.4f}, R2: {:.4f}".format(mean_squared_error(y_hat, y_test), r2_score(y_test, y_hat)))
plt.plot(y_test, marker='.', color='red', label="Real value")
plt.plot(y_hat, marker='.', color='blue', label="Predicted value")
plt.legend()
plt.show()

import scipy.stats as stats

spearman_correlation, p_value = stats.spearmanr(y_hat, y_test)

# Output the correlation coefficient and p-value
print("Spearman correlation coefficient:", spearman_correlation)
print("p-value:", p_value)