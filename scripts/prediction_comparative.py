# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:20:46 2025

@author: 58211
"""
import pandas as pd
import os
#from keras.models import load_model
from joblib import load


# In[] load regr model
directory = os.getcwd()



#model = load('../models/model_gbr.joblib')
model = load('../models/model_rf.joblib')
model = load('../models/model_svr.joblib')
model = load_model('../models/lstm.keras')
model = load_model('../models/cnn.keras')



# In[]
f = '../data/encoded_unknown_0_1.csv'
X_derep = pd.read_csv(f)
X = X_derep.iloc[:,1:14]
y_score = model.predict(X)


