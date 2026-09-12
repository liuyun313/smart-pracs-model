# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 20:57:54 2025

@author: huawei
"""

import os
import pandas as pd
import numpy as np
from keras._tf_keras.keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

# read data        
directory = os.getcwd()

f = '../data/data.xlsx'

data = pd.read_excel(f, sheet_name='data')
        
X, target = data.iloc[:,1:-1], data.iloc[:,-1] 

genes_name = X.columns

target = (target - np.mean(target)) / np.std(target)
target = (target - min(target))/(max(target) - min(target))

X_docs = []
x = np.array(X)

for i in range(len(X)):
    str_comb = ''        
    for j in range(13):
        if x[i,j] == 1:
            if j == 12:
                str_comb = str_comb + X.columns[j].split('.')[1]
            else:
                str_comb = str_comb + X.columns[j].split('.')[1] + ' '
        else:
            if j == 12:
                str_comb = str_comb + X.columns[j].split('.')[1][::-1]
            else:
                str_comb = str_comb + X.columns[j].split('.')[1][::-1] + ' '
            
    
    X_docs.append(str_comb)
    
# # Encode the sentences above by index; one_hot encoding maps to [1, vocab_size], excluding 0
vocab_size = 100

encoded_docs = [one_hot(d, vocab_size) for d in X_docs]
    
# Encode the text into numeric format and pad to the same length (here the length is set to 4, padding with zeros at the end),
# which is also why the one-hot encoding above does not map to 0.

padded_docs = pad_sequences(encoded_docs, maxlen=13, padding='post')

padded_docs = pd.DataFrame(padded_docs)

padded_docs['y'] = target

# padded_docs.to_csv('encoded_known.csv', index=True)
  

# In[] unknown data

def get_binary_combinations_recursive(n):
    if n == 0:
        return []

    if n == 1:
        return ['0', '1']

    sub_combinations = get_binary_combinations_recursive(n-1)    
    combinations = []
    for sub_combination in sub_combinations:
        combinations.append('0' + sub_combination)
        combinations.append('1' + sub_combination)

    return combinations

n = 13
combinations = get_binary_combinations_recursive(n)

X = []

for i in combinations:
    lst = [int(char) for char in i]
    X.append(lst)
    
X = np.array(X)    

# drep

f = './data/data.xlsx'

data = pd.read_excel(f, sheet_name='data')

X_  = data.iloc[:,1:-1]
X_ = np.array(X_)

X_derep = []
for i in range(np.size(X, 0)):
    
    iii = np.all(X_ == X[i,:], axis=1)
    
    if True not in iii:
        X_derep.append(X[i,:])


X_derep = np.array(X_derep)  

X_derep = pd.DataFrame(X_derep)

X_derep.to_csv('encoded_unknown_0_1.csv', index=True)


X_docs = []
x = np.array(X_derep)

for i in range(len(X_derep)):
    str_comb = ''        
    for j in range(13):
        if x[i,j] == 1:
            if j == 12:
                str_comb = str_comb + genes_name[j].split('.')[1]
            else:
                str_comb = str_comb + genes_name[j].split('.')[1] + ' '
        else:
            if j == 12:
                str_comb = str_comb + genes_name[j].split('.')[1][::-1]
            else:
                str_comb = str_comb + genes_name[j].split('.')[1][::-1] + ' '
            
    
    X_docs.append(str_comb)
    
# # Encode the sentences above by index; one_hot encoding maps to [1, vocab_size], excluding 0
vocab_size = 100

encoded_docs = [one_hot(d, vocab_size) for d in X_docs]
    
# Encode the text into numeric format and pad to the same length (here the length is set to 4, padding with zeros at the end),
# which is also why the one-hot encoding above does not map to 0.

padded_docs = pad_sequences(encoded_docs, maxlen=13, padding='post')

X_derep = pd.DataFrame(padded_docs)

# X_derep.to_csv('encoded_unknown.csv', index=True)


