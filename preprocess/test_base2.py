# -*- coding: utf-8 -*-
"""
Copyright 2019 Lummetry.AI (Knowledge Investment Group SRL). All Rights Reserved.


* NOTICE:  All information contained herein is, and remains
* the property of Knowledge Investment Group SRL.  
* The intellectual and technical concepts contained
* herein are proprietary to Knowledge Investment Group SRL
* and may be covered by Romanian and Foreign Patents,
* patents in process, and are protected by trade secret or copyright law.
* Dissemination of this information or reproduction of this material
* is strictly forbidden unless prior written permission is obtained
* from Knowledge Investment Group SRL.


@copyright: Lummetry.AI
@author: Lummetry.AI - Andrei
@project: 
@description: 

"""

import pandas as pd
import scipy.sparse as sparse
import os
from time import time
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

if __name__ == '__main__':
  np.random.seed(seed=1234)
  
  fields = [
    'price',
    'region',
    'year',
    'manufacturer',
    'model',
    'condition',
    'cylinders',
    'fuel',
    'odometer',
    'transmission',
    'drive',
    'size',
    'type',
    'paint_color'    
    ]
  
  df = pd.read_csv('datasets/cars_sample.csv.zip')
  
  categs = [
    'region',
    'year',
    'manufacturer',
    'model',
    'condition',
    'cylinders',
    'fuel',
    'transmission',
    'drive',
    'size',
    'type',
    'paint_color'    
    ]
  
  continuous = list(set(fields) - set(categs))
  
  LAMBDA = 0.5
  TEST_SIZE = 0.2
  
  dct_res = {'CATEG_COLS': [], 'LEARNING_TIME': [], 'MAE': [], 'DF_RES': [], 
             'THETA': [], 'MISSING_CATEGS': []}
  lst_categs = [categs] + list(map(list, combinations(categs, len(categs)-1)))
  for crt_categs in lst_categs:
    #make selection from a copy of initial df
    df_crt = df.copy()
    all_cols = continuous + crt_categs
    missing_categs = list(set(categs) - set(crt_categs))
    df_crt = df_crt[all_cols]
    
    #get_dummies
    df_sparse = pd.get_dummies(df_crt, columns=crt_categs)
    
    #prepare X, y 
    y = df_sparse.pop('price')
    X = df_sparse.values
    X[:,0] =  (X[:,0] -  X[:,0].min()) / (X[:,0].max() - X[:,0].min())
  
    #prepare X_train, y_train, X_test, y_test
    train_sample = np.random.choice(
      [0, 1], 
      replace=True, 
      size=X.shape[0],
      p=[TEST_SIZE, 1-TEST_SIZE],
      ).astype(bool)
    
    X_train = X[train_sample]
    y_train = y[train_sample]
    X_test  = X[~train_sample]
    y_test  = y[~train_sample]
    
    #learn weights
    start = time()
    theta = np.linalg.inv(X_train.T.dot(X_train) + LAMBDA * np.eye(X_train.shape[1])).dot(X_train.T).dot(y_train)
    stop  = time()
    learning_time = stop - start
    
    #make predictions and calculate error            
    y_pred = X_test.dot(theta).round(0)    
    MAE = np.abs(y_pred - y_test).mean()    
    df_res = pd.DataFrame({'pred':y_pred, 'gold':y_test})
        
    print('Results for X shape: {}'.format(X.shape))
    print(' All columns: {}'.format(', '.join(all_cols)))
    print(' Categ columns: {}'.format(', '.join(crt_categs)))
    print(' Missing categ columns: {}'.format(', '.join(missing_categs)))
    print(' Learning time: {}'.format(learning_time))
    print(' MAE: {}'.format(MAE))
    print(df_res.head(10))
    
    dct_res['CATEG_COLS'].append(crt_categs)
    dct_res['LEARNING_TIME'].append(learning_time)
    dct_res['MAE'].append(MAE)
    dct_res['DF_RES'].append(df_res)
    dct_res['THETA'].append(theta)
    dct_res['MISSING_CATEGS'].append(missing_categs)
  

  y = dct_res['MAE']
  x = np.arange(len(y))
  l = dct_res['MISSING_CATEGS']
  
  plt.plot(x, y, 'bo-')
  for idx, (x, y) in enumerate(zip(x, y)):
    plt.annotate('{} ({})'.format(int(y), l[idx]), 
                 (x, y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
  
  plt.grid()
  plt.show()
  
  
  df.nunique()
  
