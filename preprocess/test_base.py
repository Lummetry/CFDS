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

import numpy as np
import scipy.sparse as sparse
import pandas as pd

if __name__ == '__main__':
  TEST_SIZE = 0.2
  LAMBDA = 1
  y = np.load('datasets/y.npy')
  csr = sparse.load_npz('datasets/x_csr.npz')
  X = csr.toarray()
  
  # X[:,0] =  (X[:,0] -  X[:,0].mean()) / X[:,0].std()
  X[:,0] =  (X[:,0] -  X[:,0].min()) / (X[:,0].max() - X[:,0].min())

  X = X[:,:100]
  
  
  
  train_sample = np.random.choice(
    [0,1], 
    replace=True, 
    size=X.shape[0],
    p=[TEST_SIZE, 1-TEST_SIZE],
    ).astype(bool)
  
  X_train = X[train_sample]
  y_train = y[train_sample]
  X_test = X[~train_sample]
  y_test =y[~train_sample]
  
  theta = np.linalg.inv(X_train.T.dot(X_train) + LAMBDA * np.eye(X_train.shape[1])).dot(X_train.T).dot(y_train)
  
  y_pred = X_test.dot(theta).round(0)
  
  MAE = np.abs(y_pred - y_test).mean()
  print(MAE)
  
  df_res = pd.DataFrame({'pred':y_pred, 'gold':y_test})
  print(df_res.head(10))
  