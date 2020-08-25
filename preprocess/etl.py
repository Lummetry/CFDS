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
import numpy as np

if __name__ == '__main__':
  if 'df' not in globals():
    FN_RAW = 'preprocess/vehicles.csv'
    if not os.path.isfile(FN_RAW):
      raise ValueError("Please download vehicles.csv or .zip file and unpack")
    # import car prices dataset https://github.com/AustinReese/UsedVehicleSearch
    df = pd.read_csv(FN_RAW)
    
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
  
  df = df.loc[:,fields]
  df = df.dropna()
  df.to_csv('datasets/car_samples.csv.zip', compression='zip', index=False)
  
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
  
  df_sparse = pd.get_dummies(df, columns=categs)
  
  y = df_sparse.pop('price')
  X = df_sparse.values
  csr = sparse.csr_matrix(X)
  sparse.save_npz('datasets/x_csr.npz', matrix=csr)
  np.save('datasets/y', y)
  
  
  

  
  
