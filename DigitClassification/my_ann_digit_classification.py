# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 08:31:10 2020

@author: Ajay
"""

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_df = pd.read_csv('train.csv')
print(train_df.shape)

test_df = pd.read_csv('test_data.csv')
print(test_df.shape)
X_train = train_df.iloc[:,1:]
print(X_train.shape)
print(X_train.head)
Y_train = train_df.iloc[:,0]
print(Y_train.shape)
print(Y_train.head(4))

from keras.layers import Dense,Dropout,Input
from keras.models import Sequential

ann = Sequential()
ann.add(Dense(units=300,input_shape=(1024,)))
ann.add(Dense(300))
ann.add(Dense(1))
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

ann.fit(X_train,Y_train,batch_size=32,epochs=10)