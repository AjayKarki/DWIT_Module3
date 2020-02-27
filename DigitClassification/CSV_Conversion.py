# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 07:54:34 2020

@author: Ajay
"""

from PIL import Image
import numpy as np
import pandas as pd
import os

root = 'dataset_preprocessed_train\\'

for directory,subdirectories,files in os.walk(root):
    for file in files:
        im = np.asanyarray(Image.open(os.path.join(directory,file))) # euta euta image read gareko
        value =  im.flatten()  # image lai flatten gareko
        value =  np.hstack((directory[-1],value))  # image ko input ho mathi ko chai directory[-1] le chai 0,1,2,3 gardai output class nikaldai append gareko
        df = pd.DataFrame(value).T # transpose gareko
        tf = df.sample(frac=1) # frac meaningless yo line ko satto sidhai df lekhda hunxa tala patti
        with open('train.csv','a') as dataset:
            tf.to_csv(dataset,header=False,index=False)
df = pd.read_csv('train.csv')
'''
Our data is stored sequentially i.e 0 then 1 then 2 then 3
but this causes problem since when we train this data this will result in biasness since the data is not random
so now we use sample(frac=1) to randomly have data
frac=1 means jumble 100% of data 
''' 
df =df.sample(frac=1)
df.to_csv('train.csv',header=False,index=False) # for train csv


root = 'dataset_preprocessed_test\\'

for directory,subdirectories,files in os.walk(root):
    for file in files:
        im = np.asanyarray(Image.open(os.path.join(directory,file))) # euta euta image read gareko
        value =  im.flatten()  # image lai flatten gareko
        value =  np.hstack((directory[-1],value))  # image ko input ho mathi ko chai directory[-1] le chai 0,1,2,3 gardai output class nikaldai append gareko
        df = pd.DataFrame(value).T # transpose gareko
        tf = df.sample(frac=1) # frac meaningless yo line ko satto sidhai df lekhda hunxa tala patti
        with open('test_data.csv','a') as dataset:
            tf.to_csv(dataset,header=False,index=False)
df = pd.read_csv('test_data.csv')
'''
Our data is stored sequentially i.e 0 then 1 then 2 then 3
but this causes problem since when we train this data this will result in biasness since the data is not random
so now we use sample(frac=1) to randomly have data
frac=1 means jumble 100% of data 
''' 
df =df.sample(frac=1)
df.to_csv('test_data.csv',header=False,index=False) # for test csv