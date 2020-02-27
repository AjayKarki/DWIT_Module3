# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 07:36:32 2020

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

