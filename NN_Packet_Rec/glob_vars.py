# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:43:25 2019
@author: tina-mac2

Devoloper:  Tina Burns
School: Rutger University
Advisor: Rchard Martin
Python Version: Python3
Code Revision: 

The contains the global variables that will be used across several files
in the NN_packet_recognition project
"""

import numpy as np

train_x =  np.array([])
train_y = np.array([])
train_label = np.array([])
test_x = np.array([])
test_y = np.array([])
test_label = np.array([])
val_x = np.array([])
val_y = np.array([])
val_label = np.array([]) 
act1 = "relu"
act2 = "softmax"
epochs = 10 
batch_size = 128 
testAct = False 
train_model = True
NN_Hist_folder = "NN_Hist"
mod = ""

 