# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:50:45 2022

@author: TINA
"""


# %%
#Imports necessary libraries 
import numpy as np, os, scipy, glob
from datetime import datetime
import sys, ntpath, time
import pandas as pd
from keras import backend as K 
np.random.seed(1200)  # For reproducibility
import read_binary_r1 as read_binary_file
import glob_vars as globs 
glVar = globs.glVar
NN_data = globs.NN_data
options = globs.glVar.options


def writeData(options):
    if options.file_results == "": 
       options.file_results = "Data/Results/" + glVar.dateCode + "_Test.csv"
    
    if os.path.exists(options.file_results): glVar.header = False;
    else:     glVar.header = True
        
    pd.DataFrame({
            #"Datatype" : [i], 
            glVar.col_param: [np.round(np.mean(glVar.param_value), 2)],
            "Acc-Test": [np.round(globs.NN_data.acc_test, 3)], 
            "Loss-Test ": [np.round(globs.NN_data.loss_test, 2)],
            "Acc-Val": [np.round(globs.NN_data.acc_val, 3)],        
            "Loss-Val ": [np.round(globs.NN_data.loss_val, 2)], 
            "Epochs": [NN_data.e],
            "Train Samples": [glVar.train_x.shape[0]],
            "Test Samples": [glVar.test_x.shape[0]],
            "Validation Samples": [glVar.val_x.shape[0]],
            "Datapoints": [NN_data.datapoints],              
            "dir_train": [os.path.basename(os.path.dirname(glVar.folder_train))],
            "folder_train": [os.path.basename(glVar.folder_train)],   
            "dir_test": [os.path.basename(os.path.dirname(glVar.folder_test))],
            "folder_test": [os.path.basename(glVar.folder_test)],
            "NN_Type": [glVar.NN_type],
            "Mod-Type": [glVar.m], 
            "time_data_collect": [glVar.time_data_collect], 
            "time_data_manip": [glVar.time_OVH], 
            "time_NN": [glVar.time_NN], 
            "time_train": [glVar.time_train], 
            "time_test": [glVar.time_test],
            "Activation 1: ": [NN_data.a1],
            "Activation 2": [NN_data.a2],
            "Param ": [options.range_param],
            "Param Train Min": [options.range_train[0]],
            "Param Train Max": [options.range_train[1]],
            "Param Test Min": [options.range_test[0]],
            "Param Test Max": [options.range_test[1]],
            "Classes": [NN_data.labels],
            }).to_csv(options.file_results, mode = 'a', 
                      header = glVar.header)
    return 0