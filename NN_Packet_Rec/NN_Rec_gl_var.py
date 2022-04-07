# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:44:29 2022

@author: TINA
"""
# %% #Imports necessary libraries 
import numpy as np, os, scipy, glob
import sys, time
import pandas as pd

#%%
#This class allows global variable to be access by all functions
class glVar ():
    IQ_pair = np.array([[], []])
    I = None
    Q = None
    fileNames = np.array([])
    mod_type = []
    mod_list = []
    mod_int = []
    mod_UT = ""
    snr = []
    train_x =  np.array([])
    train_y = np.array([])
    train_label = np.array([])
    test_x = np.array([])
    test_y = np.array([])
    test_label = np.array([])
    val_x = np.array([])
    val_y = np.array([])
    val_label = np.array([])

    perm = np.array([])
    data_hex = []
    sep_train_test = True

    myFile = ""
    logfile = []
    folder_base = "C:/Users/TINA/OneDrive - Rutgers University/Rutgers/Research/SDR/Data/"
    
    dtype = "float32"
    folder_test = ""
    folder_train = ""
    folder_results = ""
    myResults = pd.Series([])
    #featureData = pd.Series([])
    testData = pd.Series([])
    test_data = {}
    train_data = {}
    train_X_size = 0
    filePos = 0
    dataArr = {}
    NN_type = ""
    dateCode = ""
    num_points_train = 250
    NNets = ["CAT", "CAT_CONV", "AE",  "BIN","ANOM"]
    header = True
    temp = None
    temp1 = None
    pred = []
    param_value = []
    col_param = ""
    col_mods = "s1_mod"
    cycle = 0
    time_start_OVH = 0
    time_data_collect = 0
    
    NN_train = 1
    NN_Hist_folder = "NN_Hist"
    exc_type = []
    exc_list_train = []
    exc_list_test = []
    iter_f = 0; 
    iter_test_dat = 0