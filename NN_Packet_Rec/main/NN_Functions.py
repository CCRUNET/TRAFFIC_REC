# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Devoloper:  Tina Burns
School: Rutger University
Advisor: Rchard Martin
Python Version: Python3

This code sequeces to through neural netww
"""

import numpy as np, os, scipy, glob
from datetime import datetime
import sys, ntpath, time
import pandas as pd
np.random.seed(1200)  # For reproducibility

#imports appropriate files
import NN_Cat_b14 as NN_CAT
import NN_Cat_Conv_b19 as NN_CAT_CONV
import NN_Binary_b14 as NN_BIN
import NN_AE_STD_b14 as NN_AE
import NN_AE_ANOM_b15 as NN_ANOM
import NN_LSTM_b7 as NN_LSTM
import NN_Simple_b3 as NN_SIMPLE
import NN_matched_filter_b13 as MATCH

class glVar():
    temp = None
    NN_type = None

def run_NN(NNets, m):
    import glob_vars
    for NNet_test in [NNets]:
        if NNet_test == "ANOM": NNet = NN_ANOM.NN()
        elif NNet_test == "CAT": NNet = NN_CAT.NN()
        elif NNet_test == "CONV": NNet = NN_CAT_CONV.NN()
        elif NNet_test == "BIN": NNet = NN_BIN.NN()
        elif NNet_test == "AE": NNet = NN_AE.NN()
        elif NNet_test == "LSTM": NNet = NN_LSTM.NN()
        elif NNet_test == "SIMPLE": NNet = NN_SIMPLE.NN()
        elif NNet_test == "MATCH": NNet = MATCH.NN()
        else: NNet = NN_CAT.NN()
        
        NN_type = NNet.getType()    
        NNet.__init__
        print("Type of neural net is ", NNet_test)

        # #This shapes the array so that it is evenly divisible by 4
        # #which allows it to be correctly processed by the nueral network
        glob_vars.train_x = glob_vars.train_x[:, 0:(glob_vars.train_x.shape[1] - glob_vars.train_x.shape[1]%4)]
        glob_vars.test_x = glob_vars.test_x[:, 0:(glob_vars.test_x.shape[1] - glob_vars.test_x.shape[1]%4)]
        glob_vars.val_x = glob_vars.val_x[:, 0:(glob_vars.val_x.shape[1] - glob_vars.val_x.shape[1]%4)]
        # Reshapes the data so that it is 4 dimensional           
        if NNet_test == "LSTM":
            if len(glob_vars.train_x.shape) <= 2: 
                glob_vars.train_x = glob_vars.train_x.reshape(-1, glob_vars.train_x.shape[1], 1)
            glob_vars.test_x = glob_vars.test_x.reshape(-1, glob_vars.test_x.shape[1], 1)
            glob_vars.val_x = glob_vars.val_x.reshape(-1, glob_vars.val_x.shape[1], 1)
        elif NNet_test != "MATCH":
            if len(glob_vars.train_x.shape) <= 2: glob_vars.train_x = glob_vars.train_x.reshape(-1, 1, glob_vars.train_x.shape[1], 1)
            glob_vars.test_x = glob_vars.test_x.reshape(-1, 1, glob_vars.test_x.shape[1], 1)
            glob_vars.val_x = glob_vars.val_x.reshape(-1, 1, glob_vars.val_x.shape[1], 1)

        print("Starting Neural Network...")

        (loss_test, acc_test, loss_val, acc_val, pred, 
          af1, af2, time_train, time_test) = NNet.runNN(
            X_train = glob_vars.train_x, 
            Y_train = glob_vars.train_y,
            Y_train_label = glob_vars.train_label,
            X_test = glob_vars.test_x,
            Y_test = glob_vars.test_y, 
            Y_test_label = glob_vars.test_label,
            X_val = glob_vars.val_x, 
            Y_val = glob_vars.val_y, 
            Y_val_label = glob_vars.val_label,
            batch_size = 128, epochs = glob_vars.epochs, 
            mod = glob_vars.mod, act1 = glob_vars.act1, act2 = glob_vars.act2,
            testAct = glob_vars.testAct, 
            #plotCurve = False,
            train_model= glob_vars.train_model, 
            folder_NN_hist = glob_vars.NN_Hist_folder
            )

    return  (loss_test, acc_test, loss_val, acc_val, pred, 
          af1, af2, time_train, time_test)


