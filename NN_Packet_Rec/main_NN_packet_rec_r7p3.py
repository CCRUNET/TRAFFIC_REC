#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Devoloper:  Tina Burns
School: Rutger University
Advisor: Rchard Martin
Python Version: Python3

This code sequeces to through neural netww
"""
# %%
#Imports necessary libraries 
import numpy as np, os, scipy, glob
from datetime import datetime
import sys, ntpath, time
import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from keras import backend as K 
np.random.seed(1200)  # For reproducibility

#imports appropriate files
import NN_Cat_b14 as NN_CAT
import NN_Conv_b19 as NN_CAT_CONV
import NN_Binary_b14 as NN_BIN
import NN_AE_STD_b14 as NN_AE
import NN_AE_ANOM_b15 as NN_ANOM
import NN_LSTM_b7 as NN_LSTM
import NN_Simple_b3 as NN_SIMPLE
import NN_matched_filter_b13 as MATCH
import compare_prediction_actual_r6 as conf_mat
import read_binary_r1 as read_binary_file
import arg_parser
import glob_vars as globs 
glVar = globs.glVar
NN_data = globs.NN_data
import data_manip_r1 as data_manip
import write_data_to_file

import warnings                                                                                                                
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)

#Allows use of modules from the Common_Functions Folders
sys.path.append('../../_Neural_Networks')

# %%
def runTest(dateCode, datapoints = 100, samples = 200, writeData = True, 
            num_iter = 2, act1 = "", act2 = "", testAct = False, options = None):
    glVar.cycle = glVar.cycle + 1
    NN_data.datapoints = datapoints
    NN_data.testAct = testAct
    #time_start_OVH = time.time()
    epochs = [10]
    dataArr = ["IQ_pair"]
    for NNet_test in glVar.NNets:
        if NNet_test == "ANOM": NNet = NN_ANOM.NN()
        elif (NNet_test == "CAT" or NNet_test == "FCN" or NNet_test == "FCNN" \
              or NNet_test == "FC"): 
            NNet = NN_CAT.NN()
        elif NNet_test == "CONV" or NNet_test == "CNN": NNet = NN_CAT_CONV.NN()
        elif NNet_test == "BIN": NNet = NN_BIN.NN()
        elif NNet_test == "AE": NNet = NN_AE.NN()
        elif NNet_test == "LSTM": NNet = NN_LSTM.NN()
        elif NNet_test == "SIMPLE": NNet = NN_SIMPLE.NN()
        elif NNet_test == "MATCH": NNet = MATCH.NN()
        #else: NNet = NN_CAT.NN()
        
        glVar.NN_type = NNet.getType()    
        NNet.__init__
    
        if glVar.NN_type == "BIN" or glVar.NN_type == "ANOM" :
            #Gets list unique list of modulation types
            #modulations = ["bpsk", "qpsk", "16qam", "8psk"]
            modulations = set(glVar.testData[glVar.col_mods].values)
        else:
            modulations = ["all"]
        #modulations = ["bpsk"]

        for i in dataArr:
            """"""
            K.clear_session()
            for j in range(1, num_iter+1):
                for m in modulations:
                    glVar.m = m
                    #gets test Data
                    glVar.mod_UT = glVar.m
                
                    if glVar.iter_test_dat == 0: print("Generating Training Data")                    
                    #Only generates training data if it is in a different folder
                    if glVar.NN_train == 1:
                        #print("Collecting traning data")
                        train_samples = samples
                        if NNet_test == "MATCH": 
                            train_samples = 1
                            dp = glVar.num_points_train
                        else: dp = NN_data.datapoints
                        #print("Number of training samples: ", train_samples)
                        glVar.train_data = data_manip.genData(glVar.folder_train, dp, train_samples, 
                            mod = m, NN_Type = NNet_test, arr_exc = glVar.exc_list_train)
                        glVar.train_y = np.asarray(glVar.mod_int)               
                        glVar.train_label = glVar.mod_type
                        train_model = True
                        glVar.train_x = glVar.train_data[i]
                    else: 
                        #print("Not training model")  
                        train_model = False
                        #print(glVar.train_x.shape)
                        if glVar.train_x.shape[0] <= 1:
                            glVar.train_x = np.zeros((samples, NN_data.datapoints))     
                            glVar.train_y = np.zeros((samples, 1))                
                            glVar.train_label = np.zeros((samples, 1)) 
      
                    if glVar.iter_test_dat == 0: print("'\n'Generating Test Data")
                    glVar.iter_test_dat = glVar.iter_test_dat +1
                    glVar.test_data = data_manip.genData(glVar.folder_test, NN_data.datapoints, samples//2,
                        pos = NN_data.datapoints*samples, mod = m, NN_Type = NNet_test, 
                        arr_exc = glVar.exc_list_test)
                    if len(glVar.IQ_pair) < 1: 
                        # Allows for data to be trained if test data is empty and the 
                        # data has not been trained yet                         
                        if glVar.iter_f == 1: glVar.iter_f = 0 
                        continue #Breaks the loop if the IQ array is empty 

                    glVar.test_x, glVar.val_x = np.split(glVar.test_data[i], 2)
                    glVar.test_y, glVar.val_y = np.split(np.asarray(glVar.mod_int), 2)
                    glVar.test_label, glVar.val_label = np.split(glVar.mod_type, 2)
                    print("Test Shape", glVar.test_x.shape)
                    
                    # #This shapes the array so that it is evenly divisible by 4
                    # #which allows it to be correctly processed by the nueral network
                    glVar.train_x = glVar.train_x[:, 0:(glVar.train_x.shape[1] - glVar.train_x.shape[1]%4)]
                    glVar.test_x = glVar.test_x[:, 0:(glVar.test_x.shape[1] - glVar.test_x.shape[1]%4)]
                    glVar.val_x = glVar.val_x[:, 0:(glVar.val_x.shape[1] - glVar.val_x.shape[1]%4)]
                    # Reshapes the data so that it is 4 dimensional
                    # Seperates test and data intoM training, test, and validiation sets
                    #Normalizes x data                   
                    if NNet_test == "LSTM":
                        if len(glVar.train_x.shape) <= 2: glVar.train_x = glVar.train_x.reshape(-1, glVar.train_x.shape[1], 1)
                        glVar.test_x = glVar.test_x.reshape(-1, glVar.test_x.shape[1], 1)
                        glVar.val_x = glVar.val_x.reshape(-1, glVar.val_x.shape[1], 1)
                    elif NNet_test != "MATCH":
                        if len(glVar.train_x.shape) <= 2: glVar.train_x = glVar.train_x.reshape(-1, 1, glVar.train_x.shape[1], 1)
                        glVar.test_x = glVar.test_x.reshape(-1, 1, glVar.test_x.shape[1], 1)
                        glVar.val_x = glVar.val_x.reshape(-1, 1, glVar.val_x.shape[1], 1)

                    
                    # Setups up labels
                    if glVar.NN_type == "MATCH": 
                        labels = pd.get_dummies(glVar.pred).columns.tolist()
                        glVar.pred =  np.asarray(pd.get_dummies(glVar.pred).values)
                    else: labels = list(glVar.mod_list.columns.values)
                    NN_data.labels = labels
                    #Automates testing of activation
                    if not testAct: activations = [""];
                    else: activations = ["elu", "softmax", "selu", "softplus", "softsign", 
                        "relu", "tanh", "sigmoid", "hard_sigmoid", "exponential", 
                    "linear"]; 
                
                    for a1 in  activations:
                        NN_data.a1 = a1
                        for a2 in activations:
                            NN_data.a2 = a2
                            for e in epochs:
                                NN_data.e = e
                                if glVar.NN_type == "ANOM":
                                    glVar.train_y = glVar.train_x
                                    glVar.test_y = glVar.test_x
                                    glVar.val_y = glVar.val_x
                                print("Starting Neural Network...")
                                glVar.time_NN = time.time()
                                (NN_data.loss_test, NN_data.acc_test, 
                                 NN_data.loss_val, NN_data.acc_val, 
                                 glVar.pred, af1, af2, 
                                 glVar.time_train, glVar.time_test
                                  ) = NNet.runNN(
                                    X_train = glVar.train_x, 
                                    Y_train = glVar.train_y,
                                    Y_train_label = glVar.train_label,
                                    X_test = glVar.test_x,
                                    Y_test = glVar.test_y, 
                                    Y_test_label = glVar.test_label,
                                    X_val = glVar.val_x, 
                                    Y_val = glVar.val_y, 
                                    Y_val_label = glVar.val_label,
                                    batch_size = 128, epochs = NN_data.e, 
                                    mod = glVar.m, act1 = NN_data.a1, act2 = NN_data.a2,
                                    testAct = testAct, 
                                    #plotCurve = False,
                                    train_model= train_model, 
                                    folder_NN_hist = glVar.NN_Hist_folder
                                    )
                                glVar.time_NN = np.round(time.time() - glVar.time_NN, 2)                                     
                                atten = 0; snr = 100;
                                name = os.path.basename(glVar.folder_train)
                                if name.find("atten") > -1: atten = name.split("atten")[1]
                                if name.find("snr") > -1:  snr = name.split("snr")[1]
                                glVar.time_OVH = np.round(time.time() - glVar.time_start_OVH - glVar.time_NN, 2)
                             
                                # Writes data to file
                                write_data_to_file.writeData(options)
                                glVar.time_data_collect = 0
                                #Prints results to the screen
                                print("NN: "+ glVar.NN_type + "  ATTEN: " + str(atten) + "  Mod: " + m)
                                print(" Test Folder: " + os.path.basename(glVar.folder_test)) 
                                print("Time of NN: ", glVar.time_NN)
                                print(glVar.col_param, np.round(np.mean(glVar.param_value), 2))
                                print( " Activation 1:  " + af1 + " Activation 2:  " + af2)
                                if options.conf_mat: 
                                    #Sets up prediction array as a categorical arra
                                    cm = conf_mat.main(y_pred = glVar.pred, y_true = glVar.test_y,
                                        #Gets a list of modulation types that aligns with binary array 
                                        labels = labels, 
                                        myFolder = "Data/Results/", 
                                        myFile =glVar.dateCode + "_" + os.path.basename(glVar.folder_test) + "_"
                                        + glVar.NN_type)
                                    glVar.temp = cm

# %% Main unction that run the Neural Network  Code
def main(options=None):
    glVar.time_data_collect = time.time()
    if options is None:
        options = arg_parser.argument_parser().parse_args()
    globs.glVar.options = options
    print("Testing")
    #Sets the folder locations to be tested 
    glVar.dateCode = str(datetime.now()).replace('.', '').replace(' ', '_').replace(':', '')
    if options.folder_test == "": glVar.folder_test = options.folder_train
    else: glVar.folder_test = options.folder_test
    if options.folder_train[-1] == "/": glVar.folder_train = options.folder_train[0:-1]
    else: glVar.folder_train = options.folder_train
    glVar.NNets = options.NNets
    glVar.col_param= options.col_param
    glVar.col_mods= options.col_mods
    glVar.num_points_train = options.num_points_train
    glVar.dtype = options.data_type
    glVar.NN_train = options.NN_train
    
    #var = os.getcwd().replace("//","/")
    options.NN_Hist_folder = os.path.join(os.getcwd(), "NN_Hist",  options.NN_Hist_folder)
    #options.NN_Hist_folder = options.NN_Hist_folder.replace("//","/")
    glVar.NN_Hist_folder = options.NN_Hist_folder.replace("//","/").replace("./", "")
    
    if glVar.folder_test =="" or glVar.folder_test == glVar.folder_train: 
        folders_test = data_manip.getFolderList(glVar.folder_train)
        glVar.sep_train_test = False
    else: 
        folders_test = data_manip.getFolderList(glVar.folder_test)
        print("Original Test Folders: ", folders_test)
        #Removes training folder from test set
        #folders_test = [x for x in folders_test if (glVar.folder_train[0:-1] not in x and "train" not in x)]
        folders_test = [x for x in folders_test if "train" not in x]
        if glVar.folder_train in folders_test: folders_test.remove(glVar.folder_train)

    #Creates list of logfiles.  
    #If the entry is a directory, all the files in the directory are appended to the list
    li = []
    if len(options.logfile[0]) > 0: 
        for i in options.logfile:
            if os.path.isdir(i): glVar.logfile.extend( [s for s in glob.glob(i)])               
            else: glVar.logfile.append(i)
            #glVar.logfile.append(i)
        glVar.logfile = list(set(glVar.logfile)) 
        for j in glVar.logfile: li.append(pd.read_csv(j))
        #Concatentate pd dataframe. Removes entries with duplicate filenames
        glVar.testData = pd.concat(li, axis=0, ignore_index=True).drop_duplicates(subset=["filename"])
        glVar.testData["s1_mod"] = glVar.testData["s1_mod"].str.lower()
    else: sys.exit("Logfile not available. Please inNNude an appropriate logfile location")

    #Gets list of files to exclude from the training and test data
    options.range_train = list(np.asarray(options.range_train).astype(float))
    options.range_test = list(np.asarray(options.range_test).astype(float))
    if len(options.seq_train) < len(options.seq_test): options.seq_train = options.seq_test
    if len(options.seq_test) < len(options.seq_train): options.seq_test = options.seq_train
    print("Train: ", options.seq_train)
    glVar.exc_list_test = data_manip.getExclusionList(options, range_param = options.range_param, range_arr = options.range_test, 
                                                      arr_in = options.seq_test)
    glVar.exc_list_train = data_manip.getExclusionList(options, range_param = options.range_param,  range_arr = options.range_train,
                                                       arr_in = options.seq_train)
    
    glVar.time_data_collect = np.round(time.time() - glVar.time_data_collect, 2)
    
    #This for loop run the main function for all folders in the "folders" array
    glVar.iter_f = 0
    for f in folders_test:
        glVar.iter_f = glVar.iter_f + 1
        glVar.time_start_OVH = time.time()
        glVar.folder_test = f 
        if not glVar.sep_train_test: glVar.folder_train = f
        if glVar.sep_train_test and glVar.iter_f > 1: glVar.NN_train = 0
        #print("\nTest Data: ", glVar.folder_test) 
        """"""                
        runTest(glVar.dateCode, datapoints = options.num_points, 
        samples = options.samples, num_iter = options.iter, 
        testAct = data_manip.str2bool(options.test_act), options = options)

    if options.rclone_loc == "1": 
        print("Copying data to " + options.rclone_loc)
        os.system("rlcone copy Data/Results/ " + options.rclone_loc)
    print("Done")
    return options
#%% Runs main port of program
if __name__ == '__main__':
    main()
   
    
