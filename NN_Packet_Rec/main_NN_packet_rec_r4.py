#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Devoloper:  Tina Burns
School: Rutger University
Advisor: Rchard Martin
Python  Version: Python3

This code sequeces to through neural netww
"""
#Imports necessary libraries 
import numpy as np, os, scipy, glob
from datetime import datetime
import sys, ntpath, time
import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from keras import backend as K 
np.random.seed(1200)  # For reproducibility

#Allows use of modules from the Common_Functions Folders
sys.path.append('../../_Neural_Networks')

#imports appropriate files
import compare_prediction_actual_r4 as conf_mat
import read_binary_r1 as read_binary_file
import arg_parser as arg_parser
import NN_main as NN_main
import warnings                                                                                                                
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
import glob_vars


#%%
#This class allows global variable to be access by all functions
class glVar ():
    temp = None
    temp1 = None
    
    IQ_pair = np.array([[], []])
    I = None
    Q = None
    fileNames = np.array([])
    mod_type = []
    mod_list = []
    mod_int = []
    mod_UT = ""
    snr = []

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

    pred = []
    param_value = []
    col_param = ""
    col_mods = "s1_mod"
    cycle = 0
    time_start_OVH = 0
    time_data_collect = 0
    
    NN_train = 1
    exc_type = []
    exc_list_train = []
    exc_list_test = []
    iter_f = 0; 
    iter_test_dat = 0

#%%
#This portion of the code was written by Ryan Davis and modified by Tina Burns. 
def getFileData(data_path, num_points_per_sample, num_samples_per_file, 
                posStart = 0, testing = True, arr_exc = []):
    
    x = []; y = []; z = []; count = 0;  glVar.param_value = []
    for fname in os.listdir(data_path):
        #Ignores .DS_Store file ang files that don't meet the parameter specifications
        if (fname != ".DS_Store" and fname not in arr_exc and fname.find(".txt")<0 and fname.find(".csv")<0): 
            f = read_binary_file.read_binary_iq(fname = data_path+ '/'+fname, samples = num_samples_per_file, 
                    num_points = num_points_per_sample, pos_sample = posStart, d_type = glVar.dtype)
            #f = (np.asarray(f)/np.max(abs(np.asarray(f))))
            f = f[:][0:(f.shape[0] - f.shape[0]%num_points_per_sample)] #Ensures even number of points
            #print(posStart)
            if count == 0: x = f.reshape(-1, num_points_per_sample)[0:num_samples_per_file]
            else: x = np.vstack((x, f.reshape(-1, num_points_per_sample)[0:num_samples_per_file]))
            y = y + [ntpath.basename(fname)]*num_samples_per_file
        
            #The portion of the fuction get the modulation information
            mod = glVar.testData[glVar.testData["filename"] == fname][glVar.col_mods].values.item()
            z = z + [mod]*num_samples_per_file            
            if testing: 
                glVar.param_value.append(float(glVar.testData[glVar.testData["filename"] == 
                    fname][glVar.col_param].values.item()))               
                count = count +1
    return x, np.asarray(y), np.asarray(z)
#%%
# Shuffles Data
def shuffleData(x):
    np.random.seed(1200)
    myPermutation = np.random.permutation(x.shape[0])
    x = x[myPermutation] 
 
#%%
def getExclusionList(range_param  = "s1_sinr", range_arr = [-1000, 1000], exc_param = "s1_mod", exc_arr = [""]):
    arr = []
    #Makes all values in exculsion column lower case
    glVar.testData[exc_param] = glVar.testData[exc_param].str.lower() 
    for exc in exc_arr:
        arr = arr + list(glVar.testData["filename"][glVar.testData[exc_param] == exc.lower()])
    #print([glVar.testData[range_param] >= range_arr[0]])
    arr = arr + list(glVar.testData["filename"][glVar.testData[range_param] < range_arr[0]])
    arr = arr + list(glVar.testData["filename"][glVar.testData[range_param] > range_arr[1]])
    #print("making exclusion list")
    return arr
#%%
#Main function of the program tha executes the main operations
def genData(myFile, numDatapoints = 100, numSamples = 200, pos = 0, 
            mod = "", NN_Type = "CAT", testing = True, arr_exc = []):    
    my_dict = {}
    #Inputs information into global variables for later usage
    #The number of bytes must be divisible by 8 in order to properly work with the NN
    glVar.IQ_pair, glVar.fileNames, glVar.mod_type = getFileData(myFile, numDatapoints, numSamples, 
            posStart = pos, testing = testing, arr_exc = arr_exc)      
    
    #glVar.IQ_pair[:, 1::4] = 0 
    glVar.I = glVar.IQ_pair[:, 0::2]
    glVar.Q = glVar.IQ_pair[:, 1::2]
    #glVar.IQ_pair = glVar.Q/np.max(glVar.Q)
    glVar.IQ_pair = glVar.Q
    glVar.IQ_pair = glVar.IQ_pair/np.max(glVar.IQ_pair) #Normalizes data
    
    if len(glVar.IQ_pair) >= 1:
        #Shuffles the all data arrays
        glVar.IQ_pair = shuffleData(glVar.IQ_pair)
        glVar.fileNames = shuffleData(glVar.fileNames)
        glVar.mod_type = shuffleData(glVar.mod_type)
        
        #Get unique values of modulation schemes
        glVar.mod_list = pd.get_dummies(glVar.mod_type)
        #Puts mod_int list in the form of integar values
        if glVar.NN_type  == "ANOM":
            glVar.mod_int = pd.factorize(glVar.mod_type)[0]
        #Puts mod_int list in the form of binary arrays (for categorical classification)
        else:
            glVar.mod_int = glVar.mod_list.values
        #Stores information in dictionary
        my_dict = {
           #"IQ_stack": glVar.IQ_stack, 
            "IQ_pair": glVar.IQ_pair,
            "mod_type": glVar.mod_type,
            "mod_int": glVar.mod_int,
            glVar.col_param: glVar.param_value
            }
    return my_dict
    
# %%
def runTest(dateCode, datapoints = 100, samples = 200, writeData = True, 
            num_iter = 2, act1 = "", act2 = "", testAct = False, options = None):
    glVar.cycle = glVar.cycle + 1
    #time_start_OVH = time.time()
    epochs = [10]
    dataArr = ["IQ_pair"]
    for NNet_test in glVar.NNets:
        
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
                    #gets test Data
                    glVar.mod_UT = m
                
                    if glVar.iter_test_dat == 0: print("Generating Training Data")                    
                    #Only generates training data if it is in a different folder
                    if glVar.NN_train == 1:
                        #print("Collecting traning data")
                        train_samples = samples
                        if NNet_test == "MATCH": 
                            train_samples = 1
                            dp = glVar.num_points_train
                        else: dp = datapoints
                        #print("Number of training samples: ", train_samples)
                        glVar.train_data = genData(glVar.folder_train, dp, train_samples, 
                            mod = m, NN_Type = NNet_test, arr_exc = glVar.exc_list_train)
                        glob_vars.train_y = np.asarray(glVar.mod_int)               
                        glob_vars.train_label = glVar.mod_type
                        train_model = True
                        glob_vars.train_x = glVar.train_data[i]
                    else: 
                        #print("Not training model")  
                        train_model = False
                        #print(glVar.train_x.shape)
                        if glVar.train_x.shape[0] <= 1:
                            glob_vars.train_x = np.zeros((samples, datapoints))     
                            glob_vars.train_y = np.zeros((samples, 1))                
                            glob_vars.train_label = np.zeros((samples, 1)) 

                    
                    if glVar.iter_test_dat == 0: print("'\n'Generating Test Data")
                    glVar.iter_test_dat = glVar.iter_test_dat +1
                    glVar.test_data = genData(glVar.folder_test, datapoints, samples//2,
                        pos = datapoints*samples, mod = m, NN_Type = NNet_test, 
                        arr_exc = glVar.exc_list_test)
                    if len(glVar.IQ_pair) < 1: 
                        # Allows for data to be trained if test data is empty and the 
                        # data has not been trained yet                         
                        if glVar.iter_f == 1: glVar.iter_f = 0 
                        continue #Breaks the loop if the IQ array is empty 
                        
                    # Seperates test and data intoM training, test, and validiation sets 
                    glob_vars.test_x, glob_vars.val_x = np.split(glVar.test_data[i], 2)
                    glob_vars.test_y, glob_vars.val_y = np.split(np.asarray(glVar.mod_int), 2)
                    glob_vars.test_label, glob_vars.val_label = np.split(glVar.mod_type, 2)
                    print("Test Shape", glob_vars.test_x.shape)
                    

                    if not testAct: activations = [""];
                    else: activations = ["elu", "softmax", "selu", "softplus", "softsign", 
                        "relu", "tanh", "sigmoid", "hard_sigmoid", "exponential", 
                        "linear"]; 

                    for a1 in  activations:
                        for a2 in activations:                     
                            for e in epochs:                        
                                if glVar.NN_type == "ANOM":
                                    glVar.train_y = glVar.train_x
                                    glVar.test_y = glVar.test_x
                                    glVar.val_y = glVar.val_x
                                print("Starting Neural Network...")
                                time_NN = time.time()
                                (loss_test, acc_test, loss_val, acc_val, glVar.pred, 
                                  af1, af2, time_train, time_test) = NN_main.run_NN(NNets= NNet_test, m = m)
                                time_NN = np.round(time.time() - time_NN, 2)                                     
                                
                                atten = 0; snr = 100;
                                name = os.path.basename(glVar.folder_train)
                                if name.find("atten") > -1: atten = name.split("atten")[1]
                                if name.find("snr") > -1:  snr = name.split("snr")[1]
                                time_OVH = np.round(time.time() -glVar.time_start_OVH - time_NN, 2)
                             
                                if writeData:
                                    pd.DataFrame({
                                            #"Datatype" : [i], 
                                            glVar.col_param: [np.round(np.mean(glVar.param_value), 2)],
                                            "Acc-Test": [np.round(acc_test, 3)], 
                                            "Loss-Test ": [np.round(loss_test, 2)],
                                            "Acc-Val": [np.round(acc_val, 3)],        
                                            "Loss-Val ": [np.round(loss_val, 2)], 
                                            "Epochs": [e],
                                            "Train Samples": [glob_vars.train_x.shape[0]],
                                            "Test Samples": [glob_vars.test_x.shape[0]],
                                            "Validation Samples": [glob_vars.val_x.shape[0]],
                                            "Datapoints": [datapoints],              
                                            "dir_train": [os.path.basename(os.path.dirname(glVar.folder_train))],
                                            "folder_train": [os.path.basename(glVar.folder_train)],   
                                            "dir_test": [os.path.basename(os.path.dirname(glVar.folder_test))],
                                            "folder_test": [os.path.basename(glVar.folder_test)],
                                            "NN_Type": [glVar.NN_type],
                                            "Mod-Type": [m], 
                                            "time_data_collect": [glVar.time_data_collect], 
                                            "time_data_manip": [time_OVH], 
                                            "time_NN": [time_NN], 
                                            "time_train": [time_train], 
                                            "time_test": [time_test],
                                            "Activation 1: ": [a1],
                                            "Activation 2": [a2],
                                            "Param ": [options.range_param],
                                            "Param Train Min": [options.range_train[0]],
                                            "Param Train Max": [options.range_train[1]],
                                            "Param Test Min": [options.range_test[0]],
                                            "Param Test Max": [options.range_test[1]],
                                            }).to_csv("Data/Results/" + glVar.dateCode + "_Test.csv", mode = 'a', 
                                                      header = glVar.header)
                                    glVar.header = False
                                    glVar.time_data_collect = 0
                                    
                                print("NN: "+ glVar.NN_type + "  ATTEN: " + str(atten) + "  Mod: " + m)
                                
                                print(" Test Folder: " + os.path.basename(glVar.folder_test)) 
                                print("Time of NN: ", time_NN)
                                print(glVar.col_param, np.round(np.mean(glVar.param_value), 2))
                                print( " Activation 1:  " + af1 + " Activation 2:  " + af2)
                                if options.conf_mat: 
                                    #Sets up prediction array as a categorical array
                                    if glVar.NN_type == "MATCH": 
                                        labels = pd.get_dummies(glVar.pred).columns.tolist()
                                        glVar.pred =  np.asarray(pd.get_dummies(glVar.pred).values)
                                    else: labels = list(glVar.mod_list.columns.values)
                                    conf_mat.main(y_pred = glVar.pred, y_true = glVar.test_y,
                                        #Gets a list of modulation types that aligns with binary array 
                                        labels_text = labels, 
                                        myFolder = "Data/Results/", 
                                        myFile =glVar.dateCode + "_" + os.path.basename(glVar.folder_test) + "_"
                                        + glVar.NN_type)
    return 0

 
#%% Gets list of folders to be tested 
def getFolderList(loc_data):
    #a = [*set([loc_data + "/" + p for p in os.listdir(loc_data)])]
    a = []; 
    #os.join create adds a '\' when joining info.  
    for root, dirs, files in os.walk(loc_data):
        for d in dirs:
            if(d.lower().find("data") <0 and d.lower().find("plot") <0 and d.find("cleansig")<0): 
                a.append( str(root + '/' + d).replace("//", "/"))
        for f in files:
            if (f.lower().find("log") > -1 or f.lower().find(".csv") > -1): 
                glVar.logfile.append(root + "/" + f)
                
    if len(a) == 0: a.append(loc_data)
    if not os.path.exists("Data/Results"): os.makedirs("Data/Results")
    return a
# %%
def main(options=None):
    glVar.time_data_collect = time.time()
    if options is None:
        options = arg_parser.argument_parser().parse_args()       
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
    glob_vars.NN_Hist_folder = os.path.join(os.getcwd(), "NN_Hist",  options.NN_Hist_folder)
    #options.NN_Hist_folder = options.NN_Hist_folder.replace("//","/")
    glob_vars.NN_Hist_folder = glob_vars.NN_Hist_folder.replace("//","/").replace("./", "")
    
    if glVar.folder_test =="" or glVar.folder_test == glVar.folder_train: 
        folders_test = getFolderList(glVar.folder_train)
        glVar.sep_train_test = False
    else: 
        folders_test = getFolderList(glVar.folder_test)
        print("Original Test Folders: ", folders_test)
        #Removes training folder from test set
        #folders_test = [x for x in folders_test if (glVar.folder_train[0:-1] not in x and "train" not in x)]
        folders_test = [x for x in folders_test if "train" not in x]
        if glVar.folder_train in folders_test: folders_test.remove(glVar.folder_train)

    #Creates list of logfiles.  
    #If the entry is a directory, all the files in the directory are appended to teh list
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
    else: sys.exit("Logfile not available. Please include an appropriate logfile location")

    #Gets list of files to exclude from the training and test data
    #options.range_train = list(np.asarray(options.range_train).astype(float))
    #options.range_test = list(np.asarray(options.range_test).astype(float))
    # glVar.exc_list_train = getExclusionList(range_param = options.range_param, range_arr = options.range_train, 
    #                             exc_param = options.exc_param, exc_arr = options.exc_train)
    # glVar.exc_list_test = getExclusionList(range_param = options.range_param, range_arr = options.range_test, 
    #                             exc_param = options.exc_param, exc_arr = options.exc_test)
    
    glVar.exc_list_test = []
    glVar.exc_list_train = []
    
    #Fuction to return a boolean value
    #Code from:
    #https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    def str2bool(v):
        if isinstance(v, bool):return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
        else: return False
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
        testAct = str2bool(options.test_act), options = options)
    print("Done")

#%% Runs main port of program
if __name__ == '__main__':
    main()
   
    