# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:04:53 2022

@author: TINA
"""

# %%
#Imports necessary libraries 
import numpy as np, os, scipy, glob
import sys, ntpath, time
import pandas as pd
#mport matplotlib.pyplot as plt
#from keras import backend as K 
np.random.seed(1200)  # For reproducibility
import read_binary_r1 as read_binary_file
import glob_vars as globs 
glVar = globs.glVar


# %% This portion of the code was written by Ryan Davis and modified by Tina Burns. 
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
# %% Main function of the program tha executes the main operations
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
    #glVar.IQ_pair = glVar.I
    glVar.IQ_pair = glVar.IQ_pair/np.max(glVar.Q) #Normalizes data
    
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
 
# %% Gets list of folders to be tested 
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


# %% Shuffles Data
def shuffleData(x):
    np.random.seed(1200)
    myPermutation = np.random.permutation(x.shape[0])
    x = x[myPermutation]
    return x    
#%%
def getExclusionList(range_param  = "s1_sinr", range_arr = [-1000, 1000], exc_param = "s1_mod", exc_arr = [""], 
                     inc_arr = []):
    arr = []
    #Makes all values in exculsion column lower case
    glVar.testData[exc_param] = glVar.testData[exc_param].str.lower() 
    if len(inc_arr) > 0: 
        print("Using inclusion list to create an exclusion list")
        exc_arr = set(glVar.testData[exc_param]).difference(inc_arr)
    print("Creating exclusion list")   
    for exc in exc_arr:
        arr = arr + list(glVar.testData["filename"][glVar.testData[exc_param] == exc.lower()])
    #print([glVar.testData[range_param] >= range_arr[0]])
    arr = arr + list(glVar.testData["filename"][glVar.testData[range_param] < range_arr[0]])
    arr = arr + list(glVar.testData["filename"][glVar.testData[range_param] > range_arr[1]])
    #print("making exclusion list")
    return arr

