#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:43:25 2019
@author: tina-mac2

Devoloper:  Tina Burns
School: Rutger University
Advisor: Rchard Martin
Python Version: Python3
Code Revision: 

This code for an Unsperivsed Autencoder based on Keras 
"""
# %%
#Imports necessary libraries 
import os, numpy as np, pandas as pd, time
import pickle
from minisom import MiniSom 
from pathlib import Path

#from keras.utils import multi_gpu_model
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TENSORFLOW_FLAGS"] = "device=gpu%d" % (0)
np.random.seed(1200)  # For reproducibility

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import files
import AEU_clus_acc_r1 as clus_acc
# %%
class glVar():
    temp = None
    temp_1 = None
    ae_pred = None
    ae_lab = None
# %% Generates data
#Creates two matrix arrays: One in ascending order and
#One array in descending order
#Return the concantenation of those two arrays and another array for labels
def generateSequence(m, n):
    # return [randint(0, 4) for _ in range(length)]
    arr = np.empty([0, n])
    for i in range(1, m + 1):
        arr = np.append(arr, [np.arange(i, n + i)], axis=0)
    arr = np.concatenate((arr, np.flip(arr)))
    lab = np.concatenate((np.full((m, 1), 0), np.full((m, 1), 1)))
    lab = np.array(pd.get_dummies(lab.reshape(lab.shape[0])))
    #lab = pd.get_dummies(lab)
    # print(arr1.shape)
    return (arr, lab)

# %% Neural Network Class
# This neural network is fully connected Sequential network using the keras library
# Parts of the code have been modified from:
class NN():
    def __init__(self):
        """"""        
    def getType(self):
        return "SOM"
        
    # This function splits the array into three seperate arrays
    def genTrainTest(self, arr):
        sep = int(arr.shape[0] / 5)
        return np.split(arr, [sep * 3, sep * 4], axis=0)
    # Shuffles Data
    def shuffleData(self, x):
        np.random.seed(1200)
        myPermutation = np.random.permutation(x.shape[0])
        x = x[myPermutation]
        return x

    def runNN(self, X_train, Y_train, X_test, Y_test, X_val, Y_val, 
              Y_train_label= None, Y_test_label = None, 
              Y_val_label = None, act1 = "relu",  act2 = "softmax", 
              epochs = 10, batch_size = 128, testAct = False, mod = '',
              train_model = True, folder_NN_hist = "NN_Hist"):   
        #Sets files to store training information
        weight_file = os.path.join(folder_NN_hist, "SOM_weights").replace(r'\'', '/').replace('//', '//'); 
        model_file = os.path.join(folder_NN_hist, "SOM_model.p").replace(r'\'', '/').replace('//', '//'); 

        num_classes = Y_train.shape[1]        
        time_train_start = time.time()
        #Training model
        if train_model:
            #Removes previous files to prevent file creation errors
            if not os.path.exists(folder_NN_hist): os.makedirs(folder_NN_hist)
            if os.path.exists(weight_file): os.remove(weight_file)
            if os.path.exists(model_file): os.remove(model_file)

            #som = MiniSom(num_classes , X_train.shape[0], X_train.shape[1], sigma=5, learning_rate=0.6) 
            som = MiniSom (num_classes, 20, X_train.shape[1], sigma=0.5, learning_rate=0.6) 
            #som = MiniSom (num_classes, 20, X_train.shape[1]) 
            som.train(X_train, 10000)
            # saving the som in the file som.p
            pickle.dump( som, open( model_file, "wb" ) )
            np.save(weight_file, som.get_weights())
            loss_val_train = 0
            acc_val_train = 0

        else:
            # laoding file
            #weights = np.load(weight_file + ".npy")
            som = pickle.load(open( model_file, "rb" ) )
            loss_val_train = 0
            acc_val_train = 0 
                
        time_train = np.round(time.time() - time_train_start, 2)
        time_test_start = time.time()

        pred = []
        for i in X_test: pred.append(som.winner(i)[0])
        pred = np.asarray(pred)

        #Gets and outputs predciton of each class
        acc, pred  = clus_acc.cluster_acc(Y_test, pred)

        time_test = np.round(time.time() - time_test_start, 2)

        #print("Prediction: ", pred)
        print("Accuracy ", float(acc))
 

        #Validation loss is taken as the final value in the array of validation loss in the training data
        #Returns Test loss, test accuracy, validation loss, validation accuracy 
        return (float(0), acc, float(loss_val_train), 
                float(acc_val_train), pred, act1, act2, time_train, time_test)  

def reshape_arr(x):
    if np.asarray(x).shape()== 4:
        x = x.reshape(x.shape[0], x.shape[2])
    return(x)

#%%    
def test(train_data = True):
    NNet = NN()
    NNet.__init__
    x, y = generateSequence(1000, 1000)
    #x = x.reshape(-1, 1, x.shape[1], 1)
    print(x.shape)
    #x = x.reshape(-1, 1, x.shape[1], 1) / np.max(x)
    x = NNet.shuffleData(x)
    y = NNet.shuffleData(y)

    a, b, c = NNet.genTrainTest(x)
    a1, b1, c1 = NNet.genTrainTest(y)
    #NNet.runAutoencoder(a, a, b, b)
    NNet.runNN(a, a1, b, b1, c, c1, train_model = train_data)

#%% Testing Autoencoder alone
if __name__ == '__main__':
    #pred = test()
    test()

