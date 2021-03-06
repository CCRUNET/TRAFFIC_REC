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
        return "AE-U"
        
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
        weight_file = os.path.join(folder_NN_hist, "SOM_weights.csv").replace(r'\'', '/'); 
        model_file = os.path.join(folder_NN_hist, "SOM_model.p").replace(r'\'', '/');
        data_file = "Data/SOM_sigma.csv"
        num_classes = Y_train.shape[1]        
        
        #Removes previous files to prevent file creation errors
        if not os.path.exists(folder_NN_hist): os.makedirs(folder_NN_hist)
        if os.path.exists(weight_file): os.remove(weight_file)
        if os.path.exists(model_file): os.remove(model_file)
        if os.path.exists(data_file): os.remove(data_file)
        loss_val_train = 0
        acc_val_train = 0
        header = True
        for sig in [0.3, 0.5, 0.6, 0.65, 0.7, 1]:
            for lr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                #for x in [ 1, 4, 5, 10, 20, 30, 40, 50, 100, 200]:
                for size in [20, 50]:    
                    som = MiniSom(num_classes , size, X_train.shape[1], sigma=sig, learning_rate=lr) 
                    som.train(X_train, 100)
                    pred = []
                    for i in X_test: pred.append(som.winner(i)[0])
                    pred = np.asarray(pred)
                    #Gets and outputs predciton of each class
                    acc, pred  = clus_acc.cluster_acc(Y_test, pred)
                    print("Acc: " + str(acc) + " Sigma: " + str(sig) + " LR: " + str(lr) + " Size: " + str(size)) 
                    pd.DataFrame({
                        #"Datatype" : [i], 
                        "Acc":acc, 
                        "Learning Rate ": [lr],
                        "Sigma": [sig],        
                        "Size": [size], 
                        }).to_csv(data_file, mode = 'a', 
                                  header = header)
                    header = False


        print("Train Data Shape: ", X_train.shape)
        print("Accuracy ", float(acc))
        #Validation loss is taken as the final value in the array of validation loss in the training data
        #Returns Test loss, test accuracy, validation loss, validation accuracy 
        return (float(0), acc, float(loss_val_train), 
                float(acc_val_train), pred, act1, act2, 0, 0)  

def reshape_arr(x):
    if np.asarray(x).shape()== 4:
        x = x.reshape(x.shape[0], x.shape[2])
    return(x)

#%%    
def test():
    NNet = NN()
    NNet.__init__
    x, y = generateSequence(10, 10)
    #x = x.reshape(-1, 1, x.shape[1], 1)
    print(x.shape)
    #x = x.reshape(-1, 1, x.shape[1], 1) / np.max(x)
    x = NNet.shuffleData(x)
    y = NNet.shuffleData(y)

    a, b, c = NNet.genTrainTest(x)
    a1, b1, c1 = NNet.genTrainTest(y)
    #NNet.runAutoencoder(a, a, b, b)
    NNet.runNN(a, a1, b, b1, c, c1, train_model = True)

#%% Testing Autoencoder alone
if __name__ == '__main__':
    #pred = test()
    test()

