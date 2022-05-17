# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:43:25 2019
@author: tina-mac2

Devoloper:  Tina Burns
School: Rutger University
Advisor: Rchard Martin
Python Version: Python3
Code Revision:

This code for an Unsperivsed Autencoder clustering algorithms
"""
# %%
#Imports necessary libraries 
import os, numpy as np, pandas as pd
np.random.seed(1200)  # For reproducibility

# Import files
import clustering_kmean_e3 as clus_kmean_3 

#%%
# Code from: 
# https://github.com/ardamavi/Unsupervised-Classification-with-Autoencoder/blob/master/Examples/Dog-Cat/Dog-Cat%20Classification%20With%20Autoencoder.ipynb
def findClusters(encoder, X_train, Y_train, num_class = 4):
    print("Finding cluster in data... ")
    encode = encoder.predict(X_train)
    print("Getting samples...")
    class_dict = np.zeros((num_class, num_class))
    for i, sample in enumerate(Y_train):
        class_dict[np.argmax(encode[i], axis=0)][np.argmax(sample)] += 1
        
    # print(class_dict)
        
    # neuron_class = np.zeros((num_class))
    # for i in range(num_class):
    #     neuron_class[i] = np.argmax(class_dict[i], axis=0)
    
    # print(neuron_class)   
    
    return 0
# %% 
def findClusters2(Y_test, pred):
    # Code from: 
    # https://www.codegrepper.com/code-examples/python/found+array+with+dim+4.+estimator+expected+%3C%3D+2.
    # Reshapes data for the proper dimensions for kmeans clustering
    nsamples, a, nx, ny = pred.shape
    pred2 = pred.reshape((nsamples,nx*ny))
    num_clusters = np.unique(Y_test, axis=0).shape[0]
    pred_clus = clus_kmean_3.cluster_items(n_clusters = num_clusters, latent_pred = pred2)
    #print("Predicted Clusters: ", pred_clus)
    return pred_clus
