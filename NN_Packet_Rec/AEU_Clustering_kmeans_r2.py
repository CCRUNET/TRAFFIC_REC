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

Part of code from 
https://medium.com/@iampatricolee18/autoencoder-k-means-clustering-epl-players-by-their-career-statistics-f38e2ea6e375
""" 

# %%
#Imports necessary libraries 
import os, numpy as np, pandas as pd
np.random.seed(1200)  # For reproducibility
from sklearn.cluster import KMeans
import  numpy as np
from sklearn.datasets import make_classification
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)

#tf.random.seed(123)
np.random.seed(123)
# %%
'''
Used to find the optimal number of clusters. 
Ideally the optimal choice the number of clusters with the smallest euclidean distance between cluster points
and the smallest value of clusters
'''
ssd = [];
def clustering_Kmeans(n_inputs = 100, latent_pred = None):    
    for i in range(2, n_inputs):
        km = KMeans(n_clusters = i).fit(latent_pred)
        ssd.append([int(i), km.inertia_])
    
    return(ssd)

def cluster_items(n_clusters, latent_pred):
    kmeans = KMeans(n_clusters) 
    kmeans.fit(latent_pred) 
    pred_kmeans = kmeans.fit_predict(latent_pred)
    return pred_kmeans

# %% 
def findClusters2(Y_test, pred):
    # Code from: 
    # https://www.codegrepper.com/code-examples/python/found+array+with+dim+4.+estimator+expected+%3C%3D+2.
    # Reshapes data for the proper dimensions for kmeans clustering
    nsamples, a, nx, ny = pred.shape
    pred2 = pred.reshape((nsamples,nx*ny))
    num_clusters = np.unique(Y_test, axis=0).shape[0]
    pred_clus = cluster_items(n_clusters = num_clusters, latent_pred = pred2)
    #print("Predicted Clusters: ", pred_clus)
    return pred_clus
