# -*- coding: utf-8 -*-
"""
Created on Mon May 23 22:36:03 2022

@author: TINA
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import files
import AEU_Clustering_kmeans_r2 as ae_clus
import AEU_clus_acc_r1 as clus_acc
#https://www.section.io/engineering-education/dbscan-clustering-in-python/#:~:text=DBSCAN%20is%20a%20popular%20density,number%20of%20clusters%20required%20prior.

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
#%%
from sklearn.neighbors import NearestNeighbors # importing the library
def calc_nearsest_neighbor(x):
    neighb = NearestNeighbors(n_neighbors=2) # creating an object of the NearestNeighbors class
    nbrs=neighb.fit(x) # fitting the data to the object
    distances,indices=nbrs.kneighbors(x) # finding the nearest neighbours
    # Sort and plot the distances results
    distances = np.sort(distances, axis = 0) # sorting the distances
    distances = distances[:, 1] # taking the second column of the sorted distances
    
    plt.rcParams['figure.figsize'] = (5,3) # setting the figure size
    plt.plot(distances) # plotting the distances
    plt.show() # showing the plot
    
#%%    
from sklearn.cluster import DBSCAN
def do_DBSCAN(x):
    # cluster the data into five clusters
    dbscan = DBSCAN(eps = 8, min_samples = 4).fit(x) # fitting the model
    labels = dbscan.labels_ # getting the labels
    return(labels)


#%% https://towardsdatascience.com/explaining-dbscan-clustering-18eaf5c83b31
from sklearn.metrics import silhouette_score
from sklearn.metrics import v_measure_score
def runTest(X):
    pca_eps_values = np.arange(0.2,1.5,0.1) 
    pca_min_samples = np.arange(2,5) 
    pca_dbscan_params = list(product(pca_eps_values, pca_min_samples))
    pca_no_of_clusters = []
    pca_sil_score = []
    pca_epsvalues = []
    pca_min_samp = []
    for p in pca_dbscan_params:
        pca_dbscan_cluster = DBSCAN(eps=p[0], min_samples=p[1]).fit(X)
        pca_epsvalues.append(p[0])         
        pca_min_samp.append(p[1])
        pca_no_of_clusters.append(len(np.unique(pca_dbscan_cluster.labels_)))
        
        pca_sil_score.append(silhouette_score(X, pca_dbscan_cluster.labels_))
        
        pca_eps_min = list(zip(pca_no_of_clusters, pca_sil_score, pca_epsvalues, pca_min_samp))
        
        pca_eps_min_df = pd.DataFrame(pca_eps_min, columns=['no_of_clusters', 'silhouette_score', 'epsilon_values', 'minimum_points'])
        print(pca_eps_min_df)
                    
#%%    
def main():
    x, y = generateSequence(100, 100)
    pred = do_DBSCAN(x)
    glVar.temp_1 = y
    glVar.temp = pred
    #pred = clus_acc.cluster_acc(y_true, y_pred)
   
#%% Testing Autoencoder alone
if __name__ == '__main__':
    #pred = test()
    main()

