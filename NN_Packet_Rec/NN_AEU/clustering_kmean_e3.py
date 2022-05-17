# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:22:05 2022

@author: TINA
"""


# https://medium.com/@iampatricolee18/autoencoder-k-means-clustering-epl-players-by-their-career-statistics-f38e2ea6e375

from sklearn.cluster import KMeans
import tensorflow as tf
import  numpy as np

# affinity propagation clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)

#tf.random.seed(123)
np.random.seed(123)

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