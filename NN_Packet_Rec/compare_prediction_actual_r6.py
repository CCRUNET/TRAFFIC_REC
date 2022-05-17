#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 20:50:22 2020

@author: Tina Burns
School: Rutger University
Advisor: Rchard Martin
Python Version: Python3

This code compares predicted values with actual values and outputs statistical information
inclueding,accuracy results and a confusion matrix
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

class glVar():
    folder_base = "/Users/docmis/OneDrive - Rutgers University/Rutgers/Research/SDR/Data/"
    y_pred = []
    y_pred_num = []
    y_pred_cat = []
    y_true = []
    y_true_num = []
    comp = []
    acc = 0
    conf_mat = []
    temp1 = None
    temp2 = None

#Sets up the arrays and matrices
def setupArrays(y_pred = "", y_true = "", labels = []):
    # Reads in predcition data
    if y_pred == "": glVar.y_pred = pd.read_csv(glVar.folder_base + "/test_pred.csv").values
    else: glVar.y_pred= y_pred
    
    # Sets up numeical arrays for the predictions
    glVar.y_pred_cat = np.zeros_like(glVar.y_pred)
    glVar.y_pred_cat[np.arange(len(glVar.y_pred)), glVar.y_pred.argmax(1)] = 1
    glVar.y_pred_num = np.asarray(np.asmatrix(glVar.y_pred_cat*list(range(1, len(labels)+1))).sum(axis = 1))
    
    if y_true == "": glVar.y_true = pd.read_csv(glVar.folder_base + "/test_y.csv").values
    else: glVar.y_true = y_true
    glVar.y_true_num = np.asarray(np.asmatrix(glVar.y_true*list(range(1, len(labels)+1))).sum(axis = 1))
    
    glVar.comp = (glVar.y_pred_num == glVar.y_true_num).astype(int)
    glVar.acc = np.round((sum(glVar.comp)/glVar.comp.shape[0])[0], 2)
    #plot_confusion_matrix_mc(glVar.y_true, glVar.y_pred, glVar.y_labels, "test")

#Code from
#https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels/48018785
#from sklearn.metrics import confusion_matrix
def confusionMatrixPlot(y_test, pred, labels_num, labels_text = [], 
                        myFolder = "", myFile = ""):
    cm = confusion_matrix(y_test, pred, labels_num)
    glVar.conf_mat = cm
    print("\nConfusion Matrix")
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = LinearSegmentedColormap.from_list('RedGreenRed', ['red', 'white'])
    cax = sns.heatmap(cm, annot=True, ax = ax, cmap = mpl.cm.Blues);
    #cax = ax.matshow(cm)
    #plt.title("Confusion matrix of the classifier\n")
    #fig.colorbar(cax)
    if np.asarray(labels_text).shape[0] == 0: labels_text = labels_num
    ax.xaxis.set_ticks_position('top')
    ax.set_xticklabels(labels_text)
    ax.set_yticklabels(labels_text)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    #plt.show()
    plt.savefig(myFolder + myFile + "_conf_matrix")
    plt.close()
    return cm
# %%
# Code modified from: 
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
def getAccStats(cm):
    tp = np.diag(cm)
    fp = np.sum(cm, axis= 0) - tp
    fn = np.sum(cm, axis= 1) - tp
    tn = np.sum(np.sum(cm, axis= 1)) - np.sum(cm, axis= 0)
    #print("tp: ", tp)
    #print("tn: ", tn)
    #print("fp: ", fp)
    #print("fn: ", fn)
    #print("Smaples: ", np.sum(cm, axis= 1))
    prec = tp/(tp + fp)
    recall = tp/(tp + fn)
    acc = (tp + tn)/(tp +tn + fn + fp)
    f1 = 2*prec*recall/(prec+recall)
    return prec, recall, acc, f1

#%%
#Runs the main operations of the program
def main(y_pred = "", y_true ="", labels= ["16qam", "8psk", "bpsk", "qpsk"], 
         myFolder = "", myFile = ""):
    setupArrays(y_pred, y_true, labels)
    
    labels_num = list(range(1, len(labels)+1))
    cm = confusionMatrixPlot(glVar.y_true_num, glVar.y_pred_num, labels_num, 
                        labels, myFolder, myFile)

    np.savetxt(myFolder + myFile + "_conf_mat.csv", cm, delimiter=',', 
           header= ','.join(map(str, labels)))
    prec, recall, acc, f1 = getAccStats(cm)
    
    pd.DataFrame({
        "labels": labels, 
        "Precision ": prec,
        "Recall" : recall, 
        "Accuracy": acc, 
        "F1": f1
        }).to_csv(myFolder + myFile + "_conf_stats.csv", mode = 'a')


    return cm


#%%
if __name__ == '__main__':
    main()
   