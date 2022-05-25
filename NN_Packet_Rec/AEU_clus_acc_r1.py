# -*- coding: utf-8 -*-
"""
Created on Sat May 14 22:03:34 2022
Devoloper:  Tina Burns
School: Rutger University
Advisor: Rchard Martin
Python Version: Python3
Code Revision:

# Code from
# https://github.com/k-han/DTC/blob/master/utils/util.py
# https://datascience.stackexchange.com/questions/17461/how-to-test-accuracy-of-an-unsupervised-clustering-model-output
# Modified by Tina Burns for research purposes
"""

# %%
from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment as linear_assignment
# 'sklearn.utils.linear_assignment_'
import os, argparse
import random

import warnings                                                                                                                
warnings.filterwarnings('ignore')

#%%
class glVar():
    temp = None

# %%
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
# %%
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
# %%
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    # Determine if y_true is  categorical array. 
    if len(y_true.shape)>1: y_true = cat_to_num(y_true)
    y_true = y_true.astype(np.int64)
    
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    #print("Y Pred: ", y_pred)
    #print("Y True: ", y_true)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # Get assignment of actual label to predicted classes
    '''Clustering algorithms can randomly assign labels to clusters 
    This maps the corresponding predicted class labels to the
    to the actual class labels'''
    ind = linear_assignment(w.max() - w) 
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    pred_class = label_pred_arr(y_pred, ind)
    return (acc, pred_class)

# %%
def label_pred_arr(y_pred, clus_lab):
    pred_class = np.full((y_pred.shape[0]), -1)    
    for i in range (0, clus_lab.shape[0]): 
        pred_class[np.where(y_pred==i)] = clus_lab[i, 1]
    return pred_class
#%% Converts a categorical array to a numerical array
def cat_to_num(arr_cat):
    lab = np.arange(arr_cat.shape[1])  + 1
    arr_num = np.sum(arr_cat*lab, axis = 1)-1
    return arr_num
