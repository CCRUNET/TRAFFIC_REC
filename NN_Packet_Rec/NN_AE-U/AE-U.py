#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:43:25 2019
@author: tina-mac2

Devoloper:  Tina Burns
School: Rutger University
Advisor: Rchard Martin
Python Version: Python3
Code Revision: R3

This code for an Unsperivsed Autencoder based on Keras 
"""
# %%
#Imports necessary libraries 
import os, keras, numpy as np, time, pandas as pd
import tensorflow as tf
from keras.layers import Dense, Flatten, Dropout
from keras import backend as K 
from keras.layers import Input
from keras.models import Model
from keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
from keras import utils as np_utils
from tensorflow.keras.utils import to_categorical


#from keras.utils import multi_gpu_model
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TENSORFLOW_FLAGS"] = "device=gpu%d" % (0)
np.random.seed(1200)  # For reproducibility

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# %%
class glVar():
    temp = None
    temp_1 = None
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
    # print(arr1.shape)
    return (arr, lab)

# %% Neural Network Class
# This neural network is fully connected Sequential network using the keras library
# Parts of the code have been modified from:
# https://www.analyticsvidhya.com/blog/2021/06/complete-guide-on-how-to-use-autoencoders-in-python/
class NN():
    def __init__(self):
        """"""
        # clears preious keras nn session
        K.clear_session()
        
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

    # encoder
    def encode(self, hidden, act1 = "relu"):
        #Encoder
        #hidden = Flatten()(hidden)
        hidden = Dense(256, activation= act1)(hidden)
        hidden = Dense(128, activation= act1)(hidden)
        hidden = Dense(64, activation = act1)(hidden)
        hidden = Dense(32, activation = act1)(hidden)
        return hidden
    
    #Decodeds data
    def decode(self, hidden, act1 = "relu", samples = 1000):
        #Decoder
        #hidden1 = Flatten()(hidden1)
        hidden = Dense(32, activation= act1)(hidden)
        hidden = Dense(64, activation=  act1)(hidden)
        hidden = Dense(128, activation= act1)(hidden)
        hidden = Dense(256, activation= act1)(hidden)
        hidden = Dense(samples, activation= act1)(hidden)
        return hidden
    
                
    def runNN(self, X_train, Y_train, X_test, Y_test, X_val, Y_val, 
              Y_train_label= None, Y_test_label = None, 
              Y_val_label = None, act1 = "relu",  act2 = "softmax", 
              epochs = 10, batch_size = 128, testAct = False, mod = '',
              train_model = True, folder_NN_hist = "NN_Hist"):   
        #Sets files to store training information
        weight_file = os.path.join(folder_NN_hist, "AE-U_weights.h5").replace(r'\'', '/'); 
        model_file = os.path.join(folder_NN_hist, "AE-U_model").replace(r'\'', '/'); 
        model_file_encoder = os.path.join(folder_NN_hist, "AE-U_encoder_model.h5").replace(r'\'', '/'); 
        hist_file = os.path.join(folder_NN_hist, "AE-U_history.csv").replace(r'\'', '/');
        
        if not testAct: act1 = "elu"; act2 = "softmax" 

        num_classes = Y_train.shape[1]
        input_img = Input(shape=(X_train.shape[1],))  
         
        time_train_start = time.time()
        #Training model
        if train_model:
            #Removes previous files to prevent file creation errors
            if not os.path.exists(folder_NN_hist): os.makedirs(folder_NN_hist)
            if os.path.exists(weight_file): os.remove(weight_file)
            if os.path.exists(model_file): os.remove(model_file)
            if os.path.exists(model_file): os.remove(model_file_encoder)
            if os.path.exists(hist_file): os.remove(hist_file)
            
            glVar.temp = input_img
            # encoded representation of input
            encoded = self.encode(input_img)
            # decoded representation of code 
            decoded = self.decode(encoded, samples = input_img.get_shape().as_list()[1])
        
            # This model shows encoded images
            encoder = Model(inputs = input_img, outputs=encoded, name = 'encoder')
            encoder.save(model_file_encoder)
            
            # Model which take input image and shows decoded images
            autoencoder = Model(inputs=input_img, outputs=decoded, name = 'autoencoder')
            # Creating a decoder model
            encoding_dim = 256
            encoded_input = Input(shape=(encoding_dim,))
            # last layer of the autoencoder model
            decoder_layer = autoencoder.layers[len(autoencoder.layers) - 1]
            # decoder model
            decoder = Model(encoded_input, decoder_layer(encoded_input), name = 'decoder')

            encoder.summary()
            decoder.summary()
            autoencoder.summary()
        
            autoencoder.compile(optimizer='adam',
                              loss='binary_crossentropy',
                              metrics=['accuracy'])  
            hist = autoencoder.fit(X_train, X_train,
                              epochs=epochs, batch_size=batch_size, 
                              #validation_data=(X_val, Y_train),
                              shuffle=False) 
            autoencoder.save_weights(weight_file)
            
            
    
#%%
# Code from: 
# https://github.com/ardamavi/Unsupervised-Classification-with-Autoencoder/blob/master/Examples/Dog-Cat/Dog-Cat%20Classification%20With%20Autoencoder.ipynb
def findClusters(encoder, X_train, Y_train, num_class = 4):
    encode = encoder.predict(X_train)
    glVar.temp = encode
    sample = X_train.shape[2]; 
    class_dict = np.zeros((num_class, num_class))
    for i, sample in enumerate(Y_train):
        class_dict[np.argmax(encode[i], axis=0)][np.argmax(sample)] += 1
        
    print(class_dict)
        
    neuron_class = np.zeros((num_class))
    for i in range(num_class):
        neuron_class[i] = np.argmax(class_dict[i], axis=0)
    
    print(neuron_class)   
    
    return 0

#%%    
def test():
    NNet = NN()
    NNet.__init__
    x, y = generateSequence(100, 100)
    #x = x.reshape(-1, 1, x.shape[1], 1)
    print(x.shape)
    x = NNet.shuffleData(x)
    y = NNet.shuffleData(y)
    y = keras.utils.to_categorical(y)
    a, b, c = NNet.genTrainTest(x)
    a1, b1, c1 = NNet.genTrainTest(y)
    #NNet.runAutoencoder(a, a, b, b)
    NNet.runNN(a, a1, b, b1, c, c1, train_model = True)

#%% Testing Autoencoder alone
if __name__ == '__main__':
    #pred = test()
    test()

