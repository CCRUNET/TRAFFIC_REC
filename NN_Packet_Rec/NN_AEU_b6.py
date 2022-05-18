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
# from keras import utils as np_utils
# from tensorflow.keras.utils import to_categorical

#from keras.utils import multi_gpu_model
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TENSORFLOW_FLAGS"] = "device=gpu%d" % (0)
np.random.seed(1200)  # For reproducibility

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import files
import AEU_Clustering_r1 as ae_clus
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
    #lab = pd.get_dummies(lab)
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
        input_img = Input(shape=(X_train.shape[1], X_train.shape[2], 1)) 
        time_train_start = time.time()
        #Training model
        if train_model:
            #Removes previous files to prevent file creation errors
            if not os.path.exists(folder_NN_hist): os.makedirs(folder_NN_hist)
            if os.path.exists(weight_file): os.remove(weight_file)
            if os.path.exists(model_file): os.remove(model_file)
            if os.path.exists(model_file): os.remove(model_file_encoder)
            if os.path.exists(hist_file): os.remove(hist_file)

            # encoded representation of input
            encoded = self.encode(input_img)
            # decoded representation of code 
            decoded = self.decode(encoded, samples = input_img.get_shape().as_list()[1])
        
            # Model which take input image and shows decoded images
            autoencoder = Model(inputs=input_img, outputs=decoded, name = 'autoencoder')

            # Creating encoder model
            encoder = Model(inputs = input_img, outputs=encoded, name = 'encoder')
            encoder.save(model_file_encoder) 
            encoding_dim = 256
            encoded_input = Input(shape=(encoding_dim,))
            # last layer of the autoencoder model
            decoder_layer = autoencoder.layers[len(autoencoder.layers) - 1]
            # decoder model
            decoder = Model(encoded_input, decoder_layer(encoded_input), name = 'decoder')
            # Prints a summary of each mode
            encoder.summary()
            decoder.summary()
            autoencoder.summary()
            
            autoencoder.compile(optimizer='adam',
                              loss='mse',
                              metrics=['accuracy'])  
            
            hist = autoencoder.fit(X_train, X_train,
                              epochs=epochs, batch_size=batch_size, 
                              validation_data=(X_val, X_val),
                              shuffle=False) 
            model = autoencoder
            model.save_weights(weight_file)
            model.save(model_file)
            
            #loss_test_train = np.asarray(hist.history['loss'])
            #acc_test_train = np.asarray(hist.history['accuracy'])
            loss_val_train = np.asarray(hist.history['val_loss'])
            acc_val_train = np.asarray(hist.history['val_accuracy'])
            pd.DataFrame({"loss_val_train" : loss_val_train,
                           "acc_val_train" : acc_val_train,
                            }).to_csv(hist_file)

        else:
            model = load_model(model_file)
            #model()
            hist = pd.read_csv(hist_file)
            #print(hist)
            loss_val_train = hist["loss_val_train"].values
            acc_val_train = hist["acc_val_train"].values                    
        time_train = np.round(time.time() - time_train_start, 2)
        time_test_start = time.time()

        pred_ae = autoencoder.predict(X_test)
        pred_enc = encoder.predict(X_test)
        pred_clus = ae_clus.findClusters2(Y_test = Y_test, pred = pred_enc)
        score = model.evaluate(X_test, X_test, verbose=1)
        #Gets and outputs predciton of each class
        acc, pred  = clus_acc.cluster_acc(Y_test, pred_clus)

        time_test = np.round(time.time() - time_test_start, 2)

        print("Train Data Shape: ", X_train.shape)
        print("Accuracy ", acc)
        print("Loss: ", score[0], '\n')

        #Validation loss is taken as the final value in the array of validation loss in the training data
        #Returns Test loss, test accuracy, validation loss, validation accuracy 
        return (acc, float(score[1]), float(loss_val_train[0]), 
                float(acc_val_train[0]), pred, act1, act2, time_train, time_test)  


#%%    
def test():
    NNet = NN()
    NNet.__init__
    x, y = generateSequence(20, 10)
    #x = x.reshape(-1, 1, x.shape[1], 1)
    print(x.shape)
    x = x.reshape(-1, 1, x.shape[1], 1) / np.max(x)
    x = NNet.shuffleData(x)
    y = NNet.shuffleData(y)
    #y = keras.utils.to_categorical(y)
    glVar.temp = x

    a, b, c = NNet.genTrainTest(x)
    a1, b1, c1 = NNet.genTrainTest(y)
    #NNet.runAutoencoder(a, a, b, b)
    NNet.runNN(a, a1, b, b1, c, c1, train_model = True)

#%% Testing Autoencoder alone
if __name__ == '__main__':
    #pred = test()
    test()

