#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:43:25 2019
@author: tina-mac2

Devoloper:  Tina Burns
School: Rutger University
Advisor: Rchard Martin
Python Version: Python3
Code Revision: R1

This code runs a categorical keras network.  
"""
# %%
#Imports necessary libraries
import numpy as np, os, pandas as pd, time
import keras
from sklearn_som.som import SOM
seed(1337)
from tensorflow.keras.models import load_model
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TENSORFLOW_FLAGS"] = "device=gpu%d" % (0)

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
    #lab = np.array(pd.get_dummies(lab.reshape(lab.shape[0])))
    # print(arr1.shape)
    return (arr, lab)

#%% Neural Network Class
class NN():
    def __init__(self):
        """"""
        # clears preious keras nn session
        K.clear_session()
    
    def getType(self):
        return "SOM"
    
    # This function splits the array into three seperate arrays
    def genTrainTest(self, arr, axis=0):
        sep = int(arr.shape[0] / 5)
        return np.split(arr, [sep * 3, sep * 4], axis=axis)
   
    # Shuffles Data
    def shuffleData(self, x):
        np.random.seed(1200)
        myPermutation = np.random.permutation(x.shape[0])
        x = x[myPermutation]
        return x
    #Main Neural Network
    def NN_CONV(self, data, act1 = "elu", act2 = "softmax", numClasses = 2):   

        return output

    #Main NN Execution
    def runNN(self, X_train, Y_train, X_test, Y_test, X_val, Y_val, 
              Y_train_label= None, Y_test_label = None, 
              Y_val_label = None, act1 = "softplus",  act2 = "softmax", 
              epochs = 10, batch_size = 128, testAct = False, mod = '', NN_layers = "CONV",               
              train_model = True, test_model = True, folder_NN_hist = "NN_Hist"):

        weight_file = os.path.join(folder_NN_hist, "SOM_weights.h5").replace(r'\'', '/');
        model_file = os.path.join(folder_NN_hist, "SOM_model").replace(r'\'', '/'); 
        hist_file = os.path.join(folder_NN_hist, "SOM_history.csv").replace(r'\'', '/');

        time_train_start = time.time()
        if train_model:
            #Removes previous files to prevent file creation errors
            if not os.path.exists(folder_NN_hist): os.makedirs(folder_NN_hist)
            if os.path.exists(weight_file): os.remove(weight_file)
            if os.path.exists(model_file): os.remove(model_file)
            if os.path.exists(hist_file): os.remove(hist_file)
            output = self.NN_CONV(input_img, numClasses = Y_train.shape[1], 
                                    act1 = act1, act2 = act2)               
            model = None
            print(model.summary()) 
    
            model.compile(optimizer='Adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
            
            hist = model.fit(X_train, Y_train,
                             epochs=epochs,batch_size=batch_size, 
                             validation_data=(X_val, Y_val),
                             shuffle=True) 
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
            hist = pd.read_csv(hist_file)
            #print(hist)
            loss_val_train = hist["loss_val_train"].values
            acc_val_train = hist["acc_val_train"].values                    
        time_train = np.round(time.time() - time_train_start, 2)
        
        if test_model:
            time_test_start = time.time()
            score = model.evaluate(X_test, Y_test, verbose=1)
            #Gets and outputsS predciton of each class
            pred = model.predict(X_test, verbose = 1)
            time_test = np.round(time.time() - time_test_start, 2)
        else:
            time_test = 0
            pred = [0]
            score = [0, 0]

        #print("Data shape: ", x.shape)
        print("Train Data Shape: ", X_train.shape)
        print("Loss: ", 0)
        print( '\n', "Accuracy ", acc)
        #print("NN Time: ", time_test + time_train)

        #Validation loss is taken as the final value in the array of validation loss in the training data
        #Returns Test loss, test accuracy, validation loss, validation accuracy 
        return (0, acc, 0, 0, 
                pred, act1, act2, time_train, time_test)
# %% Execution
def test():
    NNet = NN()
    NNet.__init__
    x, y = generateSequence(500, 100)    
    x = NNet.shuffleData(x)
    y = NNet.shuffleData(y)
    y = keras.utils.to_categorical(y)

    #This shapes the array so that it is evenly divisible by 4
    #which allows it to be correctly processed by the nueral network
    x = x[:, 0:(x.shape[1] - x.shape[1]%4)]

    # Reshapes the data so that it is 4 dimensional
    # Seperates test and data into training, test, and validiation sets
    x = x.reshape(-1, 1, x.shape[1], 1) / np.max(x)
    #y = y.reshape(-1, 1, y.shape[1], 1) / np.max(y)
    X_train, X_test, X_val = NNet.genTrainTest(x)
    Y_train, Y_test, Y_val = NNet.genTrainTest(y)
    
    NNet.runNN(X_train, Y_train, X_test, Y_test, X_val, Y_val,epochs=10,
                                train_model = True)
    #print(pred)
    # return acc, loss, pred
    return 0

if __name__ == '__main__':
    for i in range (1, 2):
        test()


    