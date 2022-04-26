# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 20:39:56 2021
Devoloper:  Tina Burns
School: Rutger University
Advisor: Rchard Martin
Python Version: Python3
Project: Packet Recognition Using Machine Learning
Description: This file provide list of user inputs that will be parsed
the main program.

"""
from argparse import ArgumentParser

#%%
def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--folder-train", dest="folder_train", type=str, 
        default= "C:/Users/TINA/OneDrive - Rutgers University/Rutgers/Research/SDR/Data/2020-06-10_additive-noise_gNoise-2",
        help="Sets the training data folder location [default=%(default)r]")
    parser.add_argument(
        "--folder-test", dest="folder_test", type=str, 
        default= "",
        help="Set testing data folder location [default=%(default)r]")
    parser.add_argument(
        "--file-results", dest="file_results", type=str, 
        default= "",
        help="Sets the location for the results summary [default=%(default)r]")
    parser.add_argument(
        "--neural-nets", dest="NNets", type=str, nargs='+', 
        default= ["CAT", "CONV", "LSTM", "BIN", "ANOM",  "AE"],
        help='''Enter the list of nueral networks to be tested. \n
        BIN --> Binary Classifier \n
        FCN --> Categorical \n
        CNN --> Convolutional \n
        AE --> Autoencoder \n
        AMOM --> Anomaly Detector \n
        SIMP--> Simple \n
        Options [default=%(default)r]'''
        )
    parser.add_argument(
        "--iter", dest="iter", type=int, 
        default= "1",
        help="Number of iterations [default=%(default)r]")
    parser.add_argument(
        "--samples", dest="samples", type=int, 
        default= "10000",
        help="Number of samples [default=%(default)r]")
    parser.add_argument(
        "--num-points", dest="num_points", type=int, 
        default= 1000,
        help="Number of datapoints [default=%(default)r]")
    parser.add_argument(
        "--num-points-train", dest="num_points_train", type=int, 
        default= "250",
        help="Number of samples [default=%(default)r]")
    parser.add_argument(
        "--test-act", dest="test_act", type=str, 
        default= "0",
        help="Do you want to test activation functions? Enter 0 for no and 1 for yes. [default=%(default)r]")    
    parser.add_argument(
        "--conf-mat", dest="conf_mat", type=int, 
        default= 0,
        help="Do you want to plot confusion matrices? Enter 0 for no and 1 for yes. [default=%(default)r]")    

    parser.add_argument(
        "--exc-param", dest="exc_param", type=str, 
        default= "s1_mod",
        help="Name of the parameter colume for the exclusion list [default=%(default)r]")
    parser.add_argument(
        "--exc-train", dest="exc_train", type=str, nargs='+',
        default= [""],
        help="Values in training data to be exluded [default=%(default)r]")
    parser.add_argument(
        "--exc-test", dest="exc_test", type=str, nargs='+',
        default= [""],
        help="Values in test data to be exluded [default=%(default)r]")
    parser.add_argument(
        "--inc-arr", dest="inc_arr", type=str, nargs='+',
        default= [""],
        help="Values in data to be included in the test c_default=%(default)r]")
    parser.add_argument(
        "--exc-seq", dest="exc_seq", type = int,
        default= 0,
        help='''
        Indicates if the exclusion we should sequence through the exclsion array \n\r
        0 --> do not sequence \n\r
        1 --> Sequence
        [default=%(default)r]
        ''')
    parser.add_argument(
        "--exc-start", dest="exc_start", type = int,
        default= 1,
        help='''Indicates the element of the exlusion array to where the sequence
        should start
        [default=%(default)r]''')
    
    
    parser.add_argument(
        "--range-param", dest="range_param", type=str, 
        default= "s1_sinr",
        help="Name of the parameter to be evaulated based on range [default=%(default)r]")
    parser.add_argument(
        "--range-train", dest="range_train", type=float, nargs='+',
        default= [-1000.0, 1000.0],
        help="Range of values in training data to be included [default=%(default)r]")
    parser.add_argument(
        "--range-test", dest="range_test", type=float, nargs='+',
        default= [-1000.0, 1000.0],
        help="Range of values in test data to be included [default=%(default)r]")
    
    parser.add_argument(
        "--col-param", dest="col_param", type=str, 
        default= "s1_sinr",
        help="Name of column in logfile for parameter to be tested [default=%(default)r]")  
    # parser.add_argument(
    #     "--col-filename", dest="col_filename", type=str, 
    #     default= "filename",
    #     help="Name of column in logfile for parameter for filenames [default=%(default)r]")        
    parser.add_argument(
        "--col-mods", dest="col_mods", type=str, 
        default= "s1_mod",
        help="Enter the name of column header for modulations [default=%(default)r]")
    parser.add_argument(
        "--logfile", dest="logfile", type=str, nargs='+', 
        default= [""],
        help="List of logfiles [default=%(default)r]")
    parser.add_argument(
        "--NN-train", dest="NN_train", type=int, 
        default= 1,
        help="Do you want to train the neural network? Enter 0 for no and 1 for yes. [default=%(default)r]")   
    parser.add_argument(
        "--NN-Hist-folder", dest="NN_Hist_folder", type=str, default= "NN_Hist",
        help="Sets the training data folder location [default=%(default)r]")
    parser.add_argument(
        "--data-type", dest="data_type", type=str, 
        default= "float32",
        help='''Sets the data type for reading the file  [default=%(default)r]
        int8 --> 8 bit signed integar '\n'
        int16 --> 16 bit signed integar '\n'
        int32 --> 32 bit signed integar '\n'
        int64 --> 64 bit signed integar '\n'
        uint8 --> 8 bit unsigned integar '\n'
        uint16 --> 16 bit unsigned integar '\n'
        uint32 --> 32 bit unsigned integar '\n'
        uint64 --> 64 bit unsigned integar '\n'        
        float32 --> 32 bit floating point number '\n'
        float64 --> 64 bit floating point number '\n'
        ''')    
    parser.add_argument(
        "--rclone-loc", dest="rclone_loc", type=str,
        default= "0",
        help="Rclone location for the data to be copied to [default=%(default)r]")
    return parser