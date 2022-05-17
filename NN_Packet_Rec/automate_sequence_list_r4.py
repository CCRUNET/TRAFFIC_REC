# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 01:27:31 2022
Devoloper:  Tina Burns
School: Rutger University
Advisor: Rchard Martin
Python Version: Python3

This code sequeces to through neural netww
"""
# %%
#Imports necessary libraries 
''' '''
#imports appropriate files
import arg_parser
import glob_vars as globs 
glVar = globs.glVar
NN_data = globs.NN_data
import main_nn_packet_rec_r7p3   as NN_main
# %% 
class loc_var():
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# %% Main unction that cycles through the exlcusion list
def main(options=None):
    if options is None:
        options = arg_parser.argument_parser().parse_args()
    globs.glVar.options = options
    
    # When working with an inclusion list
    if options.seq_type.lower() == "inc" or options.seq_type.lower()=="exc": 
        arr = options.seq_test
        n = len(arr)
        for i in range(options.seq_start-1, n):
            options.seq_test = arr[0: i+1]
            o = NN_main.main(options)
            # Store data file to global
            options.file_results = o.file_results
    return 0
#%% Runs main port of program
if __name__ == '__main__':
    main()
   