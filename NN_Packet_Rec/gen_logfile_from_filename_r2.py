# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 21:00:09 2020

@author: TINA
"""

#%% imports the appropriate libraries
#!/usr/bin/env python3
import numpy as np, signal, sys, pandas as pd, os
from argparse import ArgumentParser
from datetime import datetime

#%% Setups up global variables
class glVar:
    temp = None
    data_path = ""
    date_code = ""

#%% Parses user input
def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--files-test", dest="files_test", nargs = '+', type=str, 
        default= [""],
        help="List of files or folders to test [default=%(default)r]")
    parser.add_argument(
        "--col-vals", dest="col_vals", nargs = '+', type=str,  
        default= [""],
        help="Keywords to identify files by [default=%(default)r]")
    parser.add_argument(
        "--col-name", dest="col_name", type=str,  
        default= "s1_mod",
        help="Name of column header for logfile [default=%(default)r]")
    return parser

#%%
def get_file_list(loc_data):     
    #a = [*set([loc_data + "/" + p for p in os.listdir(loc_data)])]
    data = []
    for root, dirs, files in os.walk(loc_data):
        for f in files:
            if (f.lower().find("log") > -1 or f.lower().find(".txt") > -1 or 
                f.lower().find(".csv") > -1) : print(f + " is being excluded")
            else: 
                data.append((root + "/" + f).replace("\\", "/"))
                print(f)                                                    
    return data
#%%
def get_info_from_name(data_path, filenames, col_name, col_vals):
    glVar.header = True
    for f in filenames: 
        f = os.path.basename(f)
        fname = f.upper().replace('/','_')

        
        for val in col_vals:
            if fname.find(val.upper()) > -1: mod = val; break;
            else: mod = "NA"

        pd.DataFrame({
        "filename": [f],
        col_name: [mod],
        "s1_sinr": [100],
        }).to_csv( data_path +"/logfile_" + glVar.dateCode + ".csv", mode = 'a', 
                  header = glVar.header)
        glVar.header = False

        
    return 0

#%%Main program
def main(options = None):
    if options is None:
        options = argument_parser().parse_args()        
    
    def sig_handler(sig=None, frame=None):
        sys.exit(0)
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    
    glVar.dateCode = str(datetime.now()).replace('.', '').replace(' ', '').replace(':', '').replace("-","")
    for data_path in options.files_test:
        get_info_from_name(data_path, get_file_list(data_path), options.col_name, options.col_vals)
    return 0

#%%
if __name__ == '__main__':
    main()
   