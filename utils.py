# Import the required packages
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast, os
import bambi as bmb
import pymc as pm
import arviz as az
import scipy.stats as stat
from collections import Counter
import itertools
import hssm

def clean_data(filename, which_data = 'mem data', root_dir = 'source_mem_data'):
    # print(filename)
    # Handle errors. If there is an error, go to 'except' and return nothing.
    try:
        data = pd.read_csv(f'{root_dir}/{filename}')
    
        # Drop instruction rows by dropping rows with missing data in column: 'blocks.thisRepN'
        data = data.dropna(subset=['trials.thisRepN']).reset_index(drop=True)
    
        #If data file is incomplete, raise an error. 
        #Label conditions based on participant number as was designed in the experiment
        if data['participant'][0]%2 == 0:
            data['condition'] = 'structured'
        else:
            data['condition'] = 'unstructured'
    
        # data['trial'] = np.arange(len(data))
        
        
        #filtering exposure data, removing rows with no path id
        exposure  = data.loc[data['path id'].notna(), ['trials.thisN', 'path id', 'stim chosen', 'key_resp.rt', 'key_resp.corr', 'condition']].reset_index(drop=True)
        exposure.rename(columns = {'trials.thisN': 'trials', 'key_resp.corr':'accuracy', 'key_resp.rt':'rt'}, inplace = True)
        #New column to define a block    
        exposure['block'] = np.repeat(np.arange(3), 250)
        #node types
        exposure['node type'] = 'nonboundary'
        exposure.loc[(exposure['path id'].isin([0, 4, 5, 9, 10, 14])), 'node type'] = 'boundary'
        exposure['participant'] = data['participant'][0]
        exposure['transition_type'] = ['cross cluster' if (exposure['node type'] == 'boundary')[i] & (exposure['node type'].shift() == 'boundary')[i] else 'within cluster' for i in range(len(exposure))]
    
        #extracting actual boundary vs non boundary stimuli files that were shown. 
        boundary_stim = exposure.loc[exposure['node type'] == 'boundary', 'stim chosen'].unique()
        nonboundary_stim = exposure.loc[exposure['node type'] == 'nonboundary', 'stim chosen'].unique()
        
        #Filtering memory data, removing rows with nothing in the 'old or new' column
        memory = data.loc[data['old or new'].notna(), ['mem_test_key_resp.rt', 'old or new', 'old or new accuracy', 'test item', 'condition']].reset_index(drop=True)
        memory['trials'] = np.arange(90)
        memory['block'] = np.repeat(np.arange(3), 30)
        #Node types
        memory['node type'] = 'new'
        #Testing whether if each test item is in the boundary stim list
        memory.loc[memory['test item'].isin(boundary_stim), 'node type'] = 'boundary'
        memory.loc[memory['test item'].isin(nonboundary_stim), 'node type'] = 'nonboundary'
        memory.rename(columns={'mem_test_key_resp.rt':'rt', 'old or new accuracy':'accuracy'}, inplace=True)
        #Translating string rts to numerical rts
        memory['rt'] = [ast.literal_eval(i)[0] for i in memory['rt']]
        memory['participant'] = data['participant'][0]

    except:
        return None

    #Return the dataframe with relevant columns
    if which_data == 'mem data':
        return memory 
    else:
        return exposure
        # return filtered_data.loc[filtered_data['old or new'].isna()].reset_index(drop=True)
        
