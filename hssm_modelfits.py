# import warnings
# warnings.filterwarnings('ignore')
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
from utils import *
from hssm.likelihoods import DDM
import nutpie

hssm.set_floatX("float32")

#reads in all the NAMES of the data files from the 'data' folder. 
data_files = []
for f in os.listdir('source_mem_data/'):
    if (f.startswith('24') & f.endswith('csv')):
        data_files.append(f)
        
df_clean_exposure = pd.concat([clean_data(f, which_data='non mem data') for f in data_files]).reset_index(drop = True)
df_clean_memory = pd.concat([clean_data(f) for f in data_files]).reset_index(drop = True)

df_clean_memory['resp_old'] = 1
df_clean_memory.loc[((df_clean_memory['old or new'] == 'new') & df_clean_memory.accuracy), 'resp_old'] = 0
df_clean_memory.loc[((df_clean_memory['old or new'] == 'old') & (1 - df_clean_memory.accuracy)), 'resp_old'] = 0


df_clean_memory['response'] = df_clean_memory['resp_old'].astype(int)
df_clean_memory.loc[df_clean_memory.resp_old == 0, 'response'] = -1
# df_clean_memory['response'] = df_clean_memory['accuracy'].astype(int)
# df_clean_memory.loc[df_clean_memory.accuracy == 0, 'response'] = -1

df_clean_memory['node_type'] = df_clean_memory['node type']
df_clean_memory['participant_id'] = df_clean_memory['participant']


df_clean_memory['node_type'] = df_clean_memory['node type']
df_clean_memory['participant_id'] = df_clean_memory['participant']
# df_clean_memory_finalblock = df_clean_memory.loc[df_clean_memory['block'] == 2].reset_index(drop=True)

df_clean_exposure_grouped = df_clean_exposure.groupby(['stim chosen', 'participant', 'block']).mean(numeric_only=True).reset_index()
df_clean_exposure_acc = df_clean_exposure_grouped.groupby(['participant', 'block']).mean(numeric_only=True).reset_index()

#Join first to incorporate exposure accuracy for old stimuli
df_clean_memory_joined = pd.merge(df_clean_memory, df_clean_exposure_grouped, left_on=['participant', 'test item', 'block'], 
                                  right_on=['participant', 'stim chosen', 'block'], suffixes=('', '_exposure'), how = 'left')


#Replace the new stimuli exposure accuracy NaNs with averaes for that participant/block.
for ppt in df_clean_exposure_acc.participant.unique():
    for b in range(3):
        df_clean_memory_joined.loc[((df_clean_memory_joined.node_type == 'new') & (df_clean_memory_joined.participant == ppt) & (df_clean_memory_joined.block == b)), 'accuracy_exposure'] = df_clean_exposure_acc.loc[((df_clean_exposure_acc.participant == ppt)&(df_clean_exposure_acc.block == b)), 'accuracy'].values[0]

df_clean_memory_joined = df_clean_memory_joined.loc[((df_clean_memory_joined.rt > 0.25) & (df_clean_memory_joined.rt < 15))].reset_index(drop=True)

t_range = np.linspace(0.05, 0.25, 30)
ddm_model_samples = {}

# for t in t_range:
#     ddm_model = hssm.HSSM(hierarchical=False, prior_settings='safe', t = t, loglik_kind="approx_differentiable", 
#                                data = df_clean_memory_joined[['participant_id', 'response', 'rt', "node_type", "condition", "block", "accuracy_exposure"]],  
#                           include=[
#                                     {
#                                         "name": "v",
#                                         "formula": "v ~ 0 + C(node_type):C(condition):C(block) + accuracy_exposure",
#                                         "link": "identity"
#                                         # "prior": {"name": "Normal", "mu": 0, "sigma": 0.01},
#                                     },
#                               {
#                                   "name": "z",
#                                   "formula": "z ~ 0 + C(block)"
#                               },
                              
#                               {
#                                   "name": "a",
#                                   "formula": "a ~ 0 + C(condition):C(block)"
#                               },
                              
#                                   ],
#                          )

    

#     ddm_model_samples[f'v~hrl_nodetype:cond:block+accexp_z~block_a~cond:block_trange_{t}'] = ddm_model.sample(sampler = "nuts_numpyro", idata_kwargs=dict(log_likelihood = True))
#     print("Model done: ", t)
# compare_df = az.compare(ddm_model_samples)
# compare_df.to_csv('hssm_results/t_range/v~hrl_nodetype:cond:block+accexp_z~block_a~cond:block_t_param_compares.csv')


ddm_model = hssm.HSSM(noncentered = True, #hierarchical = True, prior_settings = 'safe',
                      loglik_kind="approx_differentiable", 
                      p_outlier = None, lapse = None,
                      data = df_clean_memory_joined[['participant_id', 'response', 'rt', "node_type", "condition", "block", "accuracy_exposure"]],  
                      include=[
                                {
                                    "name": "v",
                                    "formula": "v ~ 0 + C(node_type):C(condition):C(block) + accuracy_exposure",
                                    # "link": "identity"
                                    # "prior": {"name": "Normal", "mu": 0, "sigma": 0.01},
                                },
                          {
                              "name": "z",
                              "formula": "z ~ 0 + C(block)"
                          },

                          {
                              "name": "a",
                              "formula": "a ~ 0 + C(condition):C(block)"
                          },
                          
                          {
                              "name": "t",
                              "formula": "t ~ 1 + (1 | participant_id)",
                              "prior": { 
                                  "initval": 0.0001,
                                  "name": "Uniform", 
                                  "lower": 0.00001, 
                                  "upper": 0.225
                                       }
                          #     "link": "identity"
                          }
                          
                              ],
                     )

ddm_model_samples = ddm_model.sample(sampler = "nuts_numpyro", tune = 3000)
az.to_netcdf(ddm_model_samples, f'hssm_results/v~nodetype:cond:block_z~block_a~cond:block_t~hrl.nc')


