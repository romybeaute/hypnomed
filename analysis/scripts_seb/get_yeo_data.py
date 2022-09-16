#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:42:44 2021

@author: sebastien
"""

#decoding states/expertise diffusion_embedding
#get averaged and pvalues from data on the cluster
#data are too heavy to be downloaded

import os
import pandas as pd
import numpy as np
from nilearn import surface


class p:
    pass

class d:
    pass

cluster = True
if cluster:
    # # careful on cluster, use path from root !!!!
    p.data_path = '/mnt/data/sebastien/data'
    p.results_path = '/mnt/data/sebastien/results/emb_decoding'
    p.surf_path = '/mnt/data/sebastien/recon_all_success/freesurfer'

else:
    p.data_path = '/media/sebastien/LaCie/CosmoMVPA/scripts/data'
    p.results_path = '/media/sebastien/LaCie/CosmoMVPA/scripts/results/emb_decoding'
    p.surf_path = '/media/sebastien/LaCie/ERC_Antoine/fmriprep_preprocessing_step/recon_all_success/freesurfer/fsaverage5'

p.repetitions_nb = 200 #the number of parallel nodes used
threshold = 0.1

yeo_r = surface.load_surf_data(os.path.join(p.surf_path,'label/rh.Yeo2011_7Networks_N1000.annot'))
yeo_l = surface.load_surf_data(os.path.join(p.surf_path,'label/lh.Yeo2011_7Networks_N1000.annot'))
yeo = np.hstack([yeo_l,yeo_r])

for job_id in range(p.repetitions_nb):    
    if job_id == 0:
        d.df = pd.read_pickle(os.path.join(p.results_path,f'decoding_results_{job_id+1}.pkl'))
    else:
        current_df = pd.read_pickle(os.path.join(p.results_path,f'decoding_results_{job_id+1}.pkl'))
        for row_id in range(d.df.shape[0]):
            d.df.iloc[row_id]['scores'] = np.concatenate((d.df.iloc[row_id]['scores'],current_df.iloc[row_id]['scores']))
            d.df.iloc[row_id]['coefs'] = np.concatenate((d.df.iloc[row_id]['coefs'],current_df.iloc[row_id]['coefs']))


for analysis_id in range(d.df['coefs'].shape[0]):
    current_df = d.df.iloc[analysis_id]
    current_coefs = np.zeros([current_df['coefs'].shape[0],
                             int(current_df['coefs'].shape[2]/3)])
    freq_total = np.zeros([current_df['coefs'].shape[0],7])

    for dimension_id in range(3):
        dimension_coord = range(0+(dimension_id*20484),(dimension_id+1)*20484)
        current_coefs += np.squeeze(current_df['coefs'][:,:,dimension_coord]!=0)
    for network_id in range(1,8):
        significant_voxels = (yeo==network_id) & (current_coefs>threshold)
        freq_total[:,network_id-1] = np.sum(significant_voxels,1)/np.sum(yeo==network_id)*100
    
    d.df['scores'][analysis_id] = np.mean(current_df['scores'])
    d.df['coefs'][analysis_id] = freq_total
    
    print(np.mean(freq_total,0))
    
d.df.to_pickle(os.path.join(p.results_path,'unaverage_data.pkl'))
