#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 17:20:48 2022

@author: sebastien
"""
import numpy as np
import os
from embedding import embedding
from misc_functions import *
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.stats import ttest_ind
class p:
    pass

import time
start_time = time.time()


p.data_path = '/media/sebastien/LaCie/CosmoMVPA/scripts/data'
p.results_path = '/media/sebastien/LaCie/CosmoMVPA/scripts/results/emb_decoding'
p.surf_path = '/media/sebastien/LaCie/ERC_Antoine/fmriprep_preprocessing_step/recon_all_success/freesurfer/fsaverage5'
p.DMN_mask_path = '/media/sebastien/LaCie/CosmoMVPA/scripts/data/DMN_mask'

# movements = [32,62,65,71,72,73,75,76,102,106]
p.outliers = [50,73,32,65,90,102]

p.radius = 1

p.networks_labels = ['Vis', 'SM', 'DA', 'VA','Lim', 'FP','MTC','IFG','AG','MPFC','PMC']
p.networks_wanted = ['Vis', 'SM', 'DA', 'VA','Lim', 'FP','MTC','IFG','AG','MPFC','PMC']
p.order = np.array(['Vis', 'SM', 'DA', 'VA','FP','Lim','MTC','IFG','AG','PMC','MPFC'])

def bootstrap_corrected(stat_list):
    alpha = 0.10
    boots_rep = stat_list.shape[1]
    tmaxnull = np.sort(stat_list.max(0))    
    threshold = tmaxnull[int(((1-alpha)*boots_rep))]

def radarchart(p):
    boot_rep = 10000
    yeo = np.load(os.path.join(p.DMN_mask_path,'yeo_DMN_IFG.npy'))

    diff_emb, df_quest = load_data(p,0,states_wanted,group,score_wanted)
    data = {}
    exp_mask = diff_emb.expertise['Experts']==1
    labels = p.networks_wanted
    stat_list = np.zeros((len(labels),boot_rep))

    for label_id, label in enumerate(p.networks_wanted):
        network_mask = yeo==(p.networks_labels.index(label)+1)
        current_network = diff_emb.emb[:,network_mask]
        current_network = (current_network - np.mean(current_network))/np.std(current_network)
        if interaction:
            current_network = current_network[diff_emb.states[states_wanted[0]]==1,:] - current_network[diff_emb.states[states_wanted[1]]==1,:]
            exp_mask = exp_mask[diff_emb.states[states_wanted[0]]==1]
            
        x,y = np.median(current_network[exp_mask,:],1), np.median(current_network[~exp_mask,:],1)
        print(label)
        stat_list[label_id,:],pval = two_sample_bootstrap(x,y,repetition=boot_rep)
        labels[label_id] += f'\n(p={np.round(pval,3)})'
        network_difference = difference(x,y)
        data[label] = [network_difference]
        
    N = len(p.networks_wanted)    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    df = pd.DataFrame(data)
    
    
    
    #Draw axes
    ax = plt.subplot(polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], p.networks_wanted, color='k', size=15)
    ax.spines['polar'].set_visible(False)
    
    #Draw Y labels
    ax.set_rlabel_position(0)
    
    #values for group constrast
    # plt.yticks([-0.3,0,0.3], ["-0.3","0","0.3"], color="k", size=15)
    # plt.ylim(-0.5,0.5)
    
    #values for interaction constrast
    plt.yticks([-0.7,0,0.3], ["-0.7","0","0.3"], color="k", size=15)
    plt.ylim(-0.9,0.9)
    
    
    
    values=df.values.flatten().tolist()
    values += values[:1]
    zeros = np.zeros_like(values)
    ax.plot(angles, zeros, color='tab:orange', linewidth=2, linestyle='solid')
    ax.plot(angles, values, color='tab:blue', linewidth=2, linestyle='solid')
    ax.fill(angles, values, color='tab:blue', alpha=0.2)
    
    
    return df

states_wanted = ['OpenPresence','RestingState']
group = 'all'
score_wanted = 'None'
interaction = False
df = radarchart(p)

print("--- %s seconds ---" % (time.time() - start_time))
