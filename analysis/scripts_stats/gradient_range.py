#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

import numpy as np
import os,sys
sys.path.append('/home/romy.beaute/projects/hypnomed/analysis/scripts_stats/')
from embedding import embedding
from misc_functions import *
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns 


def difference(x,y):
    return (np.mean(x)-np.mean(y))
    
    
def std_diff(x,y):
    return (np.std(x)-np.std(y))


def two_sample_bootstrap_mean(x,y,repetition=10000,alpha=0.05):
    
    x = np.array(x)
    y = np.array(y)
    
    statistic = difference
    test_stat, _ = statistic(x,y)
    x_h0 = x - np.mean(x) 
    y_y0 = y - np.mean(y) 
    
    stat_list = np.zeros(repetition)
    stat_list_h0 = np.zeros(repetition)
    
    for rep_id in range(repetition):
        current_x = list(np.random.choice(x,x.shape[0]))
        current_y = list(np.random.choice(y,y.shape[0]))
        current_stat,_ = statistic(current_x,current_y)
        stat_list[rep_id] = current_stat

        
        current_x_h0 = list(np.random.choice(x_h0,x_h0.shape[0]))
        current_y_h0 = list(np.random.choice(y_y0,y_y0.shape[0]))
        current_stat_h0,_ = statistic(current_x_h0,current_y_h0)
        stat_list_h0[rep_id] = current_stat_h0


    CI = f'{str(np.quantile(stat_list,alpha/2).round(2))}, {str(np.quantile(stat_list,1-alpha/2).round(2))}'
    
    if test_stat<0:
        test_stat = test_stat*-1
    pval = (np.sum(np.abs(stat_list_h0)>=test_stat))/(repetition)

    return test_stat ,pval, CI


def two_sample_bootstrap_std(x,y,repetition=10000):
    rng = np.random.default_rng()
    x = np.array(x)
    y = np.array(y)

    test_stat = np.abs(std_diff(x,y))

    stat_list = np.zeros(repetition)

    for rep_id in range(repetition):
        current_x = rng.choice(np.vstack((x,y)),x.shape[0],axis=0)
        current_y = rng.choice(np.vstack((x,y)),y.shape[0],axis=0)
        current_stat = std_diff(current_x,current_y) #stat value for each repetition

        stat_list[rep_id] = current_stat #len(stat_list)=repetition


    pval = (np.sum(np.abs(stat_list)>=test_stat))/(repetition)
    print('pval : {}    (n={} repetitions)'.format(pval,repetition))

    return stat_list,pval



class p:
    pass


p.data_path = '/home/romy.beaute/projects/hypnomed/analysis/scripts_stats/data'
p.results_path = '/home/romy.beaute/projects/hypnomed/analysis/results'
p.surf_path = '/home/romy.beaute/projects/hypnomed/fsaverage5'
p.DMN_mask_path = '/home/romy.beaute/projects/hypnomed/data/DMN_mask'


p.outliers = [27,32]

p.radius = 1

p.networks_labels = ['Vis', 'SM', 'DA', 'VA','Lim', 'FP','MTC','IFG','AG','MPFC','PMC']
p.networks_wanted = ['Vis', 'SM', 'DA', 'VA','Lim', 'FP','MTC','IFG','AG','MPFC','PMC']
p.order = np.array(['Vis', 'SM', 'DA', 'VA','FP','Lim','MTC','IFG','AG','PMC','MPFC'])


# states_wanted = ['control','meditation']
# states_wanted = ['control','hypnose']
# states_wanted = ['meditation','hypnose']
# group = 'all'
# score_wanted = 'DDS'


comparisons = [['control','meditation'],['control','hypnose'],['meditation','hypnose']]

for states_wanted in comparisons :
    
    
    diff_emb = load_data(p,0,states_wanted)
    diff_emb.emb = diff_emb.emb[:,diff_emb.emb[0,:]!=0] #shape(76,18715) : without white matter
    n_embs = diff_emb.emb.shape[0] #nb subjetcs * 2 (for each condition)
    n_subjs_in_group = n_embs//2 #nb of subjects in each diff emb gradient

    #define gradients based on condition
    x = diff_emb.emb[:n_subjs_in_group] #gradients for first condition (eg control)
    y = diff_emb.emb[n_subjs_in_group:] #gradients for second condition (eg meditation)


    sns.histplot((np.mean(x,0), np.mean(y,0)))

    fig, ax = plt.subplots(figsize=(15, 7))
    repetitions = 10000
    stat_list,pval = two_sample_bootstrap_std(x,y,repetitions)
    print(pval) #1.32e-01 (for control vs meditation)

    #get variance for condition 1 and condition 2
    x,y = np.var(x,1), np.var(y,1)

    data = {}

    data = {'data':np.hstack((x,y)),'conditions':np.hstack(([states_wanted[0]]*len(x),[states_wanted[1]]*len(y)))}
    df = pd.DataFrame(data)
    sns.violinplot(x="conditions", y="data", data=df)
    ax.set_ylabel('Principal Gradient STD\n (pval = {}, n={} repetitions)'.format(pval,repetitions))
    plt.savefig('/home/romy.beaute/projects/hypnomed/analysis/scripts_stats/figures/violonplots/PG_std_{}_vs_{}.png'.format(states_wanted[0],states_wanted[1]))
