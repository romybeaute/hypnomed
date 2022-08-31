#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:51:03 2022

@author: sebastien
@modification: romy
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
import statsmodels.api as sm
from statsmodels.formula.api import ols
from embedding import embedding
from scipy.stats import ttest_ind

# def load_pheno_data(states_wanted): #load phenomenological data harvested during fMRI scans
#     states_correspondance = {'RestingState':'Rs_',
#                              'Meditation':'Med_',
#                              'Hypnosis':'Hyp_'} #maps the states correspondance with the .csv files
#     for state_id, state in enumerate(states_wanted):

#         for quest_id in range(len(p.quest_names)):
#             current_df_quest = pd.read_csv(os.path.join(p.data_path,'questionnaires',p.quest_names [quest_id]+'.csv'),sep=';')

#             if quest_id == 0:
#                 current_df_quest_state = current_df_quest[['Subject','Expertise']].copy()
#                 current_df_quest_state = current_df_quest_state.rename(columns={"Subject": "id", "Expertise":"group"})
#                 current_df_quest_state['state'] = [state]*current_df_quest_state.shape[0]

#             quest_column = states_correspondance[state]+p.quest_names[quest_id].capitalize()
#             current_df_quest_state[p.quest_names[quest_id]] = current_df_quest[quest_column]
#         if state_id ==0:
#             df_quest = current_df_quest_state.copy()
            
#         else:
#             df_quest = df_quest.append(current_df_quest_state)
            
#     return df_quest

# def load_scores(p,score_wanted,states_wanted):
    
#     if score_wanted == 'pheno':
#         df_quest = load_pheno_data(states_wanted)
        
#     elif score_wanted == 'composite':
#         df_quest = pd.read_pickle(os.path.join(p.data_path,'questionnaires','composite_score.pkl'))
        
#     elif score_wanted == 'MAIA':
#         df_quest = pd.read_csv(os.path.join(p.data_path,'questionnaires','QUEST_scores_MAIA_2020-02-29.csv'),sep=',')
#         df_quest = df_quest[np.isnan(df_quest['NOT'])==False]
        
#     elif score_wanted == 'FFMQ':
#         df_quest = pd.read_csv(os.path.join(p.data_path,'questionnaires','QUEST_scores_FFMQ_2020-02-29.csv'),sep='\t')
#         df_quest = df_quest[np.isnan(df_quest['OBS'])==False]
        
#     elif score_wanted == 'DDS':
#         df_quest = pd.read_csv(os.path.join(p.data_path,'questionnaires','QUEST_scores_2018-12-03.tsv'),sep='\t')
#         df_quest = df_quest[np.isnan(df_quest['DDS'])==False]
#     elif score_wanted == 'HOURS':
#         df_quest = pd.read_csv(os.path.join(p.data_path,'questionnaires','QUEST_scores_2018-12-03.tsv'),sep='\t')
#         df_quest = df_quest[np.isnan(df_quest['HOURS'])==False]
#     elif score_wanted == "PAIN":
#         df_quest = pd.read_csv(os.path.join(p.data_path,'questionnaires','PAR_defusion_data.csv'), sep=',')
#         df_quest = df_quest[~np.isnan(df_quest['int'])]

#     elif score_wanted == 'None':
#         df_quest = pd.read_csv(os.path.join(p.data_path,'questionnaires','QUEST_scores_2018-12-03.tsv'),sep='\t')
#     else:
        
#         print('score_wanted should be either "pheno", "composite", "MAIA" or "FFMQ"')
        
#     return df_quest


''''''
states = ['control','hypnose','meditation']
blocks = ['run-1','run-2','run-3']
emb_mat_path  = '/home/romy.beaute/projects/hypnomed/diffusion_embedding/emb_matrices/group'
''''''



def load_data(p,dimension,states_wanted):
    #load diffusion mebedding data and remove unwanted subjects or missing data
    diff_emb = embedding(p.data_path,
                         'group_control_meditation_hypnose_embedding.mat',
                         'state_covs.csv',
                         'subject_covs.csv')
    
    # if 'Med_mean' in states_wanted:
    #     med_mean = np.mean([diff_emb.emb[diff_emb.states['OpenPresence']==1],diff_emb.emb[diff_emb.states['Compassion']==1]] ,0)
    #     diff_emb.states = diff_emb.states.rename(columns={'Compassion':'Med_mean'})
    #     diff_emb.emb[diff_emb.states['Med_mean']==1] = med_mean
    # elif 'med' in [x.lower() for x in states_wanted]:
    #     # careful, hard coded here!!
    #     diff_emb.states.columns = ['Med', 'Med2', 'RestingState']
    #     diff_emb.states['Med'][diff_emb.states['Med2'] != 0] = 1
        
        
    diff_emb.get_states(states_wanted)
    # diff_emb.get_expertise(group)

    # df_quest = load_scores(p,score_wanted,states_wanted)
    commun_sub_id = np.array(list(set(np.unique(diff_emb.sub))))
    commun_sub_id = commun_sub_id[~np.in1d(commun_sub_id,p.outliers)]
    # df_quest = df_quest[df_quest['id'].isin(commun_sub_id)]
    outliers = np.hstack((p.outliers,(np.unique(diff_emb.sub['Subs_ID'][~diff_emb.sub.isin(commun_sub_id).to_numpy()[:,0]]))))

    diff_emb.remove_sub(outliers)
    diff_emb.emb = diff_emb.emb[:,:,dimension]

    #get diff emb data into fsaverage5 rather than just the cortex; necessary to work with atlases
    diff_emb.get_neighborhood(p)
    diff_emb.get_surf_emb(p)
    # df_quest = df_quest.sort_values('id')
    return diff_emb

# def quest_stat_analysis(df_quest):
#     #perform a 2-way ANOVA then t-tests 
#     data = list()
#     subscore = list()
#     group = list()
    
#     sub_nb = df_quest['id'].shape[0]
#     for quest in df_quest.columns:
#         if quest not in ['group','id']:
#             data += list(df_quest[quest].to_numpy())
#             subscore += [quest]*sub_nb
#             group += list(df_quest['group'].to_numpy())
            
#     test_df = pd.DataFrame({'data':data,'subscore':subscore,'group':group})
#     model = ols('data ~ C(subscore) + C(group) + C(subscore):C(group)', data=test_df).fit()
#     print('Anova results:')
#     print(sm.stats.anova_lm(model, typ=2))
#     print('\n t-test results: \n')
#     for quest in df_quest.columns:
#         if quest not in ['group','id']:
#             current_quest = df_quest[quest].to_numpy()
#             tval,pval = stats.ttest_ind(current_quest[df_quest['group']=='exp'],
#                             current_quest[df_quest['group']=='nov'])
#             diff = np.mean(current_quest[df_quest['group']=='exp']) - np.mean(current_quest[df_quest['group']=='nov'])
                            
#             print(quest,diff,pval)

            
            
def duplicate_elements(old_list,n):
    #duplicate elements in a list such that 
    #old list = [1,2] => new_list = [1,1,1,2,2,2] if n=3
    new_list = list()
    for elements in old_list:
        for duplicate_id in range(n):
            new_list.append(elements)
    return np.array(new_list)


def difference(x,y):
    return (np.mean(x)-np.mean(y))
    
def std_diff(x,y):
    return (np.std(x)-np.std(y))

def two_sample_bootstrap(x,y,repetition=10000):
    rng = np.random.default_rng()
    x = np.array(x)
    y = np.array(y)
      # statistic = partial(stats.mannwhitneyu,alternative='two-sided') #the stats to use based on H0 (involves the mean)
    test_stat = std_diff(x,y)
    # x = x - np.mean(x) 
    # y = y - np.mean(y) 
    x = x/np.std(x)
    y= y/np.std(y)
    stat_list = np.zeros(repetition)
    # se = np.zeros(repetition)
    for rep_id in range(repetition):
        current_x = rng.choice(x,x.shape[0])
        current_y = rng.choice(y,y.shape[0])
        current_stat = std_diff(current_x,current_y)
        # se[rep_id] = np.sqrt(np.var(current_x)/current_x.shape[0]+np.var(current_y)/current_y.shape[0])
        stat_list[rep_id] = current_stat

    # t_statistics = (stat_list - test_stat) / se
    # resample_sd = np.std(stat_list)
    # lower, upper = np.percentile(t_statistics, [2.5, 97.5])
    # CI = (test_stat - resample_sd * upper,
    #         test_stat - resample_sd * lower)
    # if not ((0<CI[1]) & (0>CI[0])):
    #     print(CI)
    test_stat = np.abs(test_stat)
    pval = (np.sum(np.abs(stat_list)>=test_stat))/(repetition)

    return stat_list,pval


def pairwise_partial_correlation(x, y, z):
    #perform pairwise correlation between n conditions of m samples
    # x and y should be of shape (n,m)
    
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    
    condition_nb = x.shape[0]
    rho_list = list()
    pval_list = list()

    for condition_id in range(condition_nb):
        df = pd.DataFrame({'x':x[condition_id,:],'y':y[condition_id,:],'z':z[condition_id,:]})
        results = pg.partial_corr(data=df,x='x',y='y',covar='z',method='spearman')
        rho_current, pval_current = results['r'],results['p-val']
        
        rho_list.append(rho_current)
        pval_list.append(pval_current)
        
    return np.array(rho_list), np.array(pval_list)

def p_adjust_partial_cor(x,y,z,permutation_nb=10000):
    #use bootstrap to adjust p-vlues using westfall-young method
    #Westfall et al. - 1993 - On Adjusting P-Values for Multiplicity
    
    # x and y should be of shape (n,m) 
    # where n is the number of conditions and m the number of samples
    
    #stat can only be correlation for now 
    #to make it work for stat difference (mean e.g.) you would need to permute smples between x and y 
    condition_nb = x.shape[0]
    sample_nb = x.shape[1]
    stat_val,_ = pairwise_partial_correlation(x,y,z)

    stat_val = np.squeeze(stat_val)
    ranks = np.argsort(np.abs(stat_val))[::-1]
    counts = np.zeros((permutation_nb, condition_nb))
    
    
    for perm_id in range(permutation_nb):
        
         
        u = np.zeros(condition_nb)
        sidx = np.random.permutation(sample_nb)
        x_resamp = x[:, sidx]
        
        stat_boot,_ = pairwise_partial_correlation(x_resamp,y,z)
        
        u[condition_nb-1] = np.abs(stat_boot[ranks[condition_nb-1]])
        for cond_id in range(condition_nb-2, -1, -1):
            u[cond_id] = max(u[cond_id+1], np.abs(stat_boot[ranks[cond_id]]))
        counts[perm_id,:] = (u >= np.abs(stat_val[ranks]))
        
        
    p_adjusted = np.sum(counts, axis=0)/permutation_nb
    for i in range(1, condition_nb):
        p_adjusted[i] = max(p_adjusted[i],p_adjusted[i-1])
    return p_adjusted[np.argsort(ranks)]