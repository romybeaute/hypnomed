# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd

from nilearn import datasets, surface

from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, KFold

from scipy.io import loadmat

import sys
import os.path
import time

start_time = time.time()



# self = diff_emb

class embedding:
    def __init__(self, data_path, emb_file, expertise_file, states_file, sub_file):
        self.data_path = data_path
        self.emb = loadmat(os.path.join(data_path, emb_file))['emb']
        self.expertise = pd.read_csv(os.path.join(data_path, expertise_file)) #informs whether each condition for each subject corresponds to G1 or G1 
        self.states = pd.read_csv(os.path.join(data_path, states_file))
        self.sub = pd.read_csv(os.path.join(data_path, sub_file)) #Subs_ID : [1 1 1 2 2 2 ... 32 32 ... 40 40 40]

    def remove_sub(self, sub_ids):
        #Define the outliers from the group we want to remove + to remove p.outliers (=[15,27,32,40])) in a later step (outside the p def)
        # sub_ids needs to be a list or a tupple
        outliers_mask = ~self.sub['Subs_ID'].isin(sub_ids) #output True for subjects corresponding to the group we want to keep (create mask with them)
        self.expertise = self.expertise[outliers_mask] #self.expertise.shape = (n_emb (ie 3* each subj), value for each Group (ie either 0 or 1))
        self.states = self.states[outliers_mask] #pareil, on garde que les valeurs des states pour les sujets du group of interest
        self.sub = self.sub[outliers_mask] #Subs_ID : place each subject (n conditions times) in the list (eg [1 1 1,2 2 2,...,40 40 40])
        self.emb = self.emb[outliers_mask, :, :] #self.emb.shape = (119,18715,5) --> we apply the mask on the first dim, ie the subjects to keep only the embeddings of subs of interest

    def get_expertise(self, group):
        if group == 'G1': #exclude G2 subjects 
            sub_ids = list(self.sub['Subs_ID'][self.expertise.G1 == 0]) #list subjects dont G1 == 0 (to exclude)
            self.remove_sub(sub_ids) #call sub_ids to remove subjects for which value G1 == 0 (ie exclude subjects from G2)
        elif group == 'G2': #exclude G1 subjects 
            sub_ids = list(self.sub['Subs_ID'][self.expertise.G2 == 0])
            self.remove_sub(sub_ids) 
        elif group != 'all':
            sys.exit(
                'error in p.group, group not recognized ! Should be G1, G2 or all')

    def get_states(self, states_wanted):
        states_mask = np.zeros(self.emb.shape[0]) #revient à faire np.zeros(n_subjects)
        for states_id in range(len(states_wanted)): # iterate index for states wanted (eg if states_wanted = ['meditation','hypnose'] --> indx med == 0)
            states_mask = states_mask + self.states[states_wanted[states_id]] #pour chaque sujet defini mask si correspond à un state in state_wanted
        self.states = self.states[states_wanted]
        self.expertise = self.expertise[states_mask != 0] #rassemble le mask expertise et mask states : donne une liste de sujets 
        self.states = self.states[states_mask != 0] #garde seulement states of interest
        self.sub = self.sub[states_mask != 0]
        self.emb = self.emb[states_mask != 0, :, :] #redefini l'embedding avec les conditions d'intérêt

    def get_neighborhood(self, p):
        self.fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
        coords_left, _ = surface.load_surf_mesh(self.fsaverage['infl_left'])
        coords_right, _ = surface.load_surf_mesh(self.fsaverage['infl_right'])
        # same order than in pipeline
        self.coords = np.vstack([coords_left, coords_right])
        nn = neighbors.NearestNeighbors(radius=p.radius)
        self.adjacency = nn.fit(self.coords).radius_neighbors_graph(self.coords).tolil()

    def get_surf_emb(self,p):
        cortex = loadmat(os.path.join(p.data_path, 'cortex.mat'))
        cortex = cortex['cortex']
        surface_emb = np.zeros([self.emb.shape[0], 20484,len(p.dimension)])  # 20484 = nb of voxels for fsaverage5
        surface_emb[:, np.squeeze(cortex),:] = self.emb
        self.emb = surface_emb
        return self


class p:
    pass


def load_data(p):
    data_path = p.data_path
    diff_emb = embedding(data_path, #data_path
                         'group_{}_embedding.mat'.format(p.data_file), #emb_file
                         'expertise_covs.csv', #expertise_file (group in our case)
                         'state_covs.csv', #states_file
                         'subject_covs.csv') #subfile
    diff_emb.emb = diff_emb.emb[:, :, p.dimension] #keep only first dimension, corresp. to principal gradient (ie index 0)
    diff_emb.get_states(p.states_wanted)
    diff_emb.remove_sub(p.outliers)
    diff_emb.get_expertise(p.group)
    return diff_emb


def create_labels(p, diff_emb):
    # careful, hard coded here!!
    if p.analysis == 'states':
        states_labels = diff_emb.states.to_numpy()[:, 0] #state_label pour la première condition de chaque sujet
        Y = states_labels.T
    elif p.analysis == 'blocks': #prend que les sujets qui ont eu expertise (groupe) d'intérêt
        blocks_labels = diff_emb.expertise.G1.to_numpy()
        Y = blocks_labels.T
    return Y


def randomundersample(X, Y, diff_emb,p):
    X_res, Y_res = np.copy(X), np.copy(Y)
    # checking which label should undersampled
    if np.sum(Y == 0) > np.sum(Y== 1):
        ltu = 0  # ltu = label_to_undersample
    elif np.sum(Y == 0) < np.sum(Y== 1):
        ltu = 1  # ltu = label_to_undersample
    else:
        ltu = None  # ltu = label_to_undersample
    subj_id_res = diff_emb.sub
    if ltu != None:
        # if p.analysis == 'expertise':
        #     # undersampling has to be handled differently if it corresponds to expertise or states sadly :(
        #     oversampled_subjects_id = np.unique(
        #         diff_emb.sub[Y_res== ltu])
        #     states_nb = diff_emb.states.shape[0]/75
        #     subjects_id_to_remove = np.random.choice(oversampled_subjects_id,
        #                                              int(np.abs(
        #                                                  sum(Y_res== 0)-sum(Y_res == 1))/states_nb),
        #                                              replace=False)
        #     mask_index_to_remove = ~np.in1d(
        #         diff_emb.sub, subjects_id_to_remove)
        #     X_res, Y_res = X[mask_index_to_remove,
        #                      :], Y[mask_index_to_remove]
        #     subj_id_res = diff_emb.sub[mask_index_to_remove]
        if p.analysis == 'states':

            # undersampling states (could be improved by removing as many OP than COMP states)
            labels_index = np.array(range(len(Y_res)))
            ltu_index = labels_index[Y_res == ltu]
            index_to_remove = np.random.choice(ltu_index,
                                                np.abs(
                                                    sum(Y_res== 0)-sum(Y_res == 1)),
                                                replace=False)
            mask_index_to_remove = ~np.in1d(labels_index, index_to_remove)
            X_res, Y_res = X_res[mask_index_to_remove,
                                  :], Y_res[mask_index_to_remove]
            subj_id_res = diff_emb.sub[mask_index_to_remove]

    return X_res, Y_res, subj_id_res


def preprocess_data(X, Y, diff_emb,p):
    X, Y, subj_id_res = randomundersample(X, Y, diff_emb,p)
    X = StandardScaler().fit_transform(X)

    return X, Y, subj_id_res

def scorer(y_true,y_pred):
    return np.mean(~np.logical_xor(y_pred>0,y_true>0),0)

def get_destrieux():
    destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
    destrieux_atlas.map_right += len(destrieux_atlas.labels)-1
    destrieux_atlas.map = np.hstack([destrieux_atlas.map_left,destrieux_atlas.map_right])
    destrieux_atlas.roi_ids = np.unique(destrieux_atlas.map)
    return destrieux_atlas

def get_ROI(X,atlas,roi_id):
    roi = X[:,atlas.map==roi_id]
    return roi

def main(p):
    diff_emb = load_data(p)
    diff_emb.get_surf_emb(p)
    print('Loaded emb shape : {} (n={} subjects)'.format(diff_emb.emb.shape,diff_emb.emb.shape[0]))
    X = np.squeeze(diff_emb.emb)  # your data
    print('X shape : ',X.shape)
    Y = create_labels(p, diff_emb)  # your variables to decode
    print('Y shape : ',X.shape)
    scores = np.zeros([p.repetitions_nb,p.n_splits]) 
    coefs = list()
    # scores_random = np.zeros([p.repetitions_nb,p.n_splits])

    clf_params = { 'l1_ratio':np.linspace(0.05,0.95,5)}
    sgdc = SGDClassifier(loss='hinge',alpha=0.5,
                         penalty='elasticnet',fit_intercept=False,n_jobs=1)#ElasticNetCV(fit_intercept=False,cv=cv)

    clf = GridSearchCV(sgdc, clf_params, scoring='accuracy', cv=p.n_splits-1, verbose=1, n_jobs=1)

    kf = KFold(n_splits=p.n_splits, shuffle=True)
    if len(X.shape) == 3: #if there are more than 1 gradient, then stack them
        X = np.reshape(X,[X.shape[0],X.shape[1]*X.shape[2]],'F')
    for repetition_id in range(p.repetitions_nb):
        X_splitted,Y_splitted,subj_id_splitted = preprocess_data(X,Y,diff_emb,p)
        for kf_id,[train_index, test_index] in enumerate(kf.split(X_splitted,groups=subj_id_splitted)):
        
            clf.fit(X_splitted[train_index],Y_splitted[train_index])
            scores[repetition_id,kf_id] = clf.score(X_splitted[test_index],Y_splitted[test_index])
            coefs.append(clf.best_estimator_.coef_)
        # Y_random = np.copy(Y_splitted).ravel()
        # np.random.shuffle(Y_random)
        # Y_random = Y_random.reshape(Y_splitted.shape)
        
        # nested_score_random = cross_val_score(clf, X=X_splitted, y=Y_random, cv=5)
        # scores_random[repetition_id] = nested_score_random.mean()
    return scores, np.array(coefs)



# set up parameters
p = p()

cluster = False
if cluster:
    # # careful on cluster, use path from root !!!!
    job_id = sys.argv[1] #see associated batch file
    p.data_path = '/mnt/data/romy/hypnomed/git/analysis/scripts_stats/data'
    p.results_path = '/mnt/data/romy/hypnomed/git/analysis/pheno_stats/results/emb_decoding'
else:
    job_id = '' 
    p.data_path = '/home/romy.beaute/projects/hypnomed/analysis/scripts_stats/data'
    p.results_path = '/home/romy.beaute/projects/hypnomed/analysis/pheno_stats/results/emb_decoding'



if p.analysis == 'states':
    p.data_file = 'control_meditation_hypnose' #states analysis
else: 
    p.data_file = 'run-1_run-2_run-3' #block analysis 

    

p.outliers = [15,27,32,40]  # subject(s) id that you want to exclude from the analysis
p.dimension = [0]  # dimension(s) that you want to study (keep only principal gradient, ie dim 0)
p.repetitions_nb = 1
p.n_splits = 5

# p.states_wanted : states that you want to study : 'Med','RestingState' , 'Compassion', 'OpenPresence'
# p.group : 'experts'  # group that you want to study : 'novices','experts','all'
# p.analysis : 'states', 'expertise'

p.analysis_framework = [
    ###states analysis : group that you want to study : 'G1','G2','all'
    
    {'states_wanted':['control','meditation','hypnose'],'group':'G1','analysis':'states'},
    {'states_wanted':['meditation','hypnose'],'group':'G1','analysis':'states'},
    # {'states_wanted':['control'],'group':'G1','analysis':'states'},
    # {'states_wanted':['meditation'],'group':'G1','analysis':'states'},
    # {'states_wanted':['hypnose'],'group':'G1','analysis':'states'},

    {'states_wanted':['control','meditation','hypnose'],'group':'G2','analysis':'states'},
    {'states_wanted':['meditation','hypnose'],'group':'G2','analysis':'states'},
    # {'states_wanted':['control'],'group':'G2'},
    # {'states_wanted':['meditation'],'group':'G2'},
    # {'states_wanted':['hypnose'],'group':'G2'},
    
    {'states_wanted':['control','meditation','hypnose'],'group':'all','analysis':'states'},
    {'states_wanted':['meditation','hypnose'],'group':'all','analysis':'states'}
    # {'states_wanted':['control'],'group':'all'},
    # {'states_wanted':['meditation'],'group':'all'},
    # {'states_wanted':['hypnose'],'group':'all'}
    ]

df = {'states_wanted':[],'group':[],'analysis':[],'scores':[],'coefs':[]}

for analysis in p.analysis_framework:
    p.states_wanted = analysis['states_wanted']
    p.group = analysis['group']
    p.analysis = analysis['analysis']
    
    
    scores, coefs = main(p)
    print('mean score :',np.mean(scores))

    df['states_wanted'].append(p.states_wanted)
    df['group'].append(p.group)
    df['analysis'].append(p.analysis)
    df['scores'].append(scores) #score for each split (n=5)
    df['coefs'].append(coefs)

    
df = pd.DataFrame(df)
print(df)
df.to_pickle(os.path.join(p.results_path,f'decoding_results_{job_id}.pkl'))

print("--- %s seconds ---" % (time.time() - start_time))


df_embres = pd.read_pickle(os.path.join(p.results_path,f'decoding_results_{job_id}.pkl'))
