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
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from scipy.linalg import pinv
from scipy.io import loadmat

import sys
import os.path
import time

start_time = time.time()

class TransformedTargetB2B(RegressorMixin, BaseEstimator):

    def __init__(self, regressor, G, H, n_splits, metric='causal_factor'):
        self.regressor = regressor
        self.n_splits = n_splits
        self.G = G
        self.H = H
        self.metric = metric #can be either 'causal_factor' or 'pred'

    def fit(self, X, y, **fit_params):

        y_trans = self.transform(X, y)
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, y_trans, **fit_params)
        return self

    def predict(self, X):
        if self.metric == 'pred':
            check_is_fitted(self)
            pred = self.regressor_.predict(X)
            pred_trans = self.inverse_transform(pred)
    
            return pred_trans
        elif self.metric == 'causal_factor':
            return self.S
    
    def score(self,y_true,y_pred):
        if self.metric == 'pred':    
            return np.mean(~np.logical_xor(y_pred>0,y_true>0),0)
        elif self.metric == 'causal_factor':
            return self.S

    def transform(self, X, Y):
        ensemble = ShuffleSplit(n_splits=self.n_splits, test_size=.5)
        H_hats = list()
        for G_set, H_set in ensemble.split(X, Y):
            Y_hat = self.G.fit(X[G_set], Y[G_set]).predict(X)
            H_hat = self.H.fit(Y[H_set], Y_hat[H_set]).coef_
            H_hats.append(H_hat)
        self.S = np.mean(H_hats, 0)
        self.array = np.array(H_hats)
        return Y@self.S

    def inverse_transform(self, Y):
        return np.round(Y@pinv(self.S))

    @property
    def n_features_in_(self):
        # For consistency with other estimators we raise a AttributeError so
        # that hasattr() returns False the estimator isn't fitted.
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_features_in_ attribute."
                .format(self.__class__.__name__)
            ) from nfe

        return self.regressor_.n_features_in_


class embedding:
    def __init__(self, data_path, emb_file, expertise_file, states_file, sub_file):
        self.data_path = data_path
        self.emb = loadmat(os.path.join(data_path, emb_file))['emb']
        self.expertise = pd.read_csv(os.path.join(data_path, expertise_file))
        self.states = pd.read_csv(os.path.join(data_path, states_file))
        self.sub = pd.read_csv(os.path.join(data_path, sub_file))

    def remove_sub(self, sub_ids):
        # sub_ids needs to be a list or a tupple
        outliers_mask = ~self.sub['Subs_ID'].isin(sub_ids)
        self.expertise = self.expertise[outliers_mask]
        self.states = self.states[outliers_mask]
        self.sub = self.sub[outliers_mask]
        self.emb = self.emb[outliers_mask, :, :]

    def get_expertise(self, group):
        if group == 'novices':
            sub_ids = list(self.sub['Subs_ID'][self.expertise.Novices == 0])
            self.remove_sub(sub_ids)
        elif group == 'experts':
            sub_ids = list(self.sub['Subs_ID'][self.expertise.Experts == 0])
            self.remove_sub(sub_ids)
        elif group != 'all':
            sys.exit(
                'error in p.group, group not recognized ! Should be novices, experts or all')

    def get_states(self, states_wanted):
        states_mask = np.zeros(self.emb.shape[0])
        for states_id in range(len(states_wanted)):
            states_mask = states_mask + self.states[states_wanted[states_id]]
        self.states = self.states[states_wanted]
        self.expertise = self.expertise[states_mask != 0]
        self.states = self.states[states_mask != 0]
        self.sub = self.sub[states_mask != 0]
        self.emb = self.emb[states_mask != 0, :, :]

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
    diff_emb = embedding(data_path,
                         'co_om_rs_group_embedding_%s.mat' % p.data_file,
                         'expertise_covs.csv',
                         'state_covs.csv',
                         'subject_covs.csv')
    diff_emb.emb = diff_emb.emb[:, :, p.dimension]
    df_composite_score = pd.read_pickle(os.path.join(p.data_path,'questionnaires','composite_score.pkl'))

    if 'med' in [x.lower() for x in p.states_wanted]:
        # careful, hard coded here!!
        diff_emb.states.columns = ['Med', 'Med2', 'RestingState']
        diff_emb.states['Med'][diff_emb.states['Med2'] != 0] = 1
    diff_emb.get_states(p.states_wanted)
    diff_emb.get_expertise(p.group)
    
    commun_sub_id = np.array(list(set(np.unique(diff_emb.sub)) & set(df_composite_score['id'].to_numpy())))
    commun_sub_id = commun_sub_id[~np.in1d(commun_sub_id,p.outliers)]
    df_composite_score = df_composite_score[df_composite_score['id'].isin(commun_sub_id)]
    p.outliers = np.hstack((p.outliers,(np.unique(diff_emb.sub['Subs_ID'][~diff_emb.sub.isin(commun_sub_id).to_numpy()[:,0]]))))
    
    diff_emb.remove_sub(p.outliers)

    return diff_emb,df_composite_score


def create_labels(p, df_composite_score):
    # careful, hard coded here!!
    Y = list()
    for column in df_composite_score:
        if column not in ['id','group']:
            if len(p.states_wanted) > 1:
                dupplicated_list = list()
                for element in df_composite_score[column]:
                    for duplication_id in range(len(p.states_wanted)):
                        dupplicated_list.append(element)
                Y.append(dupplicated_list)
            else:
                Y.append(df_composite_score[column])
    Y = np.array(Y).T
    return Y


def randomundersample(X, Y, diff_emb):
    X_res, Y_res = np.copy(X), np.copy(Y)
    for label_id in range(Y.shape[1]):
        # checking which label should undersampled
        if np.sum(Y[:, label_id] == 0) > np.sum(Y[:, label_id] == 1):
            ltu = 0  # ltu = label_to_undersample
        elif np.sum(Y[:, label_id] == 0) < np.sum(Y[:, label_id] == 1):
            ltu = 1  # ltu = label_to_undersample
        else:
            ltu = None  # ltu = label_to_undersample

        if ltu != None:
            # undersampling has to be handled differently if it corresponds to expertise or states sadly :(
            if label_id == 0:  # undersampling expertise
                oversampled_subjects_id = np.unique(
                    diff_emb.sub[Y_res[:, label_id] == ltu])
                states_nb = diff_emb.states.shape[0]/75
                subjects_id_to_remove = np.random.choice(oversampled_subjects_id,
                                                         int(np.abs(
                                                             sum(Y_res[:, label_id] == 0)-sum(Y_res[:, label_id] == 1))/states_nb),
                                                         replace=False)
                mask_index_to_remove = ~np.in1d(
                    diff_emb.sub, subjects_id_to_remove)
                X_res, Y_res = X[mask_index_to_remove,
                                 :], Y[mask_index_to_remove, :]

            # undersampling states (could be improved by removing as many OP than COMP states)
            else:
                labels_index = np.array(range(len(Y_res)))
                ltu_index = labels_index[Y_res[:, label_id] == ltu]
                index_to_remove = np.random.choice(ltu_index,
                                                   np.abs(
                                                       sum(Y_res[:, label_id] == 0)-sum(Y_res[:, label_id] == 1)),
                                                   replace=False)
                mask_index_to_remove = ~np.in1d(labels_index, index_to_remove)
                X_res, Y_res = X_res[mask_index_to_remove,
                                     :], Y_res[mask_index_to_remove, :]

    return X_res, Y_res


def preprocess_data(X, Y, diff_emb):
    X, Y = randomundersample(X, Y, diff_emb)
    Y = StandardScaler().fit_transform(Y)
    return X, Y

def scorer(y_true,y_pred):
    return np.mean(~np.logical_xor(y_pred>0,y_true>0),0)


def get_sphere(brain_data,diff_emb,sphere_id):
    X = brain_data[:,diff_emb.adjacency.rows[sphere_id]]
    return X

def main(p):
    diff_emb,df_composite_score = load_data(p)
    diff_emb.get_surf_emb(p)
    diff_emb.get_neighborhood(p)
    
    brain_data = np.squeeze(diff_emb.emb)  # your data
    Y = create_labels(p, df_composite_score)  # your variables to decode
    G = RidgeCV(fit_intercept=False)  # it's important to set a linear regression without bias
    H = LinearRegression(fit_intercept=False)
    
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    b2b_est = TransformedTargetB2B(
                regressor=pipe, G=G, H=H, n_splits=p.n_splits)
    b2b_random = TransformedTargetB2B(
                regressor=pipe, G=G, H=H, n_splits=p.n_splits)
    
    voxels_nb = diff_emb.adjacency.shape[0]
    factors = np.zeros([p.repetitions_nb,voxels_nb,Y.shape[1]])
    factors_random = np.zeros([p.repetitions_nb,voxels_nb,Y.shape[1]])


    for sphere_id in range(voxels_nb):
        X = get_sphere(brain_data,diff_emb,sphere_id)
        if len(X.shape) == 3: #if there are more than 1 gradient, then stack them
            X = np.reshape(X,[X.shape[0],X.shape[1]*X.shape[2]],'F')
        for repetition_id in range(p.repetitions_nb):
            X_splitted,Y_splitted = preprocess_data(X,Y,diff_emb)
            Y_random = np.copy(Y_splitted).ravel()
            np.random.shuffle(Y_random)
            Y_random = Y_random.reshape(Y_splitted.shape)
            b2b_est.transform(X_splitted,Y_splitted)
            b2b_random.transform(X_splitted,Y_random)
            factors[repetition_id,sphere_id,:] = b2b_est.S.diagonal()
            factors_random[repetition_id,sphere_id,:] = b2b_random.S.diagonal()
           
                
    return diff_emb, factors, factors_random



# set up parameters
p = p()

p.data_file = 'new'
p.outliers = np.array([73])  # subject(s) id that you want to exclude from the analysis
p.dimension = [0]  # dimension(s) that you want to study
p.repetitions_nb = 1
p.n_splits = 50
p.radius = 6. #~41 voxels per sphere


p.analysis_framework = [
    #states #'Med','RestingState'  //  'Compassion', 'OpenPresence'
    # group that you want to study : 'novices','experts','all'
    {'states_wanted':['Compassion','OpenPresence','RestingState'],'group':'experts'},
    {'states_wanted':['Compassion','OpenPresence'],'group':'experts'},
    {'states_wanted':['RestingState'],'group':'experts'},
    {'states_wanted':['Compassion'],'group':'experts'},
    {'states_wanted':['OpenPresence'],'group':'experts'},

    {'states_wanted':['Compassion','OpenPresence','RestingState'],'group':'novices'},
    {'states_wanted':['Compassion','OpenPresence'],'group':'novices'},
    {'states_wanted':['RestingState'],'group':'novices'},
    {'states_wanted':['Compassion'],'group':'novices'},
    {'states_wanted':['OpenPresence'],'group':'novices'},
    
    {'states_wanted':['Compassion','OpenPresence','RestingState'],'group':'all'},
    {'states_wanted':['Compassion','OpenPresence'],'group':'all'},
    {'states_wanted':['RestingState'],'group':'all'},
    {'states_wanted':['Compassion'],'group':'all'},
    {'states_wanted':['OpenPresence'],'group':'all'},
    
    ]

df = {'states_wanted':[],'group':[],'factors':[],'factors_random':[]}




cluster = False
if cluster:
    # # careful on cluster, use path from root !!!!
    job_id = sys.argv[1] #see associated batch file
    p.data_path = '/mnt/data/sebastien/data'
    p.results_path = '/mnt/data/sebastien/results/b2b_emb_quest_searchlight'
else:
    job_id = '' 
    p.data_path = 'data'
    p.results_path = 'results/b2b_emb_quest_searchlight'
    

for analysis in p.analysis_framework:
    p.states_wanted = analysis['states_wanted']
    p.group = analysis['group']
    
    
    diff_emb, factors, factors_random = main(p)
    df['states_wanted'].append(p.states_wanted)
    df['group'].append(p.group)
    df['factors'].append(factors)
    df['factors_random'].append(factors_random)

    
df = pd.DataFrame(df)
df.to_pickle(os.path.join(p.results_path,f'decoding_results_{job_id}.pkl'),compression='zip')


# print("--- %s seconds ---" % (time.time() - start_time))

