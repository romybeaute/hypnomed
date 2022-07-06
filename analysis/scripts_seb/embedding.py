#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 18:03:27 2021

@author: sebastien
"""
import numpy as np
import pandas as pd 

from nilearn import datasets, surface

from sklearn import neighbors
from scipy.io import loadmat

import sys
import os.path

class embedding:
    def __init__(self, data_path, emb_file, expertise_file, states_file, sub_file):
        self.data_path = data_path
        self.emb = loadmat(os.path.join(data_path, emb_file))['emb']
        self.expertise = pd.read_csv(os.path.join(data_path, expertise_file))
        self.states = pd.read_csv(os.path.join(data_path, states_file))
        self.sub = pd.read_csv(os.path.join(data_path, sub_file))
         
        self.fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
        coords_left, _ = surface.load_surf_mesh(self.fsaverage['infl_left'])
        coords_right, _ = surface.load_surf_mesh(self.fsaverage['infl_right'])
        self.coords = np.vstack([coords_left, coords_right])
        
    def remove_sub(self, sub_ids):
        #remove subs based on their ids
        # sub_ids needs to be a list or a tupple
        outliers_mask = ~self.sub['Subs_ID'].isin(sub_ids)
        self.expertise = self.expertise[outliers_mask]
        self.states = self.states[outliers_mask]
        self.sub = self.sub[outliers_mask]
        self.emb = self.emb[outliers_mask, :, :]

    def get_expertise(self, group):
        #keep group wanted 
        #group should be one of 'novices', 'experts' or 'all'
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
        #keep states wanted
        #states should be one of 'Compassion', 'OpenPresence' or 'RestingState'
        #states_wanted should be a list or a tupple
        
        states_mask = np.zeros(self.emb.shape[0])
        for states_id in range(len(states_wanted)):
            states_mask = states_mask + self.states[states_wanted[states_id]]
        self.states = self.states[states_wanted]
        self.expertise = self.expertise[states_mask != 0]
        self.states = self.states[states_mask != 0]
        self.sub = self.sub[states_mask != 0]
        self.emb = self.emb[states_mask != 0, :, :]

    def get_neighborhood(self, p):
        #generates spheres for searchlight analysis
        #a radius of 6. corresponds to ~41 voxels per sphere
        nn = neighbors.NearestNeighbors(radius=p.radius)
        self.adjacency = nn.fit(self.coords).radius_neighbors_graph(self.coords).tolil()
        
    def get_surf_emb(self,p):
        cortex = loadmat(os.path.join(p.data_path, 'cortex.mat'))
        cortex = cortex['cortex']
        surface_emb = np.zeros([self.emb.shape[0], self.coords.shape[0]])
        surface_emb[:, np.squeeze(cortex)] = np.squeeze(self.emb)
        self.emb = surface_emb
        return self