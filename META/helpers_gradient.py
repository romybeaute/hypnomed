"""
STEP 3 : Rotate and gather all subjects into a matrix
NB : A single file with all individual embeddings is created that can be read into another software for group-level analyses.
"""

import nibabel as nib
import numpy as np
import os, glob
import pandas as pd
from scipy.io import savemat
from scipy.io import loadmat
from nilearn import surface
import os,sys
from scipy.io import loadmat



def run_realign(emb, tar):
    realign = []
    for i, embed in enumerate(emb):
        u, s, v = np.linalg.svd(tar.T.dot(embed), full_matrices=False)
        xfm = v.T.dot(u.T)
        realign.append(embed.dot(xfm))
    return realign


def load_template(template_path):
    cortex = loadmat('/mnt/data/romy/hypnomed/git/data/cortex.mat')
    cortex = np.squeeze(cortex['cortex'])
    template = np.zeros([20484,5])
    for grad_id in range(5):
        template_L = surface.load_surf_data(os.path.join(template_path,'hcp.embed.grad_'+str(grad_id+1)+'.L.fsa5.func.gii'))
        template_R = surface.load_surf_data(os.path.join(template_path,'hcp.embed.grad_'+str(grad_id+1)+'.R.fsa5.func.gii'))
        template[:,grad_id] = np.hstack([template_L,template_R])
    template = template[cortex,:]
    template = template - np.mean(template,0)
    template = template/np.std(template,0)
    return template




def emb_path(condition,emb_outpath):
  return os.path.join(emb_outpath,'group_{}_embedding.mat'.format(condition))


def load_embmat(emb_path,show_infos=True):
  '''
  Return the principal gradient of the .mat file localised in emb_path
  '''
  b = loadmat(file_name=emb_path) #.mat file
  if show_infos:
    print(' - shape embedding (n_subjects, n_voxels, n_dims): {}\n - n = {} subjects\n - condition : {}\n - path : {}\n'.format(b['emb'].shape,len(b['subs']),b['states'],emb_path))
  return b,b['emb'][:,:,0]


