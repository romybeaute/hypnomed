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


def selected_embedding(condition,sublist,npy_folder):

    outcome_folder = '/home/romy.beaute/projects/hypnomed/diffusion_embedding/emb_matrices'
    
    # template = load_template('/mnt/data/romy/hypnomed/git/data/template/fsaverage')
    template = load_template('/mnt/data/romy/hypnomed/git/data/template')

    embeddings = []
    subs = []
    states = [state for state in condition.split('_')]
        
    

    for sub in sublist:
            subs.append(sub)
            for state in states:
                try:
                    # embeddings.append(np.load(data_folder+f'/embedding_dense_emb.{sub}.fa.npy'))
                    # embeddings.append(np.load(data_folder+f'/embedding_dense_emb.{sub}.om.npy'))
                    # embeddings.append(np.load(data_folder+f'/embedding_dense_emb.{sub}.rest.npy'))
                    embeddings.append(np.load(npy_folder+f'/embedding_dense_emb.{sub}.ses-001.{state}.npy'))
                    print(sub)
                    print(state)
                except:
                    print(sub)
                    print(state)


    realigned = run_realign(embeddings, template)
    for i in range(5):
        realigned = run_realign(realigned, np.asarray(np.mean(realigned, axis=0).squeeze()))

    
    if len(sublist)==1:
        path = outcome_folder+'/{}_{}_embedding.mat'.format(sub,condition)

    else:
        path = outcome_folder+'/group_{}_embedding.mat'.format(condition)
    
    savemat(path, mdict={'emb': realigned, 'subs': subs, 'states':states})
    f = loadmat(path)

    print('Group matrix succeded and saved in {}'.format(path))
    # print(len(f['subs']))
    # print(f['subs'])
    return f



def selected_embedding_reordered(condition,sublist):

    outcome_folder = '/home/romy.beaute/projects/hypnomed/diffusion_embedding/emb_matrices'
    npy_folder = '/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_outputs/emb_output_reordered'

    template = load_template('/mnt/data/romy/hypnomed/git/data/template')

    embeddings = []
    subs = []

    for sub in sublist:
            subs.append(sub)
            try:

                embeddings.append(np.load(npy_folder+f'/embedding_dense_emb.{sub}.ses-001.{condition}.npy'))
                print(sub)
                print(condition)
            except:
                print(sub)
                print(condition)


    realigned = run_realign(embeddings, template)
    for i in range(5):
        realigned = run_realign(realigned, np.asarray(np.mean(realigned, axis=0).squeeze()))


    if len(sublist)==1: #for single subject analysis
        path = outcome_folder+'/{}_{}_embedding.mat'.format(condition,sub)

    else: #for group analysis
        path = outcome_folder+'/{}_group_embedding.mat'.format(condition)
    
    savemat(path, mdict={'emb': realigned, 'subs': subs, 'condition':condition})
    f = loadmat(path)

    print('Group matrix succeded and saved in {}'.format(path))

    return f


