"""
STEP 3 : Rotate and gather all subjects into a matrix


Build gradient across groups --> to compare it to each other 
/!\ During dimensionality reduction steps the eigenvectors that we find might describe different subspaces.
==> Need to align the gradients after decomposition (function_realign).

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
    """
    Align the gradients after dimmensionality reduction (for group analyses)
    After realignement : gradients should be in the same space (then use realigned gradient for further analyses)
    """
    realign = []
    for i, embed in enumerate(emb):
        u, s, v = np.linalg.svd(tar.T.dot(embed), full_matrices=False)
        xfm = v.T.dot(u.T)
        realign.append(embed.dot(xfm))
    return realign


def load_template(template_path):
    """
    Load gradient template from HCP ressources 
    Templates can be found here : https://github.com/romybeaute/hypnomed/tree/main/data/template
    Load a total of 10 templates (R/L * 5 gradients)
    Eg : hcp.embed.grad_1.L.fsa5.func.gii
    """
    
    
    cortex = loadmat('/mnt/data/romy/hypnomed/git/data/cortex.mat')
    cortex = np.squeeze(cortex['cortex'])
    template = np.zeros([20484,5])
    for grad_id in range(5): #load a template for each diffusion map dimension
        template_L = surface.load_surf_data(os.path.join(template_path,'hcp.embed.grad_'+str(grad_id+1)+'.L.fsa5.func.gii')) #gradient template for Left Hemisphere
        template_R = surface.load_surf_data(os.path.join(template_path,'hcp.embed.grad_'+str(grad_id+1)+'.R.fsa5.func.gii')) ##gradient template for Right Hemisphere
        template[:,grad_id] = np.hstack([template_L,template_R]) #stack (right/left) arrays in sequence horizontally (column wise)
    template = template[cortex,:]
    template = template - np.mean(template,0)
    template = template/np.std(template,0)
    return template


def selected_embedding(condition,sublist,gradients_for):

    outcome_folder = '/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_matrices'
    npy_folder = '/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_outputs/emb_output_{}'.format(gradients_for)
    
    template = load_template('/mnt/data/romy/hypnomed/git/data/template') #hcp template for gradients

    embeddings = []
    subs = []

    states = [state for state in condition.split('_')] 
    if gradients_for == 'blocks': #'run-1,run-2,run-3' : need to change the name to retrieve npy file (by adding 'rs_' prefix)
        states = ['rs_{}'.format(s) for s in states]
    print('States : ',states)

        
    for sub in sublist: #compute gradient for group of selected participants
            subs.append(sub)
            for state in states:
                try:
                    embeddings.append(np.load(npy_folder+f'/embedding_dense_emb.{sub}.ses-001.{state}.npy'))
                    print(sub)
                    print(state)
                except:
                    print(sub)
                    print(state)


    realigned = run_realign(embeddings, template) #realign the gradients at the group-level 
    for i in range(5):
        realigned = run_realign(realigned, np.asarray(np.mean(realigned, axis=0).squeeze()))
    
    if len(sublist)==1: #individual-level
        path = outcome_folder+'/{}/{}_{}_embedding.mat'.format(sub,sub,condition)

    else: #group-level
        path = outcome_folder+'/group_{}_embedding.mat'.format(condition)
    
    savemat(path, mdict={'emb': realigned, 'subs': subs, 'states':states})
    f = loadmat(path)

    print('Group matrix succeded and saved in {}'.format(path))
    # print(len(f['subs']))
    # print(f['subs'])
    return f




def select_embedding(emb_condition,sublist,gradients_for):

    npy_folder = '/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_outputs/emb_output_{}'.format(gradients_for)
    template = load_template('/mnt/data/romy/hypnomed/git/data/template')

    embeddings = []
    subs = []

    states = [state for state in emb_condition.split('_')] 
    if gradients_for == 'blocks': #'run-1,run-2,run-3' : need to change the name to retrieve npy file (by adding 'rs_' prefix)
        states = ['rs_{}'.format(s) for s in states]
    print('States : ',states)

        
    for sub in sublist:
            subs.append(sub)
            for state in states:
                try:
                    embeddings.append(np.load(npy_folder+f'/embedding_dense_emb.{sub}.ses-001.{state}.npy'))
                    print(sub)
                    print(state)
                except:
                    print(sub)
                    print(state)


    realigned = run_realign(embeddings, template)
    for i in range(5):
        realigned = run_realign(realigned, np.asarray(np.mean(realigned, axis=0).squeeze()))

    
    if len(sublist) > 1:
        prefix = 'group' #group-level analysis
    else:
        prefix = sublist[0] #indiv-level analysis
    print('Gradient for : ',prefix)

    mat_folder = '/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_matrices/{}'.format(prefix)
    if not os.path.isdir:
        os.makedirs(mat_folder)
    mat_file = mat_folder+'/{}_{}_embedding.mat'.format(prefix, emb_condition)
    
    
    savemat(mat_file, mdict={'emb': realigned, 'subs': subs, 'states':states})
    f = loadmat(mat_file)

    print('{} matrix succeded and saved in {}'.format(prefix,mat_file))
    # print(len(f['subs']))
    # print(f['subs'])
    return f



def create_mat_embeddings(emb_condition,sublist,gradients_for):

    npy_folder = '/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_outputs/emb_output_{}'.format(gradients_for)
    template = load_template('/mnt/data/romy/hypnomed/git/data/template')

    embeddings = []
    subs = []

    states = [state for state in emb_condition.split('_')] 
    if gradients_for == 'blocks': #'run-1,run-2,run-3' : need to change the name to retrieve npy file (by adding 'rs_' prefix)
        states = ['rs_{}'.format(s) for s in states]
    print('States : ',states)

    for state in states:
        print('*** Generating embedding .mat file for {} ***'.format(state))
        for sub in sublist:
            subs.append(sub)
            try:
                embeddings.append(np.load(npy_folder+f'/embedding_dense_emb.{sub}.ses-001.{state}.npy'))
                print(sub)
                print(state)
            except:
                print(sub)
                print(state)


        realigned = run_realign(embeddings, template)
        for i in range(5):
            realigned = run_realign(realigned, np.asarray(np.mean(realigned, axis=0).squeeze()))

        
        if len(sublist) > 1:
            prefix = 'group' #group-level analysis
        else:
            prefix = sublist[0] #indiv-level analysis
        print('Gradient for : ',prefix)

        mat_folder = '/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_matrices/{}'.format(prefix)
        if not os.path.isdir:
            os.makedirs(mat_folder)
        mat_file = mat_folder+'/{}_{}_embedding.mat'.format(prefix, emb_condition)
        
        
        savemat(mat_file, mdict={'emb': realigned, 'subs': subs, 'states':states})
        f = loadmat(mat_file)

        print('{} matrix succeded and saved in {}'.format(prefix,mat_file))
    # print(len(f['subs']))
    # print(f['subs'])
    #return f