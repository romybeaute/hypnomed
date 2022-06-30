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

df = pd.read_csv('/mnt/data/romy/hypnomed/git/diffusion_embedding/scripts/subject_list.txt', header=None) #list of the subjects we have
sublist = np.asarray(df).flatten()

data_folder = '/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_output'
# template = load_template('/mnt/data/romy/hypnomed/git/data/template/fsaverage')
template = load_template('/mnt/data/romy/hypnomed/git/data/template')

embeddings = []
subs = []
for sub in sublist:
    try:
        subs.append(sub)
        # embeddings.append(np.load(data_folder+f'/embedding_dense_emb.{sub}.fa.npy'))
        # embeddings.append(np.load(data_folder+f'/embedding_dense_emb.{sub}.om.npy'))
        # embeddings.append(np.load(data_folder+f'/embedding_dense_emb.{sub}.rest.npy'))
        embeddings.append(np.load(data_folder+f'/embedding_dense_emb.{sub}.ses-001.rs_run-1.npy'))
        embeddings.append(np.load(data_folder+f'/embedding_dense_emb.{sub}.ses-001.rs_run-2.npy'))
        embeddings.append(np.load(data_folder+f'/embedding_dense_emb.{sub}.ses-001.rs_run-3.npy'))
        print(sub)
    except:
        print(sub)


realigned = run_realign(embeddings, template)
for i in range(5):
    realigned = run_realign(realigned, np.asarray(np.mean(realigned, axis=0).squeeze()))

savemat(data_folder+'/rs1_rs2_r3_group_embedding_new.mat', mdict={'emb': realigned, 'subs': subs})

    
f = loadmat(data_folder+'/rs1_rs2_r3_group_embedding_new.mat')
print('Group matrix succeded and saved in {}'.format(data_folder+'/rs1_rs2_r3_group_embedding_new.mat'))
# print(len(f['subs']))
# print(f['subs'])
