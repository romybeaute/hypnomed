import nibabel as nib
import numpy as np
import os, glob
import pandas as pd
from scipy.io import savemat
from scipy.io import loadmat
from nilearn import surface

def run_realign(emb, tar, firstpass = False):
    realign = []
    if firstpass:
        realign.append(tar)
    for i, embed in enumerate(emb):
        u, s, v = np.linalg.svd(tar.T.dot(embed), full_matrices=False)
        xfm = v.T.dot(u.T)
        realign.append(embed.dot(xfm))
    return realign


def load_template(template_path):
    cortex = loadmat('/mnt/data/romy/hypnomed/git/data/cortex.mat')
    #('/mnt/data/sebastien/data/cortex.mat')
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

# df = pd.read_csv('/mnt/data/sebastien/scripts/subject_list.txt', header=None)
# df = pd.read_csv('/mnt/data/romy/hypnomed/git/data/subject_list.txt', header=None)
df = pd.read_csv('/mnt/data/romy/hypnomed/git/diffusion_embedding/scripts/subject_list.txt', header=None)
sublist = np.asarray(df).flatten()

# data_folder = '/mnt/data/sebastien/diffusion_embedding_step/emb_output'
data_folder = '/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_output'
template = load_template('/mnt/data/romy/hypnomed/git/data/template/fsaverage')
#for state in ["compassion","openmonitoring","restingstate"]:
#for state in ["rs1","rs2","rs3"]:
embeddings = []
subs = []
for s in sublist:
    try:
        subs.append(s)
        # embeddings.append(np.load(data_folder+'/embedding_dense_emb.%s.compassion.npy' % s))
        # embeddings.append(np.load(data_folder+'/embedding_dense_emb.%s.openmonitoring.npy' % s))
        # embeddings.append(np.load(data_folder+'/embedding_dense_emb.%s.restingstate.npy' % s))
        embeddings.append(np.load(data_folder+'/embedding_dense_emb.%s.rs1.npy' % s))
        embeddings.append(np.load(data_folder+'/embedding_dense_emb.%s.rs2.npy' % s))
        embeddings.append(np.load(data_folder+'/embedding_dense_emb.%s.rs3.npy' % s))
        print(s)
    except:
        print(s)

    # np.shape(embeddings)

realigned = run_realign(embeddings[1:], template, firstpass=True)
for i in range(5):
    realigned = run_realign(realigned, np.asarray(np.mean(realigned, axis=0).squeeze()), firstpass = False)

savemat(data_folder+'/rs1_rs2_r3_group_embedding_new.mat', mdict={'emb': realigned, 'subs': subs})
    
f = loadmat(data_folder+'/rs1_rs2_r3_group_embedding_new.mat')
print(len(f['subs']))
print(f['subs'])
