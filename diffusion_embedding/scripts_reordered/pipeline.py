
#!/bin/python

"""
STEP 2 : Perform diffusion embedding / Correlate RS data across sessions and embed
"""

import sys, os
sys.path.append('/mnt/data/romy/hypnomed/git/diffusion_embedding/') #("/mnt/data/sebastien/diffusion_embedding_step/")
sys.path.append('/home/romy.beaute/projects/hypnomed/') #("/mnt/data/sebastien/diffusion_embedding_step/")
from load_fs import load_fs
import numpy as np
import nibabel as nib
from mapalign import embed
from numba import jit
from scipy.sparse.linalg import eigsh
from scipy import sparse
import pandas as pd
from data.get_infos import get_rs_condition

@jit(parallel=True)
def run_perc(data, thresh):
    perc_all = np.zeros(data.shape[0])
    for n,i in enumerate(data):
        data[n, i < np.percentile(i, thresh)] = 0.
    for n,i in enumerate(data):
        data[n, i < 0.] = 0.
    return data



 
def main(subj):
    print(subj)
    for ses in ["ses-001"]:
    # for ses in ["ses-001", "ses-002", "ses-003"]:
        # for state in ["rs1", "rs2", "rs3"]:
        for state in ["rs_run-1", "rs_run-2", "rs_run-3"]:
            condition = get_rs_condition(subj,state) #re-ordered rs condition (control, hypnose or meditation)
            if os.path.isfile(f'/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_output/embedding_dense_emb.{subj}.{ses}.{condition}.npy'):
            # if os.path.isfile(f'/mnt/data/sebastien/diffusion_embedding_step/emb_output/embedding_dense_emb.{subj}.{ses}.{state}.npy'):

                emb = np.load(f'/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_output/embedding_dense_emb.{subj}.{ses}.{condition}.npy')

            else:

                K = load_fs(subj,ses,state)
                K[np.isnan(K)] = 0.0

                A_mA = K - K.mean(1)[:,None]
                ssA = (A_mA**2).sum(1)
                Asq = np.sqrt(np.dot(ssA[:,None],ssA[None]))
                Adot = A_mA.dot(A_mA.T)

                K = Adot/Asq
                del A_mA, ssA, Asq, Adot
                K = run_perc(K, 90)

                norm = (K * K).sum(0, keepdims=True) ** .5
                K = K.T @ K
                aff = K / norm / norm.T
                del norm, K

                emb, res = embed.compute_diffusion_map(aff, alpha = 0.5, n_components=5, skip_checks=True, overwrite=True, eigen_solver=eigsh, return_result=True) #Compute the diffusion maps of a symmetric similarity matrix
                del aff

                np.save(f'/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_output_reordered/embedding_dense_emb.{subj}.{ses}.{condition}.npy', emb)
                np.save(f'/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_output_reordered/embedding_dense_res.{subj}.{ses}.{condition}.npy', res)
