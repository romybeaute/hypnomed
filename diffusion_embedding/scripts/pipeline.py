
#!/bin/python

import sys, os
sys.path.append("/mnt/data/romy/hypnomed/git/diffusion_embedding/") #("/mnt/data/sebastien/diffusion_embedding_step/")
from load_fs import load_fs
import numpy as np
import nibabel as nib
from mapalign import embed
from numba import jit
from scipy.sparse.linalg import eigsh
from scipy import sparse

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
            if os.path.isfile(f'/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_output/embedding_dense_emb.{subj}.{ses}.{state}.npy'):
            # if os.path.isfile(f'/mnt/data/sebastien/diffusion_embedding_step/emb_output/embedding_dense_emb.{subj}.{ses}.{state}.npy'):

                emb = np.load(f'/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_output/embedding_dense_emb.{subj}.{ses}.{state}.npy')

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

                emb, res = embed.compute_diffusion_map(aff, alpha = 0.5, n_components=5, skip_checks=True, overwrite=True, eigen_solver=eigsh, return_result=True)
                del aff

                np.save(f'/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_output/embedding_dense_emb.{subj}.{ses}.{state}.npy', emb)
                np.save(f'/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_output/embedding_dense_res.{subj}.{ses}.{state}.npy', res)


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except:
        print('error with sys.argv[1] : ',sys.argv[1])
