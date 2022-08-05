
#!/bin/python

"""
STEP 2 : Perform diffusion embedding / Correlate RS data across sessions and embed


Ressources for computing correlation matrix : 
- https://stackoverflow.com/questions/26524950/how-to-apply-corr2-functions-in-multidimentional-arrays-in-matlab/26526798#26526798
- https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays

Ressources for diffusion map embedding steps :
- https://github.com/satra/mapalign/blob/master/mapalign/embed.py

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



"""
Transform z to r correlation values with hypobolic tangent function 
Generate percentile thresholds for 90th percentile
For each raw of the matrix, keep only the values at the top 10% of connections
"""
@jit(parallel=True)
def run_perc(data, thresh):
    perc_all = np.zeros(data.shape[0])
    # Threshold each row of the matrix by setting values below 90th percentile to 0
    for n,i in enumerate(data):
        data[n, i < np.percentile(i, thresh)] = 0. #values under the connection threshold (<top 10%) are zeroed
    for n,i in enumerate(data):
        data[n, i < 0.] = 0. #zeroed the rest of voxels with negative connections
    return data


"""

DIFFUSION MAP EMBEDDING STEP (for non-linear dimensionality reduction)

Forms an affinity matrix given by the specified function and
applies spectral decomposition to the corresponding graph laplacian.
The resulting transformation is given by the value of the eigenvectors for each data point.

NB  if return_result == True    :
result = dict(lambdas=lambdas, vectors=vectors,
               n_components=n_components, diffusion_time=diffusion_times,
               n_components_auto=n_components_auto)
"""

 
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

                # CALCULATE FUNCTIONNAL CONNECTIVITY MATRIX (K) FROM NORMALIZED DATA (loaded from load_fs)
                
                #Step 1 : load data
                K = load_fs(subj,ses,state) #load normalized data
                K[np.isnan(K)] = 0.0

                #Step 2 : Compute correlation coefficient from normalized data K
                A_mA = K - K.mean(1)[:,None] #Rowwise mean of input arrays & subtract from input arrays themeselves
                ssA = (A_mA**2).sum(1)  Sum of squares across rows
                Asq = np.sqrt(np.dot(ssA[:,None],ssA[None]))
                Adot = A_mA.dot(A_mA.T)
                K = Adot/Asq #finally got K as the correlation coefficient (functional connectivity matrix)
                del A_mA, ssA, Asq, Adot
                
                #Step 3 : Transformation z --> r correlations (scales values b/w [-1;1]
                K = run_perc(K, 90) #threshold to keep only top 10% connections

                #Step 4 : Compute similarity between all pairs of rows using cosine distance (resulting in positive, symmetrix affinity matrix aff)
                norm = (K * K).sum(0, keepdims=True) ** .5 
                K = K.T @ K #numerator (dot product)
                aff = K / norm / norm.T #similarities (full affinity matrix) shape = (n_samples, n_samples)
                del norm, K

                emb, res = embed.compute_diffusion_map(aff, alpha = 0.5, n_components=5, skip_checks=True, overwrite=True, eigen_solver=eigsh, return_result=True) #Compute the diffusion maps of a symmetric similarity matrix
                del aff

                np.save(f'/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_output_reordered/embedding_dense_emb.{subj}.{ses}.{condition}.npy', emb)
                np.save(f'/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_output_reordered/embedding_dense_res.{subj}.{ses}.{condition}.npy', res)
