"""
Connectivity matrix for the human brain was based on
resting-state fMRI data. 
For each individual, a functional connectivity matrix is
calculated using the correlation coefficient across minimally
preprocessed, spatially normalized, and concatenated resting-state fMRI scans 
(Margulies, 2016 - Supplementary)


Ressource : https://brainspace.readthedocs.io/en/latest/python_doc/auto_examples/plot_tutorial0.html#sphx-glr-python-doc-auto-examples-plot-tutorial0-py
"""

def load_fs(subject, ses, state):

    #load the data from the cortex surface computed by x.mri_vol2surf.sh
    import nibabel as nib
    import numpy as np

    vol2surf_path = '/mnt/data/romy/hypnomed/git/diffusion_embedding/vol2surf_derivatives' 
    freesurfer_output = '/mnt/data/romy/hypnomed/MRI_raw/BIDS/derivatives/fmriprep-latest/sourcedata/freesurfer'

    data = []
    label = []
    for h in ['lh','rh']: #order of hemisphere is important !
        #loop through hemispheres
        data.append(nib.load(f'{vol2surf_path}/{subject}/{ses}/{subject}_{ses}_task-{state}.fsa5.{h}.mgz').get_data().squeeze())
        label.append(nib.freesurfer.read_label(f'{freesurfer_output}/fsaverage5/label/{h}.cortex.label'))

    data = np.vstack((data[0][label[0],:],data[1][label[1],:])) #concatenation 2 hemispheres (timeseries data)
    
    #normalize data before the correlation (z-score normalization)
    data = (data.T - np.nanmean(data, axis = 1)).T #mean data
    data = (data.T / np.nanstd(data, axis = 1)).T #std data

    return data
