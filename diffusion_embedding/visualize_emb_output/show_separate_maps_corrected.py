from scipy.io import loadmat
import nilearn
import nilearn.plotting
import numpy as np
import nibabel as nib
import pandas as pd

# get indices of cortical nodes:
lab_lh = nib.freesurfer.read_label('/dycog/meditation/ERC/Analyses/MINCOM/fmriprep_trials/fmriprep_output3/freesurfer/fsaverage5/label/lh.cortex.label')

lab_rh = nib.freesurfer.read_label('/dycog/meditation/ERC/Analyses/MINCOM/fmriprep_trials/fmriprep_output3/freesurfer/fsaverage5/label/rh.cortex.label') + 10242
lab    = np.concatenate((lab_lh,lab_rh))

# Load subject list
subject_wanted = 'novices'
df = pd.read_csv('/dycog/meditation/ERC/Analyses/MINCOM/diffusion_embedding_step/scripts/lists/'+subject_wanted+'_list.txt', header=None)
sublist = np.asarray(df).flatten()

one_subject = [sublist[0]]

for s in one_subject:
    for state in  ["compassion","openmonitoring","restingstate"]:
        # Note: another loop could be included to create image for each embedding component...
        # for example: for component in range(X): # where X is the number of components... 
        component = 0
        
        # Load embedding vector for subject/state
        if subject_wanted=='experts':
            emb_all   = loadmat('/dycog/meditation/ERC/Analyses/MINCOM/diffusion_embedding_step/matrix_data/separate_subj_matrices/experts/%s_%s_embedding.mat' % (state,s))['emb']
        elif subject_wanted=='novices':
            emb_all   = loadmat('/dycog/meditation/ERC/Analyses/MINCOM/diffusion_embedding_step/matrix_data/separate_subj_matrices/novices/%s_%s_embedding.mat' % (state,s))['emb']

        emb       = np.zeros(20484)
        emb[lab]  = emb_all[:,:,component] # took mean before, but now will run for each subject & state
        
        # Run for both hemispheres:
        for h in ['lh', 'rh']:
            # load values for single hemisphere
            if h == 'lh':
                vals = emb[:10242]
            elif h == 'rh': # could also be 'else:'
                vals = emb[10242:] # take the second half of the vector for rh
            nilearn.plotting.plot_surf_stat_map('/dycog/meditation/ERC/Analyses/MINCOM/fmriprep_trials/fmriprep_output3/freesurfer/fsaverage5/surf/%s.inflated' % h, vals, cmap='jet', vmax=7, output_file='diffusion_map_%s_%s_%s_%s.png' % (component,s,state,h))# vmax=5.5, commented out. better to set on group-level
        print('%s : completed' % s)

