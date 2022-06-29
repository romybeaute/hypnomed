from scipy.io import loadmat
import nilearn
import nilearn.plotting
import numpy as np
import nibabel as nib
import pandas as pd
lab_lh = nib.freesurfer.read_label('/dycog/meditation/ERC/Analyses/MINCOM/fmriprep_trials/fmriprep_output3/freesurfer/fsaverage5/label/lh.cortex.label')
lab_rh = 10242 + nib.freesurfer.read_label('/dycog/meditation/ERC/Analyses/MINCOM/fmriprep_trials/fmriprep_output3/freesurfer/fsaverage5/label/rh.cortex.label')
lab= np.concatenate((lab_lh,lab_rh))
df = pd.read_csv('/dycog/meditation/ERC/Analyses/MINCOM/diffusion_embedding_step/scripts/lists/novices_list.txt', header=None)
sublist = np.asarray(df).flatten()

for state in ["COMP","OM","RS"]:
    for s in sublist:
            try:
                b= loadmat('%s_%s_embedding.mat' % (state,s))
                b['emb'].shape
                a= np.zeros(20484)
                a[lab]=np.mean(b['emb'],axis=0)[:,0]
                nilearn.plotting.plot_surf_stat_map('/dycog/meditation/ERC/Analyses/MINCOM/fmriprep_trials/fmriprep_output3/freesurfer/fsaverage5/surf/lh.inflated',a[:10242],cmap='jet', vmax=5.5, output_file='diffusion_map_%s_%s.png' % (s,state))
                print(s)
            except:
                print(s, 'FAILED')
