from scipy.io import loadmat
import nilearn
import nilearn.plotting
import numpy as np
import nibabel as nib
import pandas as pd

lab_lh = nib.freesurfer.read_label('/home/loic/Documents/fsaverage5/label/lh.cortex.label')
lab_rh = 10242 + nib.freesurfer.read_label('/home/loic/Documents/fsaverage5/label/rh.cortex.label')
lab= np.concatenate((lab_lh,lab_rh))

df = pd.read_csv('/home/loic/Documents/emb_stats/model/novices_list.txt', header=None)
sublist = np.asarray(df).flatten()

for state in ["compassion","openmonitoring","restingstate"]:
            try:
                b= loadmat('%s_group_embedding.mat' % state)
                b['emb'].shape
                a= np.zeros(20484)
                a[lab]=np.mean(b['emb'],axis=0)[:,0]
                nilearn.plotting.plot_surf_stat_map('/home/loic/Documents/fsaverage5/surf/lh.inflated',a[:10242],cmap='jet', vmax=5.5, output_file='diffusion_map_group_%s_lh.png' % state)
                print("%s completed" % state)
            except:
                print("%s failed" % state)
