"""
Surface plot of a 3D statistical map
@author : Romy
@date : 30/06/2022


- Use 3D statistical map previously projected onto a cortical mesh using mri_vol2surf (step in function x.mri_proj_vol2surf.sh)
- Display a surface plot of the projected map using plot_surf_stat_map


Ressource : https://nilearn.github.io/stable/auto_examples/01_plotting/plot_3d_map_to_surface_projection.html#sphx-glr-auto-examples-01-plotting-plot-3d-map-to-surface-projection-py
"""




from scipy.io import loadmat
import nilearn
import nilearn.plotting
import numpy as np
import nibabel as nib
import pandas as pd



mesh = 'fsaverage5' #the low-resolution fsaverage5 mesh (10242 nodes)
fsaverage_path = '/mnt/data/romy/packages/freesurfer/subjects/{}/label'.format(mesh)

# fsaverage_path = '/mnt/data/romy/hypnomed/git/data/template/fsaverage/label'

data_folder = '/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_output_reordered'
group_emb_map = data_folder+'/cont_med_hyp_group_embedding_new.mat'
state = 'control_hypnose_meditation'

lab_lh = nib.freesurfer.read_label(fsaverage_path+'/lh.cortex.label') #Freesurfer fsaverage surface
lab_rh = 10242 + nib.freesurfer.read_label(fsaverage_path+'/rh.cortex.label')
lab= np.concatenate((lab_lh,lab_rh))

# df = pd.read_csv('/home/loic/Documents/emb_stats/model/novices_list.txt', header=None)
df = pd.read_csv('/mnt/data/romy/hypnomed/git/diffusion_embedding/scripts/subject_list.txt', header=None) #list of the subjects we have

sublist = np.asarray(df).flatten()


try:

    b= loadmat(data_folder+'/cont_med_hyp_group_embedding_new.mat')
    b['emb'].shape
    a= np.zeros(20484)
    a[lab]=np.mean(b['emb'],axis=0)[:,0]
    nilearn.plotting.plot_surf_stat_map('/mnt/data/romy/packages/freesurfer/subjects/fsaverage5/surf/lh.inflated',a[:10242],colorbar=True, cmap='jet', vmax=5.5,title='Diffusion map group level (control_hypnose_meditation) - Left Hem.',output_file='/mnt/data/romy/hypnomed/git/diffusion_embedding/visualize_emb_output/diffusion_map_group_%s_lh.png' % state)

    #lh_fig.show()
    nilearn.plotting.plot_surf_stat_map('/mnt/data/romy/packages/freesurfer/subjects/fsaverage5/surf/rh.inflated',a[10242:],colorbar=True, cmap='jet', vmax=5.5,  title='Diffusion map group level (control_hypnose_meditation) - Right Hem.',output_file='/mnt/data/romy/hypnomed/git/diffusion_embedding/visualize_emb_output/diffusion_map_group_%s_rh.png' % state)

    #nilearn.plotting.plot_surf_stat_map('/mnt/data/romy/packages/freesurfer/subjects/fsaverage5/surf/lh.inflated',np.mean(b['emb'],axis=0)[:,0],cmap='jet', vmax=5.5, output_file='diffusion_map_group_%s_lh.png' % state)

    print("%s completed" % group_emb_map)
except:
    print("%s failed" % group_emb_map)