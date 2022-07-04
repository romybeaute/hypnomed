"""
Surface plot of a 3D statistical map
@author : Romy
@date : 30/06/2022


- Use 3D statistical map previously projected onto a cortical mesh using mri_vol2surf (step in function x.mri_proj_vol2surf.sh)
- Display a surface plot of the projected map using plot_surf_stat_map

Have to define if want gradients for 'blocks' or 'states'
=> changes these parameters to generate gradients maps either between states (hypnose, meditation, control) or between blocks (rs1, rs2,rs3). The latter allows to control for potential order effect

Ressource : https://nilearn.github.io/stable/auto_examples/01_plotting/plot_3d_map_to_surface_projection.html#sphx-glr-auto-examples-01-plotting-plot-3d-map-to-surface-projection-py
"""




from scipy.io import loadmat
import nilearn
import nilearn.plotting
import numpy as np
import nibabel as nib
import pandas as pd
import os 

os.chdir('/mnt/data/romy/hypnomed/git/diffusion_embedding/visualize_emb_output')
print(os.getcwd())

import sys
sys.path.append('/home/romy.beaute/projects/hypnomed/diffusion_embedding/')
sys.path.append('/mnt/data/romy/hypnomed/git/diffusion_embedding/')

from emb_matrices.emb_matrices import *


####### TO DEFINE #######
gradients_for = 'states' #'blocks'
if gradients_for == 'states':
    emb_condition = 'control_meditation_hypnose'
else:
    emb_condition = emb_condition = 'run-1_run-2_run-3'


#########################

npy_folder = '/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_outputs/emb_output_{}'.format(gradients_for)

mesh = 'fsaverage5' #the low-resolution fsaverage5 mesh (10242 nodes)
fsaverage_path = '/mnt/data/romy/packages/freesurfer/subjects/{}/label'.format(mesh)
image_output_folder = '/mnt/data/romy/hypnomed/git/diffusion_embedding/visualize_emb_output/images_gradients/'




lab_lh = nib.freesurfer.read_label(fsaverage_path+'/lh.cortex.label') #Freesurfer fsaverage surface
lab_rh = 10242 + nib.freesurfer.read_label(fsaverage_path+'/rh.cortex.label')
lab= np.concatenate((lab_lh,lab_rh))


df = pd.read_csv('/mnt/data/romy/hypnomed/git/diffusion_embedding/scripts/subject_list.txt', header=None) #list of the subjects we have

sublist = np.asarray(df).flatten()




def make_gradients_images(sublist,emb_condition):

    
    if len(sublist) > 1:
        prefix = 'group' #group-level analysis
    else:
        prefix = sublist[0] #indiv-level analysis
    print('Gradient for : ',prefix)

    mat_folder = '/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_matrices/{}'.format(prefix)
    if not os.path.isdir:
        os.makedirs(mat_folder)
    mat_file = mat_folder+'/{}_{}_embedding.mat'.format(prefix, emb_condition)


    if not os.path.isfile(mat_file):
        select_embedding(emb_condition,sublist,gradients_for)
        print('Creating .mat file for {} conditions'.format(emb_condition))

    try:
        b = loadmat(mat_file)
        print('Loading ',mat_file)
        b['emb'].shape
        a= np.zeros(20484)
        a[lab]=np.mean(b['emb'],axis=0)[:,0]
        nilearn.plotting.plot_surf_stat_map('/mnt/data/romy/packages/freesurfer/subjects/fsaverage5/surf/lh.inflated',a[:10242],colorbar=True, cmap='jet', vmax=5.5,title='diffusion_map_group_{}_lh'.format(emb_condition),output_file='/mnt/data/romy/hypnomed/git/diffusion_embedding/visualize_emb_output/images_gradients/group/diffusion_map_group_%s_lh.png' % emb_condition)
        nilearn.plotting.plot_surf_stat_map('/mnt/data/romy/packages/freesurfer/subjects/fsaverage5/surf/rh.inflated',a[10242:],colorbar=True, cmap='jet', vmax=5.5,  title='diffusion_map_group_{}_rh'.format(emb_condition),output_file='/mnt/data/romy/hypnomed/git/diffusion_embedding/visualize_emb_output/images_gradients/group/diffusion_map_group_%s_rh.png' % emb_condition)
        print("%s completed" % mat_file)
    except:
        print("%s failed" % mat_file)


#Get gradient images for group level analysis   

make_gradients_images(sublist,emb_condition)



    

    




# for subj in sublist:
#     if not os.path.isdir(mat_folder+'/{}'):
#         os.makedirs(mat_folder+'/{}'.format(subj))
#     indiv_emb_map = mat_folder+'/{}/{}_{}_embedding.mat'.format(subj,subj,emb_condition)
#     make_images_gradients(mat_file_path=indiv_emb_map)