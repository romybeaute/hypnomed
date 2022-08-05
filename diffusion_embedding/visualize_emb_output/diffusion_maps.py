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
else: #gradients_for == 'blocks' (controling for order effect)
    emb_condition = 'run-1_run-2_run-3'


#########################

npy_folder = '/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_outputs/emb_output_{}'.format(gradients_for)

mesh = 'fsaverage5' #the low-resolution fsaverage5 mesh (10242 nodes = vertices per hemisphere)
fsaverage_path = '/mnt/data/romy/packages/freesurfer/subjects/{}/label'.format(mesh)
image_output_folder = '/mnt/data/romy/hypnomed/git/diffusion_embedding/visualize_emb_output/images_gradients/{}'.format(gradients_for) #where to store the visualisations of the gradient(s)



# Load Freesurfer fsaverage surface with read_label : return array with indices of vertices included in label
lab_lh = nib.freesurfer.read_label(fsaverage_path+'/lh.cortex.label') #shape = (9361,)
lab_rh = 10242 + nib.freesurfer.read_label(fsaverage_path+'/rh.cortex.label') #shape = (9361,)
lab= np.concatenate((lab_lh,lab_rh))


df = pd.read_csv('/mnt/data/romy/hypnomed/git/diffusion_embedding/scripts/subject_list.txt', header=None) #list of the subjects we have

sublist = np.asarray(df).flatten()
print(len(sublist))



def make_gradients_images(sublist,emb_condition):

    
    if len(sublist) > 1:
        prefix = 'group' #group-level analysis
        image_folder = os.path.join(image_output_folder,prefix)
    else:
        prefix = sublist[0] #indiv-level analysis
        image_folder = os.path.join(image_output_folder+'/indivs',prefix)
    print('Gradient for : ',prefix)

    
    if not os.path.isdir(image_folder):
        os.makedirs(image_folder)

    mat_folder = '/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_matrices/{}'.format(prefix)
    if not os.path.isdir(mat_folder):
        os.makedirs(mat_folder)
    mat_file = mat_folder+'/{}_{}_embedding.mat'.format(prefix, emb_condition)


    if not os.path.isfile(mat_file): #create .mat embedding file for specified sublist & condition
        select_embedding(emb_condition,sublist,gradients_for) 
        print('Creating .mat file for sublist : {} under {} condition(s)'.format(sublist,emb_condition))
    
    try:
        b = loadmat(mat_file)
        print('Loading ',mat_file)

        print(70*'-')
        print('Description {} : '.format(mat_file))
        for k in b.keys():
            print('- {} :\n {}\n'.format(k,b[k]))
        print('- b embedding shape : ',b['emb'].shape)
        print(70*'-')

        b['emb'].shape
        a= np.zeros(20484)
        a[lab]=np.mean(b['emb'],axis=0)[:,0] #check if corresponds to 1st gradient
        nilearn.plotting.plot_surf_stat_map('/mnt/data/romy/packages/freesurfer/subjects/fsaverage5/surf/lh.inflated',a[:10242],colorbar=True, cmap='jet', vmax=5.5,title='{}_diffusion_map_{}_lh'.format(prefix,emb_condition),output_file=image_folder+'/{}_diffusion_map_{}_lh.png'.format(prefix,emb_condition))
        nilearn.plotting.plot_surf_stat_map('/mnt/data/romy/packages/freesurfer/subjects/fsaverage5/surf/rh.inflated',a[10242:],colorbar=True, cmap='jet', vmax=5.5,  title='{}_diffusion_map_{}_rh'.format(prefix,emb_condition),output_file=image_folder+'/{}_diffusion_map_{}_rh.png'.format(prefix,emb_condition))
        print("Gradient image for %s completed" % mat_file)
    except:
        print("/!\ Gradient image for %s failed /!\ " % mat_file)





def make_gradients_images_isolated_condition(sublist,indiv_emb_state,plot_n_dims=1):

    
    if len(sublist) > 1:
        prefix = 'group' #group-level analysis
        image_folder = os.path.join(image_output_folder,prefix)
    else:
        prefix = sublist[0] #indiv-level analysis
        image_folder = os.path.join(image_output_folder+'/indivs',prefix)
    print('Gradient for : ',prefix)

    image_folder = os.path.join(image_output_folder,prefix)
    if not os.path.isdir(image_folder):
        os.makedirs(image_folder)

    mat_folder = '/mnt/data/romy/hypnomed/git/diffusion_embedding/emb_matrices/{}'.format(prefix)
    if not os.path.isdir(mat_folder):
        os.makedirs(mat_folder)
    mat_file = mat_folder+'/{}_{}_embedding.mat'.format(prefix, indiv_emb_state)


    if not os.path.isfile(mat_file): #create .mat embedding file for specified sublist & condition
        create_mat_embeddings(indiv_emb_state,sublist,gradients_for)
        print('Creating .mat file for sublist : {} under {} condition'.format(sublist,indiv_emb_state))
    
    try:
        b = loadmat(mat_file)
        print('Loading ',mat_file)

        print(70*'-')
        print('Description {} : '.format(mat_file))
        for k in b.keys():
            print('- {} :\n {}\n'.format(k,b[k]))
        print('- b embedding shape : ',b['emb'].shape)
        print('     ---> n_subjects = ',b['emb'].shape[0])
        print('     ---> n_nodes = ',b['emb'].shape[1])
        print('     ---> n_dimensions = ',b['emb'].shape[2])
        print(70*'-')

        b['emb'].shape  #(n_subjects,n_nodes,n_dims)
        n_dims = b['emb'].shape[2]
        # a = np.zeros(20484) #size concatenated vertices from fsaverage5 template
        # #mean across subjects
        # mean_embs = np.mean(b['emb'],axis=0) #mean embeddings for each dimension, across subjects (axis=0 signifies avg across subjects, which is the first dimension of b)
        # a[lab]=np.mean(b['emb'],axis=0)[:,0] #check if corresponds to 1st gradient
        # nilearn.plotting.plot_surf_stat_map('/mnt/data/romy/packages/freesurfer/subjects/fsaverage5/surf/lh.inflated',a[:10242],colorbar=True, cmap='jet', vmax=5.5,title='{}_diffusion_map_{}_lh'.format(prefix,indiv_emb_state),output_file=image_folder+'/{}_diffusion_map_{}_lh.png'.format(prefix,emb_condition))
        # nilearn.plotting.plot_surf_stat_map('/mnt/data/romy/packages/freesurfer/subjects/fsaverage5/surf/rh.inflated',a[10242:],colorbar=True, cmap='jet', vmax=5.5,  title='{}_diffusion_map_{}_rh'.format(prefix,indiv_emb_state),output_file=image_folder+'/{}_diffusion_map_{}_rh.png'.format(prefix,emb_condition))
        # print("Gradient image for %s completed" % mat_file)

        
        if plot_n_dims > n_dims:
            plot_n_dims = n_dims #control if nb of dims > actual nb

        print('Creating images for {} dims'.format(plot_n_dims))
        for dim in range(plot_n_dims):

            image_folder_dims = image_folder+'/Gradient_{}'.format(dim+1) 
            if not os.path.isdir(image_folder_dims):
                os.makedirs(image_folder_dims)

            a = np.zeros(20484) #size concatenated vertices from fsaverage5 template
            mean_embs = np.mean(b['emb'],axis=0) #mean embeddings for each dimension, across subjects (axis=0 signifies avg across subjects, which is the first dimension of b)
            a[lab]=np.mean(b['emb'],axis=0)[:,dim] #check if corresponds to 1st gradient
            nilearn.plotting.plot_surf_stat_map('/mnt/data/romy/packages/freesurfer/subjects/fsaverage5/surf/lh.inflated',a[:10242],colorbar=True, cmap='jet', vmax=5.5,title='{}_diffusion_map_{}_lh_DIMENSION#{}'.format(prefix,indiv_emb_state,dim+1),output_file=image_folder_dims+'/{}_diffusion_map_{}_lh_Dim#{}.png'.format(prefix,indiv_emb_state,dim+1))
            nilearn.plotting.plot_surf_stat_map('/mnt/data/romy/packages/freesurfer/subjects/fsaverage5/surf/rh.inflated',a[10242:],colorbar=True, cmap='jet', vmax=5.5, title='{}_diffusion_map_{}_rh_DIMENSION#{}'.format(prefix,indiv_emb_state,dim+1),output_file=image_folder_dims+'/{}_diffusion_map_{}_rh_Dim#{}.png'.format(prefix,indiv_emb_state,dim+1))
            print("Gradient image for %s completed" % mat_file)


    except:
        print("/!\ Gradient image for %s failed /!\ " % mat_file)



"""
Launch the script to get gradient images for group level analysis (have to select for block or states analysis) :
"""

# emb_conditions = emb_condition.split('_')
# print('Making gradients for following conditions : ',emb_conditions)
# print(sublist)
# for subject in sublist:
#     for condition in emb_conditions:
#         make_gradients_images(sublist=[subject],emb_condition=condition)




"""
Option 2 : launch for individual embedding conditions (not mixed)
"""
indiv_emb_states = emb_condition.split('_')
print(indiv_emb_states)
for indiv_emb_state in indiv_emb_states:
  make_gradients_images_isolated_condition(sublist,indiv_emb_state,plot_n_dims=5)



