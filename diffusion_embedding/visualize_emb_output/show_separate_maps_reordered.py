from scipy.io import loadmat
import nilearn
import nilearn.plotting
import numpy as np
import nibabel as nib
import pandas as pd

from embedding import selected_embedding, selected_embedding_reordered


mesh = 'fsaverage5' #the low-resolution fsaverage5 mesh (10242 nodes)
fsaverage_path = '/mnt/data/romy/packages/freesurfer/subjects/{}/label'.format(mesh)

outcome_folder = '/home/romy.beaute/projects/hypnomed/diffusion_embedding/visualize_emb_output/partial_embeddings'

# get indices of cortical nodes:
lab_lh = nib.freesurfer.read_label(fsaverage_path+'/lh.cortex.label') #Freesurfer fsaverage surface
lab_rh = 10242 + nib.freesurfer.read_label(fsaverage_path+'/rh.cortex.label')
lab = np.concatenate((lab_lh,lab_rh))

# Load subject list
df = pd.read_csv('/mnt/data/romy/hypnomed/git/diffusion_embedding/scripts/subject_list.txt', header=None) 
sublist = np.asarray(df).flatten()

select_sub = None 
if select_sub:
    print(sublist[int(select_sub.split('-')[1])-1])
    sublist = [sublist[int(select_sub.split('-')[1])-1]]
print('n = {} subjects processed'.format(len(sublist)))

select_condition = 'control'
print('Keeping following states : ',select_condition)

emb_all   = selected_embedding_reordered(condition=select_condition,sublist=sublist)

try:
    emb_all['emb'].shape
    a= np.zeros(20484)
    a[lab]=np.mean(emb_all['emb'],axis=0)[:,0]
    nilearn.plotting.plot_surf_stat_map('/mnt/data/romy/packages/freesurfer/subjects/fsaverage5/surf/lh.inflated',a[:10242],colorbar=True, cmap='jet', vmax=5.5,title='{}_diffusion_map_group_lh'.format(select_condition),output_file='/mnt/data/romy/hypnomed/git/diffusion_embedding/visualize_emb_output/partial_embeddings/diffusion_map_group_%s_lh.png' % select_condition)

    nilearn.plotting.plot_surf_stat_map('/mnt/data/romy/packages/freesurfer/subjects/fsaverage5/surf/rh.inflated',a[10242:],colorbar=True, cmap='jet', vmax=5.5,  title='{}_diffusion_map_group_lh'.format(select_condition),output_file='/mnt/data/romy/hypnomed/git/diffusion_embedding/visualize_emb_output/partial_embeddings/diffusion_map_group_%s_rh.png' % select_condition)


except:
    print('failed')



for s in sublist:
    emb_indiv  = selected_embedding([s],[select_condition])