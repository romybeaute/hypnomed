from scipy.io import loadmat
import nilearn
import nilearn.plotting
import numpy as np
import nibabel as nib
import pandas as pd

from embedding import selected_embedding


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

select_sub = 'sub-01'
if select_sub:
    print(sublist[int(select_sub.split('-')[1])-1])
    sublist = [sublist[int(select_sub.split('-')[1])-1]]
print('n = {} subjects processed'.format(len(sublist)))

select_states = ["rs_run-1", "rs_run-2", "rs_run-3"]
print('Keeping following states : ',select_states)

selected_embedding(statelist=select_states,sublist=select_sub)



for s in sublist:
    for state in select_states:
        # Note: another loop could be included to create image for each embedding component...
        # for example: for component in range(X): # where X is the number of components... 
        component = 0
        
        # Load embedding vector for subject/state

        # emb_all   = loadmat('/dycog/meditation/ERC/Analyses/MINCOM/diffusion_embedding_step/matrix_data/separate_subj_matrices/novices/%s_%s_embedding.mat' % (state,s))['emb']
        emb_all   = selected_embedding([s],[select_states])
        emb       = np.zeros(20484)
        emb[lab]  = emb_all[:,:,component] # took mean before, but now will run for each subject & state
        
        # Run for both hemispheres:
        for h in ['lh', 'rh']:
            # load values for single hemisphere
            if h == 'lh':
                vals = emb[:10242]
            elif h == 'rh': # could also be 'else:'
                vals = emb[10242:] # take the second half of the vector for rh
            nilearn.plotting.plot_surf_stat_map('/mnt/data/romy/packages/freesurfer/subjects/fsaverage5/surf/%s.inflated' % h, vals, cmap='jet', vmax=7, output_file=outcome_folder+'/diffusion_map_%s_%s_%s_%s.png' % (component,s,state,h))# vmax=5.5, commented out. better to set on group-level
        print('%s : completed' % s)

