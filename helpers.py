import numpy as np
import nibabel as nib
import os, glob, sys
import pandas as pd
from scipy.io import savemat
from scipy.io import loadmat
from nilearn import surface
from scipy.io import loadmat
from PIL import Image
from IPython.display import Image as im
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg 
import pathlib


data_path = '/home/romy.beaute/projects/hypnomed/data'



class embedding:
    def __init__(self, data_path, emb_file, group_file, states_file, sub_file):
        self.data_path = data_path
        self.emb = loadmat(os.path.join(data_path,'emb_matrices',emb_file))['emb']
        self.group = pd.read_csv(os.path.join(data_path,'participants',expertise_file))
        self.states = pd.read_csv(os.path.join(data_path, 'participants',states_file))
        self.sub = pd.read_csv(os.path.join(data_path, 'participants',sub_file))

    def remove_sub(self, sub_ids):
        # sub_ids needs to be a list or a tupple
        outliers_mask = ~self.sub['Subs_ID'].isin(sub_ids)
        self.group = self.group[outliers_mask]
        self.states = self.states[outliers_mask]
        self.sub = self.sub[outliers_mask]
        self.emb = self.emb[outliers_mask, :, :]

    def get_group(self, group):
        if group == 'G1':
            sub_ids = list(self.sub['Subs_ID'][self.group.G1 == 0])
            self.remove_sub(sub_ids)
        elif group == 'G2':
            sub_ids = list(self.sub['Subs_ID'][self.group.G2 == 0])
            self.remove_sub(sub_ids)
        elif group != 'all':
            sys.exit(
                'error in p.group, group not recognized ! Should be G1, G2 or all')

    def get_states(self, states_wanted):
        states_mask = np.zeros(self.emb.shape[0])
        for states_id in range(len(states_wanted)):
            states_mask = states_mask + self.states[states_wanted[states_id]]
        self.states = self.states[states_wanted]
        self.group = self.group[states_mask != 0]
        self.states = self.states[states_mask != 0]
        self.sub = self.sub[states_mask != 0]
        self.emb = self.emb[states_mask != 0, :, :]

    def get_neighborhood(self, p):
        self.fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
        coords_left, _ = surface.load_surf_mesh(self.fsaverage['infl_left'])
        coords_right, _ = surface.load_surf_mesh(self.fsaverage['infl_right'])
        # same order than in pipeline
        self.coords = np.vstack([coords_left, coords_right])
        nn = neighbors.NearestNeighbors(radius=p.radius)
        self.adjacency = nn.fit(self.coords).radius_neighbors_graph(self.coords).tolil()
        
    def get_surf_emb(self,p):
        cortex = loadmat(os.path.join(p.data_path, 'cortex.mat'))
        cortex = cortex['cortex']
        surface_emb = np.zeros([self.emb.shape[0], 20484,len(p.dimension)])  # 20484 = nb of voxels for fsaverage5
        surface_emb[:, np.squeeze(cortex),:] = self.emb
        self.emb = surface_emb
        return self

class p:
    pass


def load_data(p):
    data_path = p.data_path
    diff_emb = embedding(data_path,
                         'group_{}_embedding.mat'.format(p.data_file),
                         'group_covs.csv',
                         'state_covs.csv',
                         'subject_covs.csv')
    diff_emb.emb = diff_emb.emb[:, :, p.dimension]

    df_pheno = pd.read_csv('/home/romy.beaute/projects/hypnomed/data/participants/data_pheno.csv',sep=',')

    diff_emb.get_states(p.states_wanted)
    diff_emb.get_group(p.group)
    
    commun_sub_id = np.array(list(set(np.unique(diff_emb.sub)) & set(df_pheno['id'].to_numpy())))
    print('n={} subjects before outlier removal'.format(len(commun_sub_id)))
    commun_sub_id = commun_sub_id[~np.in1d(commun_sub_id,p.outliers)] #id of subjects after outliers removal
    df_pheno = df_pheno[df_pheno['id'].isin(commun_sub_id)] #take only the non-outliers subjects
    print('n = {} subjects (after outliers removal)'.format(len(df_pheno)))
    p.outliers = np.hstack((p.outliers,(np.unique(diff_emb.sub['Subs_ID'][~diff_emb.sub.isin(commun_sub_id).to_numpy()[:,0]]))))
    print('outliers : ',p.outliers)
    
    diff_emb.remove_sub(p.outliers)

    return diff_emb,df_pheno


def imageCrop(filename):
    """
    Used in gradients_networks_yeo.ipynb
    """
    i1 = Image.open(filename)
    i2 = np.array(i1)
    i2[i2.sum(axis=2) == 255*4,:] = 0
    i3 = i2.sum(axis=2)
    x = np.where((i3.sum(axis=1) != 0) * 1)[0]
    y = np.where((i3.sum(axis=0) != 0) * 1)[0]

    result = Image.fromarray(i2[x.squeeze()][:,y.squeeze()])
    result.save(filename)


def load_template(template_path):
    cortex = loadmat('/mnt/data/romy/hypnomed/git/data/cortex.mat')
    cortex = np.squeeze(cortex['cortex'])
    template = np.zeros([20484,5])
    for grad_id in range(5):
        template_L = surface.load_surf_data(os.path.join(template_path,'hcp.embed.grad_'+str(grad_id+1)+'.L.fsa5.func.gii'))
        template_R = surface.load_surf_data(os.path.join(template_path,'hcp.embed.grad_'+str(grad_id+1)+'.R.fsa5.func.gii'))
        template[:,grad_id] = np.hstack([template_L,template_R])
    template = template[cortex,:]
    template = template - np.mean(template,0)
    template = template/np.std(template,0)
    return template


def emb_path(condition,emb_outpath):
  return os.path.join(emb_outpath,'group_{}_embedding.mat'.format(condition))


def load_embmat(emb_path,show_infos=True):
  '''
  Return the principal gradient of the .mat file localised in emb_path
  '''
  b = loadmat(file_name=emb_path) #.mat file
  if show_infos:
    print(' - shape embedding (n_subjects, n_voxels, n_dims): {}\n - n = {} subjects\n - condition : {}\n - path : {}\n'.format(b['emb'].shape,len(b['subs']),b['states'],emb_path))
  return b,b['emb'][:,:,0]



def get_rs_condition(subj,state,df_path='/home/romy.beaute/projects/hypnomed/data/hypnomed.csv'):
    """
    Retrieve to which condition run_2 & rs_3 belong to (hypnose or meditation)
    """
    df = pd.read_csv(df_path,sep=';',index_col='sub_id')
    return df.loc[subj][state]



def make_gradients_images(sublist,emb_condition):

  """
  Get the statistical images of the gradients
  More complete utilisation here : https://github.com/romybeaute/hypnomed/blob/main/diffusion_embedding/visualize_emb_output/diffusion_maps.py
  The images of the gradients can be found here : https://github.com/romybeaute/hypnomed/tree/main/diffusion_embedding/visualize_emb_output/images_gradients
  """

  image_output_folder = '/mnt/data/romy/hypnomed/git/diffusion_embedding/visualize_emb_output/images_gradients/{}'.format(emb_condition) #where to store the visualisations of the gradient(s)
  
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



def img_is_color(img):

    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False

def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(20, 10), title_fontsize=30):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.
    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    '''

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images  = len(list_images)
    num_cols    = min(num_images, num_cols)
    num_rows    = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):

        img    = list_images[i]
        title  = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap   = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')
        
        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize) 
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    _ = plt.show()




def side2side(imgs_path_list,output_title):

    imgs_names = [pathlib.PurePath(imgs_path_list[i]).name for i in range(len(imgs_path_list))]

    list_titles = [im.split('_')[1] for im in imgs_names]


    my_dpi = 300
    fig = plt.figure(figsize=(15, 15), dpi=my_dpi)

    # ================================= STATES =================================


    # ============ AX1 ============ 
    # PIL Image
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.set_title(imgs_names[0])
    # ax1.set_xlabel('Gradient 2')
    # ax1.set_ylabel('Gradient 1')
    ax1.set_xticks([])
    ax1.set_yticks([])
    pil_img = Image.open(imgs_path_list[0])
    ax1.imshow(pil_img)

    # ============ AX1 ============ 
    # PIL Image
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.set_title(imgs_names[1])
    # ax2.set_xlabel('X label')
    # ax2.set_ylabel('Y label')
    ax2.set_xticks([])
    ax2.set_yticks([])
    pil_img = Image.open(imgs_path_list[1])
    ax2.imshow(pil_img)

    # ============ AX2 ============ 
    # mpimg image
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.set_title(imgs_names[2])
    ax3.set_xticks([])
    ax3.set_yticks([])
    mpimg_img = mpimg.imread(imgs_path_list[2]) 
    ax3.imshow(mpimg_img)



    # ================================= BLOCKS =================================


    # ============ AX1 ============ 
    # PIL Image
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.set_title(imgs_names[3])
    ax4.set_xticks([])
    ax4.set_yticks([])
    pil_img = Image.open(imgs_path_list[3])
    ax4.imshow(pil_img)

    # ============ AX1 ============ 
    # PIL Image
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.set_title(imgs_names[4])
    # ax2.set_xlabel('X label')
    # ax2.set_ylabel('Y label')
    ax5.set_xticks([])
    ax5.set_yticks([])
    pil_img = Image.open(imgs_path_list[4])
    ax5.imshow(pil_img)

    # ============ AX2 ============ 
    # mpimg image
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.set_title(imgs_names[5])
    ax6.set_xticks([])
    ax6.set_yticks([])
    mpimg_img = mpimg.imread(imgs_path_list[5]) 
    ax6.imshow(mpimg_img)


    savepath = os.path.join(str(pathlib.PurePath(imgs_path_list[0]).parent),output_title)
    fig.savefig(savepath, dpi='figure', bbox_inches='tight')


def plot_surf_stat_map(coords, faces, stat_map=None,
        elev=0, azim=0,
        cmap='jet',
        threshold=None, bg_map=None,
        mask=None,
        bg_on_stat=False,
        alpha='auto',
        vmax=None, symmetric_cbar="auto", returnAx=False,
        figsize=(14,11), label=None, lenient=None,
        **kwargs):

    ''' Visualize results on cortical surface using matplotlib'''
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    from mpl_toolkits.mplot3d import Axes3D

    # load mesh and derive axes limits
    faces = np.array(faces, dtype=int)
    limits = [coords.min(), coords.max()]

    # set alpha if in auto mode
    if alpha == 'auto':
        if bg_map is None:
            alpha = .5
        else:
            alpha = 1

    # if cmap is given as string, translate to matplotlib cmap
    if type(cmap) == str:
        cmap = plt.cm.get_cmap(cmap)

    # initiate figure and 3d axes
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    fig.patch.set_facecolor('white')
    ax1 = fig.add_subplot(111, projection='3d', xlim=limits, ylim=limits)
    # ax1._axis3don = False
    ax1.grid(False)
    ax1.set_axis_off()
    ax1.w_zaxis.line.set_lw(0.)
    ax1.set_zticks([])
    ax1.view_init(elev=elev, azim=azim)
    
    # plot mesh without data
    p3dcollec = ax1.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                                triangles=faces, linewidth=0.,
                                antialiased=False,
                                color='white')

    
    if mask is not None:
        cmask = np.zeros(len(coords))
        cmask[mask] = 1
        cutoff = 2
        if lenient:
            cutoff = 0
        fmask = np.where(cmask[faces].sum(axis=1) > cutoff)[0]
        
    # If depth_map and/or stat_map are provided, map these onto the surface
    # set_facecolors function of Poly3DCollection is used as passing the
    # facecolors argument to plot_trisurf does not seem to work
    if bg_map is not None or stat_map is not None:

        face_colors = np.ones((faces.shape[0], 4))
        face_colors[:, :3] = .5*face_colors[:, :3]

        if bg_map is not None:
            bg_data = bg_map
            if bg_data.shape[0] != coords.shape[0]:
                raise ValueError('The bg_map does not have the same number '
                                 'of vertices as the mesh.')
            bg_faces = np.mean(bg_data[faces], axis=1)
            bg_faces = bg_faces - bg_faces.min()
            bg_faces = bg_faces / bg_faces.max()
            face_colors = plt.cm.gray_r(bg_faces)

        # modify alpha values of background
        face_colors[:, 3] = alpha*face_colors[:, 3]

        if stat_map is not None:
            stat_map_data = stat_map
            stat_map_faces = np.mean(stat_map_data[faces], axis=1)
            if label:
                stat_map_faces = np.median(stat_map_data[faces], axis=1)

            # Call _get_plot_stat_map_params to derive symmetric vmin and vmax
            # And colorbar limits depending on symmetric_cbar settings
            cbar_vmin, cbar_vmax, vmin, vmax = \
                _get_plot_stat_map_params(stat_map_faces, vmax,
                                          symmetric_cbar, kwargs)

            if threshold is not None:
                kept_indices = np.where(abs(stat_map_faces) >= threshold)[0]
                stat_map_faces = stat_map_faces - vmin
                stat_map_faces = stat_map_faces / (vmax-vmin)
                if bg_on_stat:
                    face_colors[kept_indices] = cmap(stat_map_faces[kept_indices]) * face_colors[kept_indices]
                else:
                    face_colors[kept_indices] = cmap(stat_map_faces[kept_indices])
            else:
                stat_map_faces = stat_map_faces - vmin
                stat_map_faces = stat_map_faces / (vmax-vmin)
                if bg_on_stat:
                    if mask is not None:
                        face_colors[fmask,:] = cmap(stat_map_faces)[fmask,:] * face_colors[fmask,:]
                    else:
                        face_colors = cmap(stat_map_faces) * face_colors
                else:
                    face_colors = cmap(stat_map_faces)

        p3dcollec.set_facecolors(face_colors)
    
    if returnAx == True:
        return fig, ax1
    else:
        return fig

def _get_plot_stat_map_params(stat_map_data, vmax, symmetric_cbar, kwargs,
    force_min_stat_map_value=None):
    
    import numpy as np

    '''
    Helper function copied from nilearn to force symmetric colormaps
    https://github.com/nilearn/nilearn/blob/master/nilearn/plotting/img_plotting.py#L52
    '''
    # make sure that the color range is symmetrical
    if vmax is None or symmetric_cbar in ['auto', False]:
        # Avoid dealing with masked_array:
        if hasattr(stat_map_data, '_mask'):
            stat_map_data = np.asarray(
                    stat_map_data[np.logical_not(stat_map_data._mask)])
        stat_map_max = np.nanmax(stat_map_data)
        if force_min_stat_map_value == None:
            stat_map_min = np.nanmin(stat_map_data)
        else:
            stat_map_min = force_min_stat_map_value
    if symmetric_cbar == 'auto':
        symmetric_cbar = stat_map_min < 0 and stat_map_max > 0
    if vmax is None:
        vmax = max(-stat_map_min, stat_map_max)
    if 'vmin' in kwargs:
        raise ValueError('this function does not accept a "vmin" '
            'argument, as it uses a symmetrical range '
            'defined via the vmax argument. To threshold '
            'the map, use the "threshold" argument')
    vmin = -vmax
    if not symmetric_cbar:
        negative_range = stat_map_max <= 0
        positive_range = stat_map_min >= 0
        if positive_range:
            cbar_vmin = 0
            cbar_vmax = None
        elif negative_range:
            cbar_vmax = 0
            cbar_vmin = None
        else:
            cbar_vmin = stat_map_min
            cbar_vmax = stat_map_max
    else:
        cbar_vmin, cbar_vmax = None, None
    return cbar_vmin, cbar_vmax, vmin, vmax

    

def showSurf(input_data, surf, sulc, cort, showall=None, output_file=None):    

    import matplotlib.pyplot as plt
    
    f = plot_surf_stat_map(surf[0], surf[1], bg_map=sulc, mask=cort, stat_map=input_data, bg_on_stat=True, azim=0)
    plt.show()

    if output_file:
        count = 0
        f.savefig((output_file + '.%s.png') % str(count))
        count += 1
    f = plot_surf_stat_map(surf[0], surf[1], bg_map=sulc, mask=cort, stat_map=input_data, bg_on_stat=True, azim=180)
    plt.show()
    if output_file:
        f.savefig((output_file + '.%s.png') % str(count))
        count += 1
    if showall:
        f = plot_surf_stat_map(surf[0], surf[1], bg_map=sulc, mask=cort, stat_map=input_data, bg_on_stat=True, azim=90)
        plt.show()
        if output_file:
            f.savefig((output_file + '.%s.png') % str(count))
            count += 1
        f = plot_surf_stat_map(surf[0], surf[1], bg_map=sulc, mask=cort, stat_map=input_data, bg_on_stat=True, azim=270)
        plt.show()
        if output_file:
            f.savefig((output_file + '.%s.png') % str(count))
            count += 1
        f = plot_surf_stat_map(surf[0], surf[1], bg_map=sulc, mask=cort, stat_map=input_data, bg_on_stat=True, elev=90)
        plt.show()
        if output_file:
            f.savefig((output_file + '.%s.png') % str(count))
            count += 1
        f = plot_surf_stat_map(surf[0], surf[1], bg_map=sulc, mask=cort, stat_map=input_data, bg_on_stat=True, elev=270)
        plt.show()
        if output_file:
            f.savefig((output_file + '.%s.png') % str(count))
            count += 1