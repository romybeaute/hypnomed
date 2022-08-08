# import nibabel as nib
# import nibabel.gifti
# import nilearn
# from nilearn import datasets, plotting
# import pandas as pd
# import numpy as np

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import pylab as plt
# from matplotlib.colors import ListedColormap
# mpl.rcParams['svg.fonttype'] = 'none'

import os
import glob
import sys
sys.path.append('/home/romy.beaute/projects/hypnomed/diffusion_embedding/helper_functions/')
from helper_functions import *

os.chdir('/home/romy.beaute/projects/hypnomed/diffusion_embedding')


for cond in ['blocks','states']:
    imgs_path = os.path.join(os.getcwd(),'visualize_emb_output/images_gradients/{}/group/Gradient_1'.format(cond))
    #imgs_path_list = glob.glob(imgs_path+"PG*") #path to all imgs beginning by PG (Principal Grandient)
    imgs_list = [im for im in os.listdir(imgs_path) if im.startswith('PG')]
    imgs_list.sort()
    print(imgs_list)
    imgs_path_list = [os.path.join(imgs_path,img_list) for img_list in imgs_list]
    side2side(imgs_path_list=imgs_path_list,output_title='PGs_{}.jpg'.format(cond))