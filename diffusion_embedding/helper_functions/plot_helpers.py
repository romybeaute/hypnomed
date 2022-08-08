import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import matplotlib.image as mpimg 
import pathlib



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