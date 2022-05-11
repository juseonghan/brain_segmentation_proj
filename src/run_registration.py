"""
This script runs the registration and label fusion of a nii.gz brain MRI image. 
First, the image is run on the atlas to calculate the Normalized Mutual Information (NMI) of the atlas and the target image.
The atlas with the highest NMI is chosen. Registration is done to find the deformed label image. 

"""

import nibabel as nib
import numpy as np
import ants
import time
import argparse
import configparser
import glob
import matplotlib.pyplot as plt
import skimage as sk
from sklearn.metrics.cluster import normalized_mutual_info_score

def find_most_similar(img, atlas_dir):

    # use nibabel to load in the image
    img_data = nib.load(img).get_fdata()

    # glob in all the atlases   
    atlases = glob.glob(atlas_dir + '*.nii.gz')

    # best atlas
    best_atlas_MI = -1
    best_atlas_index = -1
    count = 0; 

    # loop through the atlases and find the best one 
    for atlas in atlases:

        # first resize
        atlas_data = nib.load(atlas).get_fdata()
        img_data = sk.transform.resize(img_data, atlas_data.shape)

        # loop through all of them to get the average mutual info
        avg_mutual_info = 0; 
        for i in range(atlas_data.shape[2]):
            hist_2d, x_edges, y_edges = np.histogram2d(img_data[:,:,i].ravel(), atlas_data[:,:,i].ravel(), bins=20)
            avg_mutual_info = avg_mutual_info + mutual_information(hist_2d)
        avg_mutual_info = avg_mutual_info / atlas_data.shape[2]
        
        if avg_mutual_info > best_atlas_MI:
            best_atlas_index = count
            best_atlas_MI = avg_mutual_info
        
        count = count + 1
        print('For Atlas ' + str(count) + ', MI = ' + str(avg_mutual_info))
    
    print('Best was ' + atlases[best_atlas_index] + ' with MI = ' + str(best_atlas_MI))
    return atlases[best_atlas_index]

    
def mutual_information(hgram):
    """ 
    Mutual information for joint histogram. See http://en.wikipedia.org/wiki/Mutual_information
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals

    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def register(img_Moving, img_Fixed, label_dir, output_dir):

    # img_moving is a atlas image
    # img_fixed is a undelineated image

    # register between the two volumes first
    img_moving = ants.image_read(img_Moving)
    img_fixed = ants.image_read(img_Fixed)
    img_fixed = ants.resample_image(img_fixed, img_moving.shape, use_voxels=True)
    res = ants.registration(fixed=img_fixed, moving=img_moving, type_of_transformation="SyNRA")

    # find the corresponding label map after registration
    num = img_Moving[-25:-23]
    label_img = label_dir + num + '_LABELS_MNI.nii.gz'
    print('Registering ' + label_img + ' to ' + img_Fixed)

    # read in, resize, and register the label map
    label_map = ants.image_read(label_img)
    # label_map = ants.resample_image(label_map, img_fixed.shape, use_voxels=True)
    label_registered = ants.apply_transforms(fixed=img_fixed, moving=label_map, transformlist=res['fwdtransforms'])

    # save the file and write the output
    savename = output_dir + num + '_' + img_Fixed[-13:-10] + '_registration'
    ants.plot(label_registered, title='result (registered label map)', axis=1, filename=savename+'.png')
    ants.image_write(label_registered, savename + '.nii.gz')
    print('Saved to ' + savename)


if __name__ == '__main__':
    parser = configparser.ConfigParser()
    parser.read('config.ini')
    config = []
    for sect in parser.sections():
        for k, v in parser.items(sect):
            config.append(v)

    image_dir = config[0]
    atlas_dir = config[1]
    label_dir = config[2]
    output_dir = config[3]

    start = time.time()

    images = glob.glob(image_dir + '*.nii.gz')

    for image in images:
        print('Processing ' + image_dir + '...')
        atlas_similar = find_most_similar(image, atlas_dir) # TODO: implement different similarity criterion
        register(atlas_similar, image, label_dir, output_dir)
        print('Successful.')

    print('Elapsed time: ' + str(time.time() - start))