"""
This script runs the registration and label fusion of a nii.gz brain MRI image. 
First, the image is run on the atlas to calculate the Normalized Mutual Information (NMI) of the atlas and the target image.
The atlas with the highest NMI is chosen. Registration is done to find the deformed label image. 

Packages: 



"""

import nibabel as nib
import numpy as np
import ants
import time
import argparse
import glob
import matplotlib.pyplot as plt

def find_most_similar(img, atlas_dir):
    # use nibabel to load in the image
    img_data = nib.load(img).get_fdata()
    t1_slice = img_data[:,:,94]
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Registration and Label Fusion')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()
    atlas_similar = find_most_similar(args.image, args.dir) # TODO: implement different similarity criterion