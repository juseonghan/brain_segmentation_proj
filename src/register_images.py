import nibabel as nib
import numpy as np
import sys
import skimage as sk
import os
import nilearn as nil
from nilearn import image
import argparse


def register(moving, fixed): 
    img_moving = nib.load(moving)
    img_fixed = nib.load(fixed)
    if not compare_size(img_moving, img_fixed):
        print('resize shape: '+ ', '.join([str(x) for x in img_fixed.shape]))
        #img_moving = sk.transform.resize(img_moving, img_fixed.shape, anti_aliasing=True)
        img_moving = nil.image.resample_img(img_moving, target_affine=img_fixed.affine, target_shape=img_fixed.shape)

    mia_path = '/home/diva/John/mia/'
    output_path = 'home/diva/John/mia/output_registration/' + moving[36:38] + '_' + fixed[38:41] + '.nii.gz'
    model_path = '/home/diva/John/mia/models/brain_2d_no_smooth.h5'

    command_str = mia_path + 'lib/voxelmorph/scripts/tf/register.py --moving ' + moving + ' --fixed ' + fixed + ' --moved ' + output_path + ' --model ' + model_path + ' --gpu 0'
    os.system(command_str)

def compare_size(one, two):
    for i in range(len(one.shape)):
        if one.shape[i] is not two.shape[i]:
            return False
    return True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Put the filename')
    parser.add_argument('--moving', type=str, required=True)
    parser.add_argument('--fixed', type=str, required=True)
    args = parser.parse_args()
    register(args.moving, args.fixed)