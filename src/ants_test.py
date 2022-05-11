import ants
import numpy as np
import time

def main():

    start = time.time()

    img_moving = ants.image_read('/home/juseonghan/mia/brain_segmentation_proj/output_skullstrip/delineated/01_ANAT_N4_MNI_fcm.nii.gz')
    img_fixed = ants.image_read('/home/juseonghan/mia/brain_segmentation_proj/output_skullstrip/undelineated/001_T1.nii.gz')
    img_moving = ants.resample_image(img_moving, img_fixed.shape, use_voxels=True)
    
    res = ants.registration(fixed=img_fixed, moving=img_moving, type_of_transformation="SyNRA")

    label_map = ants.image_read('/home/juseonghan/mia/brain_segmentation_proj/data/delineated/manual/01_LABELS_MNI.nii.gz')
    label_map = ants.resample_image(label_map, img_fixed.shape, use_voxels=True)
    label_registered = ants.apply_transforms(fixed=img_fixed, moving=label_map, transformlist=res['fwdtransforms'])

    ants.plot(label_map, title='atlas label image', axis=1, filename='/home/juseonghan/mia/brain_segmentation_proj/output_registration/labels.png')
    ants.plot(res['warpedmovout'], title='atlas moving image', axis=1, filename='/home/juseonghan/mia/brain_segmentation_proj/output_registration/moving.png')
    ants.plot(img_fixed, title='undelineated target image', axis=1, filename='/home/juseonghan/mia/brain_segmentation_proj/output_registration/fixed.png')
    ants.plot(label_registered, title='result (registered label map)', axis=1, filename='/home/juseonghan/mia/brain_segmentation_proj/output_registration/result.png')

    ants.image_write(label_registered, '/home/juseonghan/mia/brain_segmentation_proj/output_registration/result.nii.gz')
    end = time.time()
    
    print('Time elapsed: ', end - start)

if __name__ == '__main__':
    main()