EN 520.433/623 Medical Image Analysis
Project II: Multi-Atlas Brain Segmentation And Age Prediction

The ultimate goal of this project is to build machine learning models for predicting age from
brain MR images. You need to approach the goal step by step. From skull stripping to image
registration, and from segmentation to feature extraction, you will gain much hands-on experience
in medical image analysis through this project.

Dependencies:
- nibabel
- numpy
- antspyx
- configparse
- glob
- matplotlib
- skimage

To run skull stripping, enter the skull_strip directory. 
- skull_strip
    - main.m (main driver)
    - skullstrip.m (skull strip algorithm)
    - mixed_threshold.m (mixed thresholding algorithm)
    - get_window.m (get a window of an image with same-like boundary conditions)
    - has_only_zeros.m (returns True if a matrix has only 0s)

In main.m, specify img_path to be the path to the nifti file to skull strip. Then, specify the output_path to specify the save path to the output.


To run registration and label fusion, enter src. First, edit config.ini. 
image=path_to_target_image_to_segment
atlas_dir=path_to_atlas_volumes_dir
label_dir=path_to_atlas_labels_dir

Then run
```
$ python3 ./src/run_registration
```

