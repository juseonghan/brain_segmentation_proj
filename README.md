# Multi-Atlas Brain Segmentation And Age Prediction

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
- sklearn
- os

Submissions
1. code folder
    i. To run skull stripping, enter the skull_strip directory. 
    - skull_strip
        - main.m (main driver)
        - skullstrip.m (skull strip algorithm)
        - mixed_threshold.m (mixed thresholding algorithm)
        - get_window.m (get a window of an image with same-like boundary conditions)
        - has_only_zeros.m (returns True if a matrix has only 0s)

    In main.m, specify img_path to be the path to the nifti file to skull strip. Then, specify the output_path to specify the save path to the output.


    ii. To run registration and label fusion, enter src. First, edit config.ini. 
    image=path_to_target_image_to_segment
    atlas_dir=path_to_atlas_volumes_dir
    label_dir=path_to_atlas_labels_dir

    Then run
    ```
    $ python3 ./src/run_registration
    ```

    iii. age_prediction.py contains script to predict age based on linear regression
    - outputs model with 20 input features: "trained_model.sav"
    - contains final evaluation to predict the age of 15 brain images
    - requires:
        - label.txt: label number/name map
        - train_age.csv: age ground truth for training images

2. seg_eval folder contains segmentation results and brain masks for 5 subjects
    * brainmask result using out-of-the-box package HD-BET was used in subsequent steps

3. age_eval folder contains csv file with age predictions for 15 subjects
    - ages_pred.csv

