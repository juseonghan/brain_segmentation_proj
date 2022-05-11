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

To run skull stripping:
- clone the hd-bet repo
```
git clone https://github.com/MIC-DKFZ/HD-BET
cd HD-BET
pip install -e .
```
    
- run in terminal
```
hd-bet -i INPUT_FOLDER -o OUTPUT_FOLDER
```
To run registration and label fusion, first edit config.ini. 
image=path_to_target_image_to_segment
atlas_dir=path_to_atlas_volumes_dir
label_dir=path_to_atlas_labels_dir

Then run
```
$ python3 ./src/run_registration
```
