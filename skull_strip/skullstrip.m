clear all;
close all;

img_path = '/home/juseonghan/mia/brain_segmentation_proj/data/delineated/volumes/01_ANAT_N4_MNI_fcm.nii.gz';
img = im2uint8(niftiread(img_path));
img_slice = squeeze(img(:, :, 90));
img_slice = histeq(img_slice);

window_size = 10;
global_threshold = 130;
eps = 50;

img_thresh = mixed_threshold(img_slice, window_size, global_threshold, eps, 'mean');

imshow(img_thresh);