addpath(genpath('nifti_utils'));
% read in the image
img_path = '/home/juseonghan/mia/brain_segmentation_proj/data/delineated/volumes/01_ANAT_N4_MNI_fcm.nii.gz';
output_path = '/home/juseonghan/mia/brain_segmentation_proj/skull_strip/results/';
img = im2uint8(niftiread(img_path));
info = niftiinfo(img_path);

% hyperparameters
window_size = 20;
global_threshold = 27;
eps = 5;

result_vol = zeros(size(img));
masks_vol = zeros(size(img));

for i = 229:size(img, 3)
    [result, result_mask] = skullstrip(img(:,:,i), window_size, global_threshold, eps);
    result_vol(:,:,i) = result;
    masks_vol(:,:,i) = result_mask; 
end

template_nii = load_untouch_nii(img_path);
template_nii.img = result_vol;
nifti_utils.save_untouch_nii_using_scaled_img_info(strcat(output_path + 'result.nii.gz'),template_nii,'double');

template_nii.img = masks_vol;
nifti_utils.save_untouch_nii_using_scaled_img_info(strcat(output_path + 'result_masks.nii.gz'),template_nii,'double');
