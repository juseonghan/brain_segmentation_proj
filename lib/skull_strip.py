import numpy as np
import nibabel as nib
import cv2 as cv 

def main():
    img1 = nib.load('/home/juseonghan/mia/brain_segmentation_proj/data/12_ANAT_N4_MNI_fcm.nii.gz')
    
    # header = img1.header
    data = img1.get_fdata()
    data = 255 * data / (np.amax(data))
    data = data.astype(np.uint8)
    #img_data = data.get_fdata()
    print(data[100:110, 100:110,100])
    cv.imshow('brain', data[:,:,100])
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()