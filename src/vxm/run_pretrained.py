import voxelmorph as vxm
import numpy as np
import nibabel as nib


if __name__ == '__main__':

    # first read in the dataset 
    img = nib.load('/home/juseonghan/mia/brain_segmentation_proj/data/undelineated/volumes/050_T1.nii.gz')
    #img = nib.load('/home/juseonghan/mia/brain_segmentation_proj/data/delineated/volumes/03_ANAT_N4_MNI_fcm.nii.gz')
    img_np = np.array(img.dataobj)
    x_train = img_np[0:150, :,:]
    x_val = img_np[150:, :,:]
    vol_shape = x_train.shape[1:]

    nb_features = [
        [32, 32, 32, 32],         # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
    ]

    vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)
    vxm_model.load_weights('shapes-dice-vel-3-res-8-16-32-256f.h5')

    val_generator = vxm_data_generator(x_val, batch_size = 1)
    val_input, _ = next(val_generator)
    val_pred = vxm_model.predict(val_input)

    images = [img[0, :, :, 0] for img in val_input + val_pred] 
    titles = ['moving', 'fixed', 'moved', 'flow']
    ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

    # visualize flow
    flow = val_pred[1].squeeze()[::3,::3]
    ne.plot.flow([flow], width=5)