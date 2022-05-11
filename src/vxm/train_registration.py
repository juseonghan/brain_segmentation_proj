import voxelmorph as vxm
import numpy as np
import neurite as ne
import os, sys     
import tensorflow as tf
import nibabel as nib
import matplotlib.pyplot as plt


def vxm_data_generator(x_data, batch_size=32):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data.shape[1:] # extract data shape
    ndims = len(vol_shape)
    
    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        
        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare 
        # the resulting moved image with the fixed image. 
        # we also wish to penalize the deformation field. 
        outputs = [fixed_images, zero_phi]
        
        yield (inputs, outputs)


def plot_history(hist, loss_name='loss'):
    # Simple function to plot training history.
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

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

    # # extract some brains
    # nb_vis = 5
    # idx = np.random.randint(0, x_train.shape[0], [5,])
    # example_digits = [f for f in x_train[idx, ...]]

    # # visualize
    # ne.plot.slices(example_digits, cmaps=['gray'], do_colorbars=True)

    # vxm 
    vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)

    # losses and loss weights
    losses = ['mse', vxm.losses.Grad('l2').loss]
    loss_weights = [1, 0.01]

    vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=losses, loss_weights=loss_weights)
    # let's test it
    train_generator = vxm_data_generator(x_train, batch_size=8)
    in_sample, out_sample = next(train_generator)

    # visualize
    images = [img[0, :, :, 0] for img in in_sample + out_sample]
    titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']
    ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

    hist = vxm_model.fit_generator(train_generator, epochs=400, steps_per_epoch=5, verbose=2)

    plot_history(hist)

    val_generator = vxm_data_generator(x_val, batch_size = 1)
    val_input, _ = next(val_generator)

    val_pred = vxm_model.predict(val_input)

    # visualize registration
    images = [img[0, :, :, 0] for img in val_input + val_pred] 
    titles = ['moving', 'fixed', 'moved', 'flow']
    ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

    # visualize flow
    flow = val_pred[1].squeeze()[::3,::3]
    ne.plot.flow([flow], width=5)