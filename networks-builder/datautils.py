# https://github.com/MrGiovanni/UNetPlusPlus/blob/master/keras/segmentation_models/utils.py

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def view_image(ds, name):
    image, label = next(iter(ds)) # extract 1 batch from the dataset
    n, x, y, c = image.shape

    fig = plt.figure(figsize=(64, 8))
    for i in range(n):
        ax = fig.add_subplot(2, n, i+1, xticks=[], yticks=[])
        img = image[i]
        b_channel = np.zeros((x, y, 1)).astype(float)
        ax.imshow(np.concatenate((img, b_channel), axis=-1))
    for i in range(n):
        ax = fig.add_subplot(2, n, i+17, xticks=[], yticks=[])
        lbl = label[i,:,:,0]
        # print("non_zero (", i, "): ", np.count_nonzero(lbl))
        ax.imshow(lbl)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.savefig(name)
    plt.close(fig)

# Class to load NifTi (.nii) data
class load_data(object):
    # Load NII data
    def __init__(self, image_name):
        # Read nifti image
        nim = nib.load(image_name)
        image = nim.get_data()
        affine = nim.affine
        self.image = image
        self.affine = affine
        self.dt = nim.header['pixdim'][4]
        self.pixdim = nim.header['pixdim'][1:4]

# Function to prepare the data after opening from .nii file
def data_prep_load_2D(image_data):
    # Extract the 2D slices from the cardiac data
    image = image_data.image
    images = []
    for z in range(image.shape[2]):
        image_slice = image[:, :, z]
        images += [image_slice]
    images = np.array(images, dtype='float32')

    # Both theano and caffe requires input array of dimension (N, C, H, W)
    # TensorFlow (N, H, W, C)
    # Add the channel dimension, swap width and height dimension
    images = np.expand_dims(images, axis=3)

    return images

# Function to prepare the data after opening from .nii file
def data_prep_load_3D(image_data):
    # Extract the 2D slices from the cardiac data
    image = image_data.image
    images = np.array(image, dtype='float32')

    return images

# Function to prepare the data before saving to .nii file
def data_prep_save(image_data):
#     print(image_data.shape)
    image_data = np.squeeze(image_data)
    output_img = np.swapaxes(image_data, 0, 2)
    output_img = np.rot90(output_img)
    output_img = output_img[::-1,...]   # flipped

    return output_img

def convert_to_1hot(label, n_class):
    # Convert a label map (N x H x W x 1) into a one-hot representation (N x H x W x C)
    print(" --> SIZE = " + str(label.shape))
    print(" --> MAX = " + str(np.max(label)))
    print(" --> MIN = " + str(np.min(label)))
    label_flat = label.flatten().astype(int)
    n_data = len(label_flat)
    print(" --> SIZE = " + str(label_flat.shape))
    print(" --> LEN = " + str(n_data))
    label_1hot = np.zeros((n_data, n_class), dtype='int16')
    print(" --> 1HOT-SIZE = " + str(label_1hot.shape))
    label_1hot[range(n_data), label_flat] = 1
    label_1hot = label_1hot.reshape((label.shape[0], label.shape[1], label.shape[2], label.shape[3], n_class))

    return label_1hot


# Convert a label map (N x H x W x C) from a one-hot representation (N x H x W x 1)
def convert_from_1hot(label, to_float=False):
    N, H, W, C = label.shape
    label_flat = label.reshape((N * H * W, C))
    n_data = len(label_flat)

    if to_float:
        label_n_class = np.zeros((n_data, 1), dtype='float32')
        max_class = np.argmax(label_flat, axis=1)
        label_n_class[range(n_data), 0] = label_flat[range(n_data), max_class]
    else:
        label_n_class = np.zeros((n_data, 1), dtype='uint8')
        label_n_class[range(n_data), 0] = np.argmax(label_flat, axis=1)

    label_n_class = np.squeeze(label_n_class.reshape((N, H, W, 1)))

    return label_n_class