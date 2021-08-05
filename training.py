
# coding: utf-8

# In[1]:

''' SECTION 1: Some variables need to be specified
##
# '''

# This code is modifed from https://github.com/martinarjovsky/WassersteinGAN 
# get_ipython().magic(u'matplotlib inline')

''' SECTION 2: Call libraries
##
# '''
import os, io, h5py, math, datetime, sys

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import tensorflow as tf 
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import callbacks
from tensorflow.python.keras.utils.data_utils import Sequence

from skimage import io as skio

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

## THE NETWORKS ARE DEFINED HERE
sys.path.append('./networks-builder')   # The networks are defined here

import model            # Call the networks codes
import datautils        # Call the networks codes
import losses as loss   # Call the networks codes

# Call the networks codes
from builders.ProbUNet_GAN import generator_optimizer, discriminator_optimizer 
from builders.ProbUNet_GAN import discriminator_loss 
from builders.ProbUNet_GAN import ProbUNet2Prior_Y1Y2GAN

## FOR Data Augmentation
from albumentations import (
    Compose, 
    ShiftScaleRotate,
    Rotate,
    HorizontalFlip,
    VerticalFlip,
    RandomBrightnessContrast,
    RandomGamma,
    ElasticTransform,
    RGBShift,
    Normalize
)

np.seterr(divide='ignore', invalid='ignore')

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

# Input arguments for training
C0 = float(sys.argv[1])   # weight for Focal loss (i.e., background class)
C1 = float(sys.argv[2])   # weight for Focal loss (i.e., shrinking WMHs class)
C2 = float(sys.argv[3])   # weight for Focal loss (i.e., growing WMHs class)
C3 = float(sys.argv[4])   # weight for Focal loss (i.e., stable WMHs class)
FF = []
for iar in range(5,len(sys.argv)):
    FF.append(int(sys.argv[iar]))
print('C0:', C0)   # weight for Focal loss (i.e., background class)
print('C1:', C1)   # weight for Focal loss (i.e., shrinking WMHs class)
print('C2:', C2)   # weight for Focal loss (i.e., growing WMHs class)
print('C3:', C3)   # weight for Focal loss (i.e., stable WMHs class)
print('FF:', FF)   # For cross-validation purposes

# Memory growth of GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

''' SECTION 3: Define classes and functions
##
# '''

## Data augmentation if t1 and t2 MRI scans are used
class ImageSequence_2In(Sequence):
    def __init__(self, x1_set, x2_set, y_set, batch_size, augmentations_img):
        self.x1 = x1_set
        self.x2 = x2_set
        self.y = y_set
        self.batch_size = batch_size
        self.augment_img = augmentations_img

    def __len__(self):
        return int(np.ceil(len(self.x1) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x1 = self.x1[idx * self.batch_size:(idx + 1) * self.batch_size,:,:,:]
        batch_x2 = self.x2[idx * self.batch_size:(idx + 1) * self.batch_size,:,:,:]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size,:,:,:]

        actual_batch_size = batch_x1.shape[0]
        batch_x = np.concatenate((batch_x1,batch_x2), axis=0)
        batch_y = np.concatenate((batch_y,batch_y), axis=0)

        batch_image = []
        batch_label = []
        for x, y in zip(batch_x, batch_y):
            augmented = self.augment_img(image=x, mask=y)
            batch_image.append(augmented['image'])
            batch_label.append(augmented['mask'])

        batch_image1 = batch_image[:actual_batch_size]
        batch_image2 = batch_image[actual_batch_size:]
        batch_label_ = batch_label[:actual_batch_size]

        # Create arrays of image, label, and random noises
        batch_image1_array = np.asarray(batch_image1)
        batch_image2_array = np.asarray(batch_image2)
        batch_label__array = np.asarray(batch_label_)
        
        return [batch_image1_array, batch_image2_array], batch_label__array

''' SECTION 5: Training 4 different networks in 4-fold
##
# '''

n_label = 4     # number of labels
cost_func = loss.categorical_focal_loss([C0, C1, C2, C3])   # define cost function
weights = "C0-" + str(C0) + "_" + "C1-" + str(C1) + "_" + "C2-" + str(C2) + "_" + "C3-" + str(C3)

now = datetime.date.today()
currentDate = str(now).replace('-','')

logdir_name = "ProbUNet_wY1Y2GAN_Y2prior_32latent_FCL_" + weights + "_" + currentDate
    
print("n_label: ", n_label)
print("logdir_name: ", logdir_name)

## Specify where to save the trained model
save_file_name = logdir_name + "_fold"

## Specify the location of .txt files for accessing the training data
config_dir = 'training_dataset_server_fold'

# crop the image in 2 dimension (i.e., x and y), if needed
crop = 0
imageSizeOri = 256
imageSize = imageSizeOri - crop

# Training parameters
batch_size    = 16
num_epochs    = 64
shuffle_epoch = True
fold_list = FF

# Define the loss functions for the generator.
def generator_loss(misleading_labels, fake_logits, fake_imgs, real_imgs):
    bce = keras.losses.BinaryCrossentropy(from_logits=True)
    fcl = cost_func
    real_dem = layers.Lambda(lambda x : x[:,:,:,2:])(real_imgs)
    fake_dem = layers.Lambda(lambda x : x[:,:,:,2:])(fake_imgs)
    return bce(misleading_labels, fake_logits) + fcl(real_dem, fake_dem)

for fold in fold_list:
    save_filename_fold = save_file_name+str(fold)
    save_dirpath = "./models/" + save_filename_fold
    try:
        os.makedirs(save_dirpath)
    except OSError:
        if not os.path.isdir(save_dirpath):
            raise

    # Network parameters
    n_chn_gen = 1   # Number of input channel for generator (i.e., U-Net)
    n_chn_dsc = (n_chn_gen + 1) + n_label     # Number of input channel for discriminator
    # Note: +1 is for the follow-up data

    ## Create the DEP-UResNet
    backend.clear_session()

    discriminator = model.DiscriminatorSNGAN(
        encoder_filters=(32,64,128,256,512),
        downsample_rates=(2,2,2,2,2),
        filter_sizes=(3,3,3,3,3),
        n_downsamples=5,
        n_convs_per_block=4,
        conv_activation='relu',
        encoder_block_type='stride',
        input_shape=(imageSize, imageSize, n_chn_dsc),
        type='2D'
        )
    
    # Instantiate the ProbUNet_GAN model
    unet_gan = ProbUNet2Prior_Y1Y2GAN(
        discriminator=discriminator,
        num_filters=[64,128,256,512,1024],
        latent_dim=6,       # Choose the latent space's dimension
        discriminator_extra_steps=5,
        cost_function=cost_func,
        n_label=n_label,
        resolution_lvl=5,
        img_shape=(imageSize, imageSize, n_chn_gen),
        seg_shape=(imageSize, imageSize, n_label),
        downsample_signal=(2,2,2,2,2)
    )    

    # Compile the UNet_GAN3D model
    unet_gan.compile(
        prior_opt=generator_optimizer, 
        posterior_opt=generator_optimizer,
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )

    unet_gan.summary()
    print("DEP-UNetPP-GAN created!")

    ### NOTE:
    ## Please change the .txt files' names as you wish.
    ### Acronym:
    ## - 1tp      : 1st time point
    ## - 2tp      : 2nd time point
    ## - icv      : IntraCranial Volume
    ## - sl       : Stroke Lesions

    print("Reading data: FLAIR 1tp")
    data_list_flair_1tp = []
    f = open('./'+config_dir+'/flair_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_flair_1tp.append(line)
    data_list_flair_1tp = map(lambda s: s.strip('\n'), data_list_flair_1tp)

    print("Reading data: WMH subtracted coded")
    data_list_wsc_1tp = []
    f = open('./'+config_dir+'/wmh_subtracted_coded_2tp_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_wsc_1tp.append(line)
    data_list_wsc_1tp = map(lambda s: s.strip('\n'), data_list_wsc_1tp)

    print("Reading data: ICV 1tp")
    data_list_icv_1tp = []
    f = open('./'+config_dir+'/icv_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_icv_1tp.append(line)
    data_list_icv_1tp = map(lambda s: s.strip('\n'), data_list_icv_1tp)

    print("Reading data: SL 1tp")
    data_list_sl_1tp = []
    f = open('./'+config_dir+'/sl_cleaned_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_sl_1tp.append(line)
    data_list_sl_1tp = map(lambda s: s.strip('\n'), data_list_sl_1tp)

    print("Reading data: FLAIR 2tp")
    data_list_flair_2tp = []
    f = open('./'+config_dir+'/flair_2tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_flair_2tp.append(line)
    data_list_flair_2tp = map(lambda s: s.strip('\n'), data_list_flair_2tp)

    print("Reading data: ICV 2tp")
    data_list_icv_2tp = []
    f = open('./'+config_dir+'/icv_2tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_icv_2tp.append(line)
    data_list_icv_2tp = map(lambda s: s.strip('\n'), data_list_icv_2tp)
    
    print("Reading data: SL 2tp")
    data_list_sl_2tp = []
    f = open('./'+config_dir+'/sl_cleaned_2tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_sl_2tp.append(line)
    data_list_sl_2tp = map(lambda s: s.strip('\n'), data_list_sl_2tp)

    id = 0
    train_flair_1tp = None
    train_flair_2tp = None
    train_wscod_1tp = None
    for data, data_wsc, data_icv, data_sl, data_2tp, data_icv_2tp, data_sl_2tp in \
            zip(data_list_flair_1tp, data_list_wsc_1tp, data_list_icv_1tp, data_list_sl_1tp,
                data_list_flair_2tp, data_list_icv_2tp, data_list_sl_2tp):
        if os.path.isfile(data):
            print(data)
            print(data_wsc)
            print(data_icv)

            loaded_data_f_1tp = datautils.load_data(data)
            loaded_data_wsc_1tp = datautils.load_data(data_wsc)
            loaded_data_i_1tp = datautils.load_data(data_icv)
            loaded_data_f_2tp = datautils.load_data(data_2tp)
            loaded_data_i_2tp = datautils.load_data(data_icv_2tp)

            loaded_image_f_1tp = datautils.data_prep_load_2D(loaded_data_f_1tp)
            loaded_image_wsc_1tp = datautils.data_prep_load_2D(loaded_data_wsc_1tp)
            loaded_image_i_1tp = datautils.data_prep_load_2D(loaded_data_i_1tp)
            loaded_image_f_2tp = datautils.data_prep_load_2D(loaded_data_f_2tp)
            loaded_image_i_2tp = datautils.data_prep_load_2D(loaded_data_i_2tp)

            brain_flair_1tp = np.multiply(loaded_image_f_1tp, loaded_image_i_1tp)
            brain_flair_2tp = np.multiply(loaded_image_f_2tp, loaded_image_i_2tp)
            brain_wsc_1tp = np.multiply(loaded_image_wsc_1tp, loaded_image_i_1tp)
            
            if os.path.isfile(data_sl):
                print(data_sl)
                loaded_data_s_1tp  = datautils.load_data(data_sl)
                loaded_image_s_1tp = datautils.data_prep_load_2D(loaded_data_s_1tp)
                loaded_image_s_1tp = 1 - loaded_image_s_1tp
                brain_flair_1tp = np.multiply(brain_flair_1tp, loaded_image_s_1tp)
                brain_wsc_1tp = np.multiply(brain_wsc_1tp, loaded_image_s_1tp)

            if os.path.isfile(data_sl_2tp):
                print(data_sl_2tp)
                loaded_data_s_2tp  = datautils.load_data(data_sl_2tp)
                loaded_image_s_2tp = datautils.data_prep_load_2D(loaded_data_s_2tp)
                loaded_image_s_2tp = 1 - loaded_image_s_2tp
                brain_flair_2tp = np.multiply(brain_flair_2tp, loaded_image_s_2tp)

            norm_percentile = 0
            print("WSC 1tp [new] - max: " + str(np.max(brain_wsc_1tp)) + ", min: " + str(np.min(brain_wsc_1tp)))
            
            print("FLR 1tp [old] - mean: " + str(np.mean(brain_flair_1tp)) + ", std: " + str(np.std(brain_flair_1tp)))
            brain_flair_1tp = ((brain_flair_1tp - np.mean(brain_flair_1tp)) / np.std(brain_flair_1tp)) # normalise to zero mean unit variance 3D
            brain_flair_1tp = np.nan_to_num(brain_flair_1tp)
            print("FLR 1tp [new] - mean: " + str(np.mean(brain_flair_1tp)) + ", std: " + str(np.std(brain_flair_1tp)))

            print("FLR 2tp [old] - mean: " + str(np.mean(brain_flair_2tp)) + ", std: " + str(np.std(brain_flair_2tp)))
            brain_flair_2tp = ((brain_flair_2tp - np.mean(brain_flair_2tp)) / np.std(brain_flair_2tp)) # normalise to zero mean unit variance 3D
            brain_flair_2tp = np.nan_to_num(brain_flair_2tp)
            print("FLR 2tp [new] - mean: " + str(np.mean(brain_flair_2tp)) + ", std: " + str(np.std(brain_flair_2tp)))

            print("brain_flair_1tp.shape [ORI]: ", brain_flair_1tp.shape)
            print("brain_flair_2tp.shape [ORI]: ", brain_flair_2tp.shape)
            print("brain_wsc_1tp.shape [ORI]  : ", brain_wsc_1tp.shape)

            ## Cropping
            crop_rate = int(crop/2)
            brain_flair_1tp = brain_flair_1tp[:,crop_rate:imageSizeOri-crop_rate,crop_rate:imageSizeOri-crop_rate,:]
            brain_flair_2tp = brain_flair_2tp[:,crop_rate:imageSizeOri-crop_rate,crop_rate:imageSizeOri-crop_rate,:]
            brain_wsc_1tp = brain_wsc_1tp[:,crop_rate:imageSizeOri-crop_rate,crop_rate:imageSizeOri-crop_rate,:]

            print("brain_flair_1tp.shape [CROP]: ", brain_flair_1tp.shape)
            print("brain_flair_2tp.shape [CROP]: ", brain_flair_2tp.shape)
            print("brain_flair_1tp.shape [CROP]: ", brain_wsc_1tp.shape)

            print("brain_wsc_1tp[0] - BCK [ORI]: ", np.count_nonzero(brain_wsc_1tp == 0))
            print("brain_wsc_1tp[1] - SHR [ORI]: ", np.count_nonzero(brain_wsc_1tp == 1))
            print("brain_wsc_1tp[2] - GRW [ORI]: ", np.count_nonzero(brain_wsc_1tp == 2))
            print("brain_wsc_1tp[3] - STB [ORI]: ", np.count_nonzero(brain_wsc_1tp == 3))

            print("ALL LOADED")
            if id == 0:
                train_flair_1tp = brain_flair_1tp
                train_flair_2tp = brain_flair_2tp
                train_wscod_1tp = brain_wsc_1tp
            else:
                train_flair_1tp = np.concatenate((train_flair_1tp,brain_flair_1tp), axis=0)
                train_flair_2tp = np.concatenate((train_flair_2tp,brain_flair_2tp), axis=0)
                train_wscod_1tp = np.concatenate((train_wscod_1tp,brain_wsc_1tp), axis=0)
                print("train_flair_1tp SHAPE: " + str(train_flair_1tp.shape) + " | " + str(id))
            id += 1

    # In[17]:
    print("train_flair_1tp -> ", train_flair_1tp.shape)
    print("train_flair_2tp -> ", train_flair_2tp.shape)
    print("train_wscod_1tp -> ", train_wscod_1tp.shape)

    ''' ---- Split and shuffle the LOADED TRAINING DATA 
    '''
    ## Split the data for training and validation
    flair_1tp_train, flair_1tp_val, flair_2tp_train, flair_2tp_val, wscod_1tp_train, wscod_1tp_val = \
        train_test_split(train_flair_1tp, train_flair_2tp, train_wscod_1tp, test_size=0.02, random_state=42)

    ## Shuffle the training data
    indices = np.array(range(flair_1tp_train.shape[0]))
    np.random.seed(42)
    np.random.shuffle(indices)
    flair_1tp_train = flair_1tp_train[indices]
    flair_2tp_train = flair_2tp_train[indices]
    wscod_1tp_train = wscod_1tp_train[indices]
    nt, _, _, _ = flair_1tp_train.shape
    nt_new = int(nt / batch_size) * batch_size
    print("OLD: ", len(indices), " vs. ", nt_new)
    print("flair_1tp_train [old] -> ", flair_1tp_train.shape)
    print("flair_2tp_train [old] -> ", flair_2tp_train.shape)
    print("wscod_1tp_train [old] -> ", wscod_1tp_train.shape)
    flair_1tp_train = flair_1tp_train[:nt_new,:,:,:]
    flair_2tp_train = flair_2tp_train[:nt_new,:,:,:]
    wscod_1tp_train = wscod_1tp_train[:nt_new,:,:,:]
    print("flair_1tp_train [new] -> ", flair_1tp_train.shape)
    print("flair_2tp_train [new] -> ", flair_2tp_train.shape)
    print("wscod_1tp_train [new] -> ", wscod_1tp_train.shape)

    # Shuffle the validation data
    indices = np.array(range(flair_1tp_val.shape[0]))
    np.random.seed(42)
    np.random.shuffle(indices)
    flair_1tp_val = flair_1tp_val[indices]
    flair_2tp_val = flair_2tp_val[indices]
    wscod_1tp_val = wscod_1tp_val[indices]
    flair_val = [flair_1tp_val, flair_2tp_val]

    ## Check the size of training and validation data
    print("flair_1tp_train -> ", flair_1tp_train.shape)
    print("flair_2tp_train -> ", flair_2tp_train.shape)
    print("wscod_1tp_train -> ", wscod_1tp_train.shape)
    print("flair_1tp_val -> ", flair_1tp_val.shape)
    print("flair_2tp_val -> ", flair_2tp_val.shape)
    print("wscod_1tp_val -> ", wscod_1tp_val.shape)

    ## Check the range values of the training and validation data 
    print("flair_1tp_train - mean: " + str(np.mean(flair_1tp_train)) + ", std: " + str(np.std(flair_1tp_train)))
    print("flair_2tp_train - mean: " + str(np.mean(flair_2tp_train)) + ", std: " + str(np.std(flair_2tp_train)))
    print("wscod_1tp_train - max: " + str(np.max(wscod_1tp_train)) + ", min: " + str(np.min(wscod_1tp_train)))
    print("flair_1tp_val - mean: " + str(np.mean(flair_1tp_val)) + ", std: " + str(np.std(flair_1tp_val)))
    print("flair_2tp_val - mean: " + str(np.mean(flair_2tp_val)) + ", std: " + str(np.std(flair_2tp_val)))
    print("wscod_1tp_val - max: " + str(np.max(wscod_1tp_val)) + ", min: " + str(np.min(wscod_1tp_val)))
    wscod_1tp_train = wscod_1tp_train.astype(int)
    wscod_1tp_val = wscod_1tp_val.astype(int)
    # print("wscod_1tp_train -> ", wscod_1tp_train.shape)
    # print("wscod_1tp_val -> ", wscod_1tp_val.shape)

    print("Convert to 1-hot representation:")
    wscod_1tp_train_label = datautils.convert_to_1hot(wscod_1tp_train, n_label)
    wscod_1tp_val_label = datautils.convert_to_1hot(wscod_1tp_val, n_label)
    wscod_1tp_train_label = np.squeeze(wscod_1tp_train_label.astype(float))
    wscod_1tp_val_label = np.squeeze(wscod_1tp_val_label.astype(float))
    print("wscod_1tp_train_label -> ", wscod_1tp_train_label.shape)
    print("wscod_1tp_val_label -> ", wscod_1tp_val_label.shape)

    # Instantiate augments
    # we can apply as many augments we want and adjust the values accordingly
    # here I have chosen the augments and their arguments at random
    trans_data = Compose([
            ShiftScaleRotate(shift_limit=0.25, scale_limit=0.2, rotate_limit=90),
            HorizontalFlip(),
            VerticalFlip(),
            ElasticTransform(alpha=imageSize*3, sigma=imageSize*0.07, alpha_affine=imageSize*0.09)
    ])

    # Image generator for data augmentation
    train_gen = ImageSequence_2In(flair_1tp_train, flair_2tp_train, wscod_1tp_train_label, batch_size, augmentations_img=trans_data)

    print('Training..')
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    tensorboard = callbacks.TensorBoard(log_dir='./logdir/'+logdir_name+'/asFold'+str(fold))
    model_checkpoint = callbacks.ModelCheckpoint(
            filepath=save_dirpath+"/checkpoint-best-{epoch:03d}-{val_loss:.4f}.hdf5",
            monitor='val_loss',
            mode='min',
            save_best_only=True)

    ''' Training iterations
    '''
    for ep in range(num_epochs):
        # Print iteration's information
        print("\n---\nEPOCH using FOR-IN: " + str(ep+1) + "/" + str(int(num_epochs)))

        history_callback = unet_gan.fit(train_gen,
            epochs=ep+1,
            initial_epoch=ep,
            validation_batch_size=1,
            validation_data=(flair_val, wscod_1tp_val_label),
            callbacks=[tensorboard, reduce_lr, model_checkpoint])

    ## SAVING TRAINED MODEL AND WEIGHTS
    ## Save models
    unet_gan.save(save_dirpath)

