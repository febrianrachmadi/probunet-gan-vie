
# coding: utf-8

''' SECTION 1: Call libraries
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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

np.seterr(divide='ignore', invalid='ignore')

## THE NETWORKS ARE DEFINED HERE
sys.path.append('./networks-builder')   # The networks are defined here

import model            # Call the networks codes
import datautils        # Call the networks codes
import losses as loss   # Call the networks codes

# Call the networks codes
from builders.ProbUNet_GAN import generator_optimizer, discriminator_optimizer 
from builders.ProbUNet_GAN import discriminator_loss 
from builders.ProbUNet_GAN import ProbUNet2Prior_Y1Y2GAN

''' SECTION 2: Some variables need to be specified
##
# '''

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

sampling_number = 4    # for inference, i.e., how many samples created for one data
n_label = 4     # number of labels
print("n_label: ", n_label)

# Network parameters
n_chn_gen = 1   # Number of input channel for generator (i.e., U-Net)
n_chn_dsc = n_chn_gen + n_label + 1     # Number of input channel for discriminator
# Note: +1 is for the follow-up data

# crop the image in 2 dimension (i.e., x and y), if needed
crop = 0
imageSizeOri = 256
imageSize = imageSizeOri - crop

## Specify the location of .txt files for accessing the training data
config_dir = 'testing_dataset_server_fold'

## Specify where the trained model
saved_model_name = "ProbUNet_wY1Y2GAN_Y2prior_normal_FCL_C0-0.25_C1-0.75_C2-0.75_C3-0.5_20210614_fold"

# ---- CREATE RESULT DIRs
dirOutputPath = './outputs/'
saving_filename_dir = 'ProbUNet_wY1Y2GAN_Y2prior_FCL.' + saved_model_name

''' SECTION 3: Define classes and functions
##
# '''

# Define the loss functions for the generator.
def generator_loss(misleading_labels, fake_logits, fake_imgs, real_imgs):
    bce = keras.losses.BinaryCrossentropy(from_logits=True)
    fcl = cost_func
    real_dem = layers.Lambda(lambda x : x[:,:,:,1:])(real_imgs)
    fake_dem = layers.Lambda(lambda x : x[:,:,:,1:])(fake_imgs)
    return bce(misleading_labels, fake_logits) + fcl(real_dem, fake_dem)


''' SECTION 5: Testing 4 different networks in 4-fold
##
# '''

dirOutput = dirOutputPath + saving_filename_dir
print("dirOutput: ", dirOutput)
try:
    os.makedirs(dirOutput)
except OSError:
    if not os.path.isdir(dirOutput):
        raise

# Save the name of the data
name_all = []

## DSC Evaluation
mean_dsc_all = []
std_dsc_all = []

## VOL Evaluation
mean_vol_all = []
std_vol_all = []

C0 = 0.25   # weight for Focal loss (i.e., background class)
C1 = 0.75   # weight for Focal loss (i.e., shrinking WMHs class)
C2 = 0.75   # weight for Focal loss (i.e., growing WMHs class)
C3 = 0.50   # weight for Focal loss (i.e., stable WMHs class)

for fold in [1,2,3,4]:
    ## Create the DEP-UResNet
    backend.clear_session()

    ## Create the discriminator
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
    cost_func = loss.categorical_focal_loss(alpha=[[C0, C1, C2, C3]], gamma=2)
    my_network = ProbUNet2Prior_Y1Y2GAN(
        discriminator=discriminator,        # Put the discriminator here
        num_filters=[64,128,256,512,1024],
        latent_dim=6,
        discriminator_extra_steps=5,
        cost_function=cost_func,
        n_label=n_label,
        resolution_lvl=5,
        img_shape=(imageSize, imageSize, n_chn_gen),
        seg_shape=(imageSize, imageSize, n_label),
        downsample_signal=(2,2,2,2,2)
    )    

    # Compile the model
    my_network.compile(
        prior_opt=generator_optimizer, 
        posterior_opt=generator_optimizer,
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )

    my_network.built = True
    my_network.load_weights('./models/' + saved_model_name + str(fold) + '/best.hdf5')
    print("Generator's weights loaded!")

    ### NOTE:
    ## Please change the .txt files' names as you wish.
    ## You also can change probability map into irregularity map
    ### Acronym:
    ## - 1tp      : 1st time point
    ## - 2tp      : 2nd time point
    ## - wmh_prob : WMH's probability/irregularity map
    ## - icv      : IntraCranial Volume
    ## - sl       : Stroke Lesions

    # ---- LOAD Testing DATA
    print("Reading data: FLAIR 1tp")
    data_list_flair_1tp = []
    f = open('./'+config_dir+'/flair_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_flair_1tp.append(line)
    data_list_flair_1tp = map(lambda s: s.strip('\n'), data_list_flair_1tp)

    print("Reading data: T1W 1tp")
    data_list_t1w_1tp = []
    f = open('./'+config_dir+'/t1w_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_t1w_1tp.append(line)
    data_list_t1w_1tp = map(lambda s: s.strip('\n'), data_list_t1w_1tp)

    print("Reading data: ICV 1tp")
    data_list_icv_1tp = []
    f = open('./'+config_dir+'/icv_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_icv_1tp.append(line)
    data_list_icv_1tp = map(lambda s: s.strip('\n'), data_list_icv_1tp)

    print("Reading data: WMH 1tp")
    data_list_wmh_1tp = []
    f = open('./'+config_dir+'/wmh_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_wmh_1tp.append(line)
    data_list_wmh_1tp = map(lambda s: s.strip('\n'), data_list_wmh_1tp)

    print("Reading data: SL 1tp")
    data_list_sl_1tp = []
    f = open('./'+config_dir+'/sl_cleaned_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_sl_1tp.append(line)
    data_list_sl_1tp = map(lambda s: s.strip('\n'), data_list_sl_1tp)

    print("Reading data: WMH 2tp")
    data_list_wmh_2tp = []
    f = open('./'+config_dir+'/wmh_2tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_wmh_2tp.append(line)
    data_list_wmh_2tp = map(lambda s: s.strip('\n'), data_list_wmh_2tp)

    print("Reading data: WMH evolution coded")
    data_list_code_2tp = []
    f = open('./'+config_dir+'/wmh_subtracted_coded_2tp_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_code_2tp.append(line)
    data_list_code_2tp = map(lambda s: s.strip('\n'), data_list_code_2tp)

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

    print("Reading data: NAME")
    data_list_name = []
    f = open('./'+config_dir+'/name_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_name.append(line)
    data_list_name = map(lambda s: s.strip('\n'), data_list_name)

    id = 0
    train_flair_1tp = np.zeros((1,256,256,1))
    for data, t1w, name, icv1, sl1, wmh1, wmh2, code2, icv2, sl2 in \
            zip(data_list_flair_1tp, data_list_t1w_1tp, data_list_name, data_list_icv_1tp, data_list_sl_1tp, \
            data_list_wmh_1tp, data_list_wmh_2tp, data_list_code_2tp, data_list_icv_2tp, data_list_sl_2tp):
        if os.path.isfile(data):
            name_all.append(name)

            # Print data location
            print("name : ", name)
            print("flair : ", data)
            print("t1w  : ", t1w)
            print("icv1 : ", icv1)
            print("sl1  : ", sl1)
            print("wmh1 : ", wmh1)
            print("wmh2 : ", wmh2)
            print("code2: ", code2)
            print("icv2 : ", icv2)
            print("sl2  : ", sl2)

            # Load nifti data
            loaded_data_f_1tp = datautils.load_data(data)
            loaded_data_t_1tp = datautils.load_data(t1w)
            loaded_data_i_1tp = datautils.load_data(icv1)
            loaded_data_w_1tp = datautils.load_data(wmh1)
            loaded_data_w_2tp = datautils.load_data(wmh2)
            loaded_data_i_2tp = datautils.load_data(icv2)
            loaded_data_c_2tp = datautils.load_data(code2)

            # Prepare the loaded nifti data
            loaded_image_f_1tp = datautils.data_prep_load_2D(loaded_data_f_1tp)
            loaded_image_t_1tp = datautils.data_prep_load_2D(loaded_data_t_1tp)
            loaded_image_i_1tp = datautils.data_prep_load_2D(loaded_data_i_1tp)
            loaded_image_w_1tp = datautils.data_prep_load_2D(loaded_data_w_1tp)
            loaded_image_w_2tp = datautils.data_prep_load_2D(loaded_data_w_2tp)
            loaded_image_i_2tp = datautils.data_prep_load_2D(loaded_data_i_2tp)
            loaded_image_c_2tp = datautils.data_prep_load_2D(loaded_data_c_2tp)

            # Exclude non-brain tissues
            brain_flair_1tp = np.multiply(loaded_image_f_1tp, loaded_image_i_1tp)
            brain_t1w_1tp = np.multiply(loaded_image_t_1tp, loaded_image_i_1tp)
            brain_wmh_1tp = np.multiply(loaded_image_w_1tp, loaded_image_i_1tp)
            brain_wmh_2tp = np.multiply(loaded_image_w_2tp, loaded_image_i_2tp)
            brain_cod_2tp = np.multiply(loaded_image_c_2tp, loaded_image_i_2tp)
            
            # Exclude Stroke Lesions (SL) tissues on 1st time point data
            icv_and_sl_mask_1tp = loaded_image_i_1tp
            if os.path.isfile(sl1):
                print(sl1)
                loaded_data_s_1tp  = datautils.load_data(sl1)
                loaded_image_s_1tp = datautils.data_prep_load_2D(loaded_data_s_1tp)
                loaded_image_s_1tp = 1 - loaded_image_s_1tp
                brain_flair_1tp = np.multiply(brain_flair_1tp, loaded_image_s_1tp)
                brain_t1w_1tp = np.multiply(brain_t1w_1tp, loaded_image_s_1tp)
                brain_wmh_1tp = np.multiply(brain_wmh_1tp, loaded_image_s_1tp)
                icv_and_sl_mask_1tp = np.multiply(icv_and_sl_mask_1tp, loaded_image_s_1tp)
                brain_cod_2tp = np.multiply(brain_cod_2tp, loaded_image_s_1tp)

            # Exclude Stroke Lesions (SL) tissues on 2nd time point data
            icv_and_sl_mask_2tp = loaded_image_i_2tp
            if os.path.isfile(sl2):
                loaded_data_s_2tp  = datautils.load_data(sl2)
                loaded_image_s_2tp = datautils.data_prep_load_2D(loaded_data_s_2tp)
                loaded_image_s_2tp = 1 - loaded_image_s_2tp
                brain_wmh_2tp = np.multiply(brain_wmh_2tp, loaded_image_s_2tp)
                icv_and_sl_mask_2tp = np.multiply(loaded_image_i_2tp, loaded_image_s_2tp) 
                brain_cod_2tp = np.multiply(brain_cod_2tp, loaded_image_s_2tp)
            
            print("FLR 1tp [old] - mean: ", np.mean(brain_flair_1tp), ", std: ", np.std(brain_flair_1tp))
            brain_flair_1tp = ((brain_flair_1tp - np.mean(brain_flair_1tp)) / np.std(brain_flair_1tp)) # normalise to zero mean unit variance 3D
            brain_flair_1tp = np.nan_to_num(brain_flair_1tp)
            print("FLR 1tp [new] - mean: ", np.mean(brain_flair_1tp), ", std: ", np.std(brain_flair_1tp))
            print("FLR 1tp SHAPE: ", brain_flair_1tp.shape)

            print("T1W 1tp [old] - mean: ", np.mean(brain_t1w_1tp), ", std: ", np.std(brain_t1w_1tp))
            brain_t1w_1tp = ((brain_t1w_1tp - np.mean(brain_t1w_1tp)) / np.std(brain_t1w_1tp)) # normalise to zero mean unit variance 3D
            brain_t1w_1tp = np.nan_to_num(brain_t1w_1tp)
            print("T1W 1tp [new] - mean: ", np.mean(brain_t1w_1tp), ", std: ", np.std(brain_t1w_1tp))
            print("T1W 1tp SHAPE: ", brain_t1w_1tp.shape)
            
            test_flair_t1w = brain_flair_1tp
            print("FLR-T1W 1tp SHAPE: ", test_flair_t1w.shape)

            # Create output directories for each data
            dirOutData = dirOutput + '/' + str(name)
            print("dirOutData -> ", dirOutData)
            try:
                os.makedirs(dirOutData)
            except OSError:
                if not os.path.isdir(dirOutData):
                    raise

            ''' Starting from here, the codes are used to evaluate all metrics used in the manuscript.
            '''
            seg_result = None
            seg_result_shrink = None
            seg_result_grow   = None
            seg_result_stable = None
            dsc_per_sample = []
            vol_per_sample = []
            for si in range(sampling_number):
                print("")
                print("Sample #: ", si)

                ## Inference
                output_img_pred = my_network.predict(test_flair_t1w, batch_size=16)
                
                if si == 0 and seg_result is None:
                    seg_result = output_img_pred
                else:
                    seg_result = seg_result + output_img_pred 

                print("")
                print("output_img_pred.shape    : ", output_img_pred.shape)
                output_img_pred_lbl = datautils.convert_from_1hot(output_img_pred)
                print("output_img_pred_lbl.shape: ", output_img_pred_lbl.shape)

                ## Evaluate the "Volumetric Changes" in ml
                # print("WMH volume in the 1st time point")
                wmh_mask = brain_wmh_1tp
                wmh_from_iam_1tp = np.multiply(icv_and_sl_mask_1tp, wmh_mask)
                vol_1tp_mm3 = np.count_nonzero(wmh_from_iam_1tp) * np.prod(loaded_data_f_1tp.pixdim)
                vol_1tp__ml = vol_1tp_mm3 / 1000
                print("VOL (vol_1tp__ml): ", "{:.4f}".format(vol_1tp__ml))
                
                # print("WMH volume in the 2nd time point")
                wmh_mask = brain_wmh_2tp
                wmh_from_iam_2tp = np.multiply(icv_and_sl_mask_2tp, wmh_mask)
                vol_2tp_mm3 = np.count_nonzero(wmh_from_iam_2tp) * np.prod(loaded_data_f_1tp.pixdim)
                vol_2tp__ml = vol_2tp_mm3 / 1000
                print("VOL (vol_2tp__ml): ", "{:.4f}".format(vol_2tp__ml))
                
                # print("OUTPUT (predicted) WMH volume of the 2nd time point")
                wmh_mask = np.zeros(output_img_pred_lbl.shape)
                wmh_mask[output_img_pred_lbl >= 2] = 1
                wmh_from_out_2tp = wmh_mask
                vol_out_mm3 = np.count_nonzero(wmh_from_out_2tp) * np.prod(loaded_data_f_1tp.pixdim)
                vol_out__ml = vol_out_mm3 / 1000
                print("VOL (vol_out__ml): ", "{:.4f}".format(vol_out__ml))

                err_vol = vol_2tp__ml - vol_out__ml 
                print("ERR of VOL       : ", "{:.4f}".format(err_vol))
                print("---")

                ## Spatial Dynamic WMH evolution
                wmh_change_mask_fake = output_img_pred_lbl
                wmh_change_mask_real = np.squeeze(brain_cod_2tp).astype(int)
                
                smooth = 1e-7
                # dice_1: Dice for Shrinking WMH
                k = 1
                dice_1 = (np.count_nonzero(wmh_change_mask_fake[wmh_change_mask_real == k] == k)*2.0 + smooth) / \
                            (smooth + np.count_nonzero(wmh_change_mask_real[wmh_change_mask_real == k] == k) + \
                            np.count_nonzero(wmh_change_mask_fake[wmh_change_mask_fake == k] == k))

                # dice_1: Dice for Growing WMH
                k = 2
                dice_2 = (np.count_nonzero(wmh_change_mask_fake[wmh_change_mask_real == k] == k)*2.0 + smooth) / \
                            (smooth + np.count_nonzero(wmh_change_mask_real[wmh_change_mask_real == k] == k) + \
                            np.count_nonzero(wmh_change_mask_fake[wmh_change_mask_fake == k] == k))

                # dice_1: Dice for Stable WMH
                k = 3
                dice_3 = (np.count_nonzero(wmh_change_mask_fake[wmh_change_mask_real == k] == k)*2.0 + smooth) / \
                            (smooth + np.count_nonzero(wmh_change_mask_real[wmh_change_mask_real == k] == k) + \
                            np.count_nonzero(wmh_change_mask_fake[wmh_change_mask_fake == k] == k))
                
                avg_dices = (dice_1 + dice_2 + dice_3) / 3
                print(
                    "DSC (1HOT) - Shrink: ", "{:.4f}".format(dice_1),
                    ", Grow: ", "{:.4f}".format(dice_2),
                    ", Stable: ", "{:.4f}".format(dice_3),
                    " || AVG: ", "{:.4f}".format(avg_dices))

                ## SPATIAL WMH evolution -- ONLY SHRINKING -- 
                wmh_change_mask_fake = output_img_pred
                wmh_change_mask_fake_temp = np.delete(wmh_change_mask_fake, [2, 3], 3)
                wmh_change_mask_fake_temp = datautils.convert_from_1hot(wmh_change_mask_fake_temp)
                wmh_change_mask_fake_shrk = wmh_change_mask_fake_temp
                
                wmh_change_mask_real_temp = np.squeeze(brain_cod_2tp).astype(int)

                wmh_change_mask_real_temp[wmh_change_mask_real_temp == 1] = 1
                wmh_change_mask_real_temp[wmh_change_mask_real_temp == 2] = 0
                wmh_change_mask_real_temp[wmh_change_mask_real_temp == 3] = 0
                
                smooth = 1e-7
                # dice_1: Dice for SHRINKING WMH
                k = 1
                dice_1_shrink = (np.count_nonzero(wmh_change_mask_fake_temp[wmh_change_mask_real_temp == k] == k)*2.0 + smooth) / \
                            (smooth + np.count_nonzero(wmh_change_mask_real_temp[wmh_change_mask_real_temp == k] == k) + \
                            np.count_nonzero(wmh_change_mask_fake_temp[wmh_change_mask_fake_temp == k] == k))
                
                ## SPATIAL WMH evolution -- ONLY GROWING -- 
                wmh_change_mask_fake = output_img_pred
                wmh_change_mask_fake_temp = np.delete(wmh_change_mask_fake, [1, 3], 3)
                wmh_change_mask_fake_temp = datautils.convert_from_1hot(wmh_change_mask_fake_temp)
                wmh_change_mask_fake_grow = wmh_change_mask_fake_temp

                wmh_change_mask_real_temp = np.squeeze(brain_cod_2tp).astype(int)

                wmh_change_mask_real_temp[wmh_change_mask_real_temp == 1] = 0
                wmh_change_mask_real_temp[wmh_change_mask_real_temp == 3] = 0
                wmh_change_mask_real_temp[wmh_change_mask_real_temp == 2] = 1
                
                smooth = 1e-7
                # dice_1: Dice for Growing WMH
                k = 1
                dice_1_grow = (np.count_nonzero(wmh_change_mask_fake_temp[wmh_change_mask_real_temp == k] == k)*2.0 + smooth) / \
                            (smooth + np.count_nonzero(wmh_change_mask_real_temp[wmh_change_mask_real_temp == k] == k) + \
                            np.count_nonzero(wmh_change_mask_fake_temp[wmh_change_mask_fake_temp == k] == k))
                
                ## SPATIAL WMH evolution -- ONLY STABLE -- 
                wmh_change_mask_fake = output_img_pred
                wmh_change_mask_fake_temp = np.delete(wmh_change_mask_fake, [1, 2], 3)
                wmh_change_mask_fake_temp = datautils.convert_from_1hot(wmh_change_mask_fake_temp)
                wmh_change_mask_fake_stab = wmh_change_mask_fake_temp

                wmh_change_mask_real_temp = np.squeeze(brain_cod_2tp).astype(int)

                wmh_change_mask_real_temp[wmh_change_mask_real_temp == 1] = 0
                wmh_change_mask_real_temp[wmh_change_mask_real_temp == 2] = 0
                wmh_change_mask_real_temp[wmh_change_mask_real_temp == 3] = 1

                if si == 0 and seg_result_grow is None:
                    seg_result_shrink = wmh_change_mask_fake_shrk
                    seg_result_grow = wmh_change_mask_fake_grow
                    seg_result_stable = wmh_change_mask_fake_stab
                else:
                    seg_result_shrink = seg_result_shrink + wmh_change_mask_fake_shrk 
                    seg_result_grow = seg_result_grow + wmh_change_mask_fake_grow 
                    seg_result_stable = seg_result_stable + wmh_change_mask_fake_stab 
                
                smooth = 1e-7
                # dice_1: Dice for Growing WMH
                k = 1
                dice_1_stable = (np.count_nonzero(wmh_change_mask_fake_temp[wmh_change_mask_real_temp == k] == k)*2.0 + smooth) / \
                            (smooth + np.count_nonzero(wmh_change_mask_real_temp[wmh_change_mask_real_temp == k] == k) + \
                            np.count_nonzero(wmh_change_mask_fake_temp[wmh_change_mask_fake_temp == k] == k))

                dice_1hot_avg = (dice_1_shrink + dice_1_grow + dice_1_stable) / 3
                print(
                    "DSC (ONLY) - Shrink: ", "{:.4f}".format(dice_1_shrink),
                    ", Grow: ", "{:.4f}".format(dice_1_grow),
                    ", Stable: ", "{:.4f}".format(dice_1_stable),
                    " || AVG: ", "{:.4f}".format(dice_1hot_avg))

                seg = [
                    dice_1, dice_2, dice_3, avg_dices,
                    dice_1_shrink, dice_1_grow, dice_1_stable, dice_1hot_avg]
                dsc_per_sample.append(seg)

                ## VOLUME WMH evolution -- NO SHRINKING -- 
                wmh_change_mask_fake = output_img_pred
                wmh_change_mask_fake_temp = np.delete(wmh_change_mask_fake, 1, 3)
                wmh_change_mask_fake_temp = datautils.convert_from_1hot(wmh_change_mask_fake_temp)
                
                vol_out_mm3 = np.count_nonzero(wmh_change_mask_fake_temp) * np.prod(loaded_data_f_1tp.pixdim)
                vol_out__ml_trsh_noShrink = vol_out_mm3 / 1000
                err_vol_trsh_noShrink = vol_2tp__ml - vol_out__ml_trsh_noShrink

                ## VOLUME WMH evolution -- NO SHRINKING -- 
                wmh_change_mask_fake = output_img_pred
                wmh_change_mask_fake_temp = np.delete(wmh_change_mask_fake, 2, 3)
                wmh_change_mask_fake_temp = datautils.convert_from_1hot(wmh_change_mask_fake_temp)
                
                vol_out_mm3 = np.count_nonzero(wmh_change_mask_fake_temp == 2) * np.prod(loaded_data_f_1tp.pixdim)
                vol_out__ml_trsh_noGrow = vol_out_mm3 / 1000
                err_vol_trsh_noGrow = vol_2tp__ml - vol_out__ml_trsh_noGrow 

                vol = [
                    vol_1tp__ml, vol_2tp__ml, vol_out__ml, err_vol,
                    vol_out__ml_trsh_noShrink, err_vol_trsh_noShrink,
                    vol_out__ml_trsh_noGrow, err_vol_trsh_noGrow]
                vol_per_sample.append(vol)

                print(
                    "VOL (Y1)     : ", "{:.4f}".format(vol_1tp__ml),
                    ", VOL (Y2): ", "{:.4f}".format(vol_2tp__ml))
                print(
                    "VOL (1-HOT)  : ", "{:.4f}".format(vol_out__ml),
                    ", ERR (VOL): ", "{:.4f}".format(err_vol))
                print(
                    "VOL (NO SHRK): ", "{:.4f}".format(vol_out__ml_trsh_noShrink),
                    ", ERR (VOL): ", "{:.4f}".format(err_vol_trsh_noShrink))
                print(
                    "VOL (NO GROW): ", "{:.4f}".format(vol_out__ml_trsh_noGrow),
                    ", ERR (VOL): ", "{:.4f}".format(err_vol_trsh_noGrow))
                
            ## ONLY FOR PROBABILISTIC
            # z = 1.96 # 95%
            
            # ## DSC Evaluation
            mean_dsc = np.mean(dsc_per_sample, axis=0)
            std_dsc = np.std(dsc_per_sample, axis=0)
            mean_dsc_all.append(mean_dsc)
            std_dsc_all.append(std_dsc)

            print("")
            print("DSC -- MEAN & STD")
            print("MEAN: ", np.around(mean_dsc_all, decimals=4))
            print("STD : ", np.around(std_dsc_all, decimals=4))

            # ## VOL Evaluation
            mean_vol = np.mean(vol_per_sample, axis=0)
            std_vol = np.std(vol_per_sample, axis=0)
            mean_vol_all.append(mean_vol)
            std_vol_all.append(std_vol)

            print("")
            print("VOL -- MEAN & STD")
            print("MEAN: ", np.around(mean_vol_all, decimals=4))
            print("STD : ", np.around(std_vol_all, decimals=4))

            seg_result = seg_result / float(sampling_number)
            output_img_pred_lbl = datautils.convert_from_1hot(seg_result)

            output_img = datautils.data_prep_save(output_img_pred_lbl)
            nim = nib.Nifti1Image(output_img.astype('int8'), loaded_data_f_1tp.affine)
            nib.save(nim, dirOutData + '/' + name + '_cls_map.nii.gz')

            N, H, W, C = seg_result.shape
            for c in [1, 2, 3]:
                if c == 1:
                    name_type = "shrinking"
                    seg_result_shrink = seg_result_shrink / float(sampling_number)
                    output_img = datautils.data_prep_save(seg_result_shrink)
                elif c == 2:
                    name_type = "growing"
                    seg_result_grow = seg_result_grow / float(sampling_number)
                    output_img = datautils.data_prep_save(seg_result_grow)
                elif c == 3:
                    name_type = "stable"
                    seg_result_stable = seg_result_stable / float(sampling_number)
                    output_img = datautils.data_prep_save(seg_result_stable)
                
                nim = nib.Nifti1Image(output_img.astype('float32'), loaded_data_f_1tp.affine)
                nib.save(nim, dirOutData + '/' + name + '_1hot_'+name_type+'.nii.gz')
            print("")
            
            ## Save all evaluations to .csv file
            numpy_dsc_all = np.array(dsc_per_sample)
            f = open(dirOutData + '/' + name + '_dsc_all_samples.csv', 'w')
            np.savetxt(f, numpy_dsc_all, delimiter=",")
            f.close()

            numpy_vol_all = np.array(vol_per_sample)
            f = open(dirOutData + '/' + name + '_vol_all_samples.csv', 'w')
            np.savetxt(f, numpy_vol_all, delimiter=",")
            f.close()

            id += 1

        # Save the results for all data
        numpy_mean_dsc_all = np.array(mean_dsc_all)
        f = open(dirOutput + '/mean_dsc_all_scans.csv', 'w')
        np.savetxt(f, numpy_mean_dsc_all, delimiter=",")
        f.close()
        
        numpy_std_dsc_all = np.array(std_dsc_all)
        f = open(dirOutput + '/std_dsc_all_scans.csv', 'w')
        np.savetxt(f, numpy_std_dsc_all, delimiter=",")
        f.close()

        numpy_mean_vol_all = np.array(mean_vol_all)
        f = open(dirOutput + '/mean_vol_all_scans.csv', 'w')
        np.savetxt(f, numpy_mean_vol_all, delimiter=",")
        f.close()

        numpy_std_vol_all = np.array(std_vol_all)
        f = open(dirOutput + '/std_vol_all_scans.csv', 'w')
        np.savetxt(f, numpy_std_vol_all, delimiter=",")
        f.close()





