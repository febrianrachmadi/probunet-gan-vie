# https://github.com/MrGiovanni/UNetPlusPlus/blob/master/keras/segmentation_models/unet/builder.py

from tensorflow import image
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense
from tensorflow.keras.layers import Conv3D, MaxPool3D, Concatenate
from tensorflow_addons.layers import SpectralNormalization 

from utils import get_layer_number, to_tuple, to_3Dtuple
from blocks import Transpose2D_block, Upsample2D_block, Conv2DBlock, SpectralNormalizationConv2D
from blocks import Transpose3D_block, Upsample3D_block, Conv3DBlock, SpectralNormalizationConv3D
from blocks import down_block, attention_block, handle_block_names, handle_activation_names

def build_discriminator_sngan(
    encoder_filters=(32,64,128,256,512),
    downsample_rates=(2,2,2,2,2),
    filter_sizes=(5,5,3,3,3),
    n_downsamples=5,
    n_convs_per_block=2,
    conv_activation='relu',
    encoder_block_type='downsampling',
    input_shape=(None,None,3)):


    # Using Conv+relu for the encoder
    input = Input(shape=input_shape)

    for i in range(n_downsamples):
        if i == 0:
            x = SpectralNormalizationConv2D(
                    encoder_filters[i], i, 0, 
                    kernel_size=filter_sizes[i],
                    n_convs_per_block=n_convs_per_block,
                    activation=conv_activation
                    ) (input)
        else:
            down_rate = to_tuple(downsample_rates[i])
            if encoder_block_type == 'downsampling':
                x = MaxPool2D(pool_size=down_rate) (x)
            elif encoder_block_type == 'stride':
                x = SpectralNormalization(Conv2D(
                        encoder_filters[i], 
                        kernel_size=down_rate, strides=down_rate, 
                        padding="same", activation=conv_activation
                    )) (x)
            x = SpectralNormalizationConv2D(
                    encoder_filters[i], i, 0, 
                    kernel_size=filter_sizes[i],
                    n_convs_per_block=n_convs_per_block,
                    activation=conv_activation
                    ) (x)

    x = SpectralNormalization(Conv2D(1, kernel_size=1, 
            padding='same', name='final_conv', activation=conv_activation))(x)
    x = Flatten()(x)
    x = Dense(1, kernel_initializer='he_normal')(x)

    model = Model(input, x)

    return model

def build_discriminator_sngan3D(
    encoder_filters=(32,64,128,256,512),
    downsample_rates=(2,2,2,2,2),
    filter_sizes=(5,5,3,3,3),
    n_downsamples=5,
    n_convs_per_block=2,
    conv_activation='relu',
    encoder_block_type='downsampling',
    input_shape=(None,None,3)):


    # Using Conv+relu for the encoder
    input = Input(shape=input_shape)

    for i in range(n_downsamples):
        if i == 0:
            x = SpectralNormalizationConv3D(
                    encoder_filters[i], i, 0, 
                    kernel_size=filter_sizes[i],
                    n_convs_per_block=n_convs_per_block,
                    activation=conv_activation
                    ) (input)
        else:
            down_rate = to_3Dtuple(downsample_rates[i])
            if encoder_block_type == 'downsampling':
                x = MaxPool3D(pool_size=down_rate) (x)
            elif encoder_block_type == 'stride':
                x = SpectralNormalization(Conv3D(
                        encoder_filters[i], 
                        kernel_size=down_rate, strides=down_rate, 
                        padding="same", activation=conv_activation
                    )) (x)
            x = SpectralNormalizationConv3D(
                    encoder_filters[i], i, 0, 
                    kernel_size=filter_sizes[i],
                    n_convs_per_block=n_convs_per_block,
                    activation=conv_activation
                    ) (x)

    x = SpectralNormalization(Conv3D(1, kernel_size=1, 
            padding='same', name='final_conv', activation=conv_activation))(x)
    x = Flatten()(x)
    x = Dense(1, kernel_initializer='he_normal')(x)

    model = Model(input, x)

    return model


def build_discriminator_sngan_3B(
    encoder_filters=(32,64,128,256,512),
    downsample_rates=(2,2,2,2,2),
    filter_sizes=(5,5,3,3,3),
    n_downsamples=5,
    n_convs_per_block=2,
    conv_activation='relu',
    encoder_block_type='downsampling',
    input_shape=(None,None,1)):

    # # Using Conv+relu for the encoder
    # input_1 = Input(shape=input_shape)
    # input_2 = Input(shape=input_shape)
    # input_3 = Input(shape=input_shape)
    # input_concatenated = Concatenate(axis=-1)([input_1, input_2])
    # input_concatenated = Concatenate(axis=-1)([input_concatenated, input_3])
    
    # Using Conv+relu for the encoder
    input_concatenated = Input(shape=input_shape)

    for i in range(n_downsamples):
        if i == 0:
            x = SpectralNormalizationConv2D(
                    encoder_filters[i], i, 0, 
                    kernel_size=filter_sizes[i],
                    n_convs_per_block=n_convs_per_block,
                    activation=conv_activation
                    ) (input_concatenated)
        else:
            down_rate = to_tuple(downsample_rates[i])
            if encoder_block_type == 'downsampling':
                x = MaxPool2D(pool_size=down_rate) (x)
            elif encoder_block_type == 'stride':
                x = SpectralNormalization(Conv2D(
                        encoder_filters[i], 
                        kernel_size=down_rate, strides=down_rate, 
                        padding="same", activation=conv_activation
                    )) (x)
            x = SpectralNormalizationConv2D(
                    encoder_filters[i], i, 0, 
                    kernel_size=filter_sizes[i],
                    n_convs_per_block=n_convs_per_block,
                    activation=conv_activation
                    ) (x)

    x = SpectralNormalization(Conv2D(1, kernel_size=1, 
            padding='same', name='final_conv', activation=conv_activation))(x)
    x = Flatten()(x)
    x = Dense(1, kernel_initializer='he_normal')(x)

    # model = Model([input_1, input_2, input_3], x)
    model = Model(input_concatenated, x)

    return model