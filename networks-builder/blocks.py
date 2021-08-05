# https://github.com/MrGiovanni/UNetPlusPlus/blob/master/keras/segmentation_models/xnet/blocks.py

import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D, Conv2D
from tensorflow.keras.layers import Conv3DTranspose, UpSampling3D, Conv3D
from tensorflow.keras.layers import Add, Multiply, BatchNormalization, MaxPool2D
from tensorflow.keras.layers import Concatenate, Activation, Dropout, Add
from tensorflow_addons.layers import SpectralNormalization 

from utils import to_tuple

def handle_block_names(stage, cols, type_='decoder'):
    temp = 'upsample' if type_ == 'decoder' else 'downsample'
    
    conv_name = '{}_stage{}-{}_conv'.format(type_, stage, cols)
    bn_name = '{}_stage{}-{}_bn'.format(type_, stage, cols)
    up_name = '{}_stage{}-{}_{}'.format(type_, stage, cols, temp)
    merge_name = 'merge_{}-{}'.format(stage, cols)
    
    return conv_name, bn_name, up_name, merge_name

def handle_operation_names(stage, cols, type_='decoder', type_operation='add'):    
    operation_name = '{}_stage{}-{}_{}'.format(type_, stage, cols, type_operation)
    
    return operation_name

def handle_activation_names(stage, cols, type_='decoder', type_activation='relu'):    
    activation_name = '{}_stage{}-{}_act_{}'.format(type_, stage, cols, type_activation)
    
    return activation_name

def handle_block_kiu_names(stage, cols, type_='dec'):
    temp = 'resize'
    
    conv_name = '{}_stg{}-{}_conv'.format(type_, stage, cols)
    bn_name = '{}_stg{}-{}_bn'.format(type_, stage, cols)
    up_name = '{}_stg{}-{}_{}'.format(type_, stage, cols, temp)
    merge_name = 'merge_{}-{}'.format(stage, cols)
    
    return conv_name, bn_name, up_name, merge_name

def handle_operation_kiu_names(stage, cols, type_='decoder', type_operation='add'):    
    operation_name = '{}_stg{}-{}_{}'.format(type_, stage, cols, type_operation)
    
    return operation_name

def handle_activation_kiu_names(stage, cols, type_='decoder', type_activation='relu'):    
    activation_name = '{}_stg{}-{}_act_{}'.format(type_, stage, cols, type_activation)
    
    return activation_name

def Conv2DBlock(filters, kernel_size, strides=(1, 1), activation='relu', use_batchnorm=False,
            conv_name='conv', bn_name='bn', activation_name='relu'):

    def layer(x):
        x = Conv2D(filters, kernel_size, strides=strides, padding="same", name=conv_name,
                   use_bias=not(use_batchnorm))(x)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)
        # print("activation: ", activation)
        if activation is not None:
            x = Activation(activation, name=activation_name)(x)
        return x
    return layer

def Conv3DBlock(filters, kernel_size, strides=(1, 1, 1), activation='relu', use_batchnorm=False,
            conv_name='conv3d', bn_name='bn', activation_name='relu'):

    def layer(x):
        x = Conv3D(filters, kernel_size, strides=strides, padding="same", name=conv_name,
                   use_bias=not(use_batchnorm))(x)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)
        x = Activation(activation, name=activation_name)(x)
        return x
    return layer

def Upsample2D_block(filters, stage, cols, kernel_size=(3,3), upsample_rate=(2,2), dropout=0,
                     n_convs_per_block=2, activation='relu', use_batchnorm=False, skip=None):

    def layer(input_tensor):

        conv_name, bn_name, up_name, merge_name = handle_block_names(stage, cols, type_='decoder')
        activation_name = handle_activation_names(stage, cols, type_='decoder', type_activation=activation)

        x = UpSampling2D(size=upsample_rate, name=up_name, interpolation="bilinear")(input_tensor)

        if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
            # print("\nskip = {}".format(skip))
            # print("type(skip) is list: ", type(skip) is list)
            if type(skip) is list:
                x = Concatenate(name=merge_name)([x] + skip)
            else:
                x = Concatenate(name=merge_name)([x, skip])

        if dropout > 0:
            x = Dropout(dropout)(x)

        for i in range(n_convs_per_block):
            x = Conv2DBlock(filters, kernel_size, activation=activation, use_batchnorm=use_batchnorm,
                        conv_name=conv_name + str(i+1), bn_name=bn_name + str(i+1),
                        activation_name=activation_name + str(i+1))(x)

        return x
    return layer

def Upsample3D_block(filters, stage, cols, kernel_size=(3,3,3), upsample_rate=(2,2,2), dropout=0,
                     n_convs_per_block=2, activation='relu', use_batchnorm=False, skip=None):

    def layer(input_tensor):

        conv_name, bn_name, up_name, merge_name = handle_block_names(stage, cols, type_='decoder')
        activation_name = handle_activation_names(stage, cols, type_='decoder', type_activation=activation)

        x = UpSampling3D(size=upsample_rate, name=up_name)(input_tensor)

        if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
            print("\nskip = {}".format(skip))
            print("type(skip) is list: ", type(skip) is list)
            if type(skip) is list:
                x = Concatenate(name=merge_name)([x] + skip)
            else:
                x = Concatenate(name=merge_name)([x, skip])

        if dropout > 0:
            x = Dropout(dropout)(x)

        for i in range(n_convs_per_block):
            x = Conv3DBlock(filters, kernel_size, activation=activation, use_batchnorm=use_batchnorm,
                        conv_name=conv_name + str(i+1), bn_name=bn_name + str(i+1),
                        activation_name=activation_name + str(i+1))(x)

        return x
    return layer

def Transpose2D_block(filters, stage, cols, kernel_size=(3,3), n_convs_per_block=2, upsample_rate=(2,2), dropout=0,
                        transpose_kernel_size=(4,4), activation='relu', use_batchnorm=False, skip=None):

    def layer(input_tensor):

        conv_name, bn_name, up_name, merge_name = handle_block_names(stage, cols, type_='decoder')
        activation_name = handle_activation_names(stage, cols, type_='decoder', type_activation=activation)

        x = Conv2DTranspose(filters, transpose_kernel_size, activation=activation, strides=upsample_rate,
                            padding='same', name=up_name, use_bias=not(use_batchnorm))(input_tensor)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name+'1')(x)
        x = Activation(activation, name=activation_name+'1')(x)

        if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
            # print("\nskip = {}".format(skip))
            # print("type(skip) is list: ", type(skip) is list)
            if type(skip) is list:
                merge_list = []
                merge_list.append(x)
                for l in skip:
                    merge_list.append(l)
                x = Concatenate(name=merge_name)(merge_list)
            else:
                x = Concatenate(name=merge_name)([x, skip])
        
        if dropout > 0:
            x = Dropout(dropout)(x)

        for i in range(n_convs_per_block):
            x = Conv2DBlock(filters, kernel_size, activation=activation, use_batchnorm=use_batchnorm,
                        conv_name=conv_name + str(i+2), bn_name=bn_name + str(i+2),
                        activation_name=activation_name + str(i+2))(x)
                     
        return x
    return layer

def Transpose3D_block(filters, stage, cols, kernel_size=(3,3,3), n_convs_per_block=2, upsample_rate=(2,2,2), dropout=0,
                        transpose_kernel_size=(4,4,4), activation='relu', use_batchnorm=False, skip=None):

    def layer(input_tensor):

        conv_name, bn_name, up_name, merge_name = handle_block_names(stage, cols, type_='decoder')
        activation_name = handle_activation_names(stage, cols, type_='decoder', type_activation=activation)

        x = Conv3DTranspose(filters, transpose_kernel_size, activation=activation, strides=upsample_rate,
                            padding='same', name=up_name, use_bias=not(use_batchnorm))(input_tensor)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name+'1')(x)
        x = Activation(activation, name=activation_name+'1')(x)

        if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
            # print("\nskip = {}".format(skip))
            if type(skip) is list:
                merge_list = []
                merge_list.append(x)
                for l in skip:
                    merge_list.append(l)
                x = Concatenate(name=merge_name)(merge_list)
            else:
                x = Concatenate(name=merge_name)([x, skip])
        
        if dropout > 0:
            x = Dropout(dropout)(x)

        for i in range(n_convs_per_block):
            x = Conv3DBlock(filters, kernel_size, activation=activation, use_batchnorm=use_batchnorm,
                        conv_name=conv_name + str(i+2), bn_name=bn_name + str(i+2),
                        activation_name=activation_name + str(i+2))(x)
                     
        return x
    return layer

def Downsample2DKite_block(filters, stage, cols, kernel_size=(3,3), downsample_rate=(2,2), dropout=0,
                     n_convs_per_block=2, activation='relu', use_batchnorm=False, skip=None, crfb=False):

    def layer(input_tensor):

        if crfb:
            block_type = 'crfb'
        else:
            block_type = 'encDown'

        conv_name, bn_name, up_name, merge_name = handle_block_kiu_names(stage, cols, type_=block_type)
        activation_name = handle_activation_kiu_names(stage, cols, type_=block_type, type_activation=activation)

        x = input_tensor

        if skip is not None:
            _, _, _, fm_s = skip.get_shape().as_list()
            _, _, _, fm_x = x.get_shape().as_list()
            if fm_s == fm_x:
                x = Add(name=merge_name)([x, skip])
            elif int(fm_s / fm_x) == 2:
                x = Add(name=merge_name)([Concatenate()([x, x]), skip])
            elif int(fm_x / fm_s) == 2:
                x = Add(name=merge_name)([x, Concatenate()([skip, skip])])

        for i in range(n_convs_per_block):
            activation_used = activation
            if i == n_convs_per_block-1:
                activation_used = None
            x = Conv2DBlock(filters, kernel_size, activation=activation_used, use_batchnorm=use_batchnorm,
                        conv_name=conv_name + str(i+1), bn_name=bn_name + str(i+1),
                        activation_name=activation_name + str(i+1))(x)

        x = MaxPool2D(pool_size=downsample_rate) (x)
        x = Activation(activation, name=activation_name)(x)

        if dropout > 0:
            x = Dropout(dropout)(x)

        return x
    return layer


def Stride2DKite_block(filters, stage, cols, kernel_size=(3,3), downsample_rate=(2,2), dropout=0,
                     n_convs_per_block=2, activation='relu', use_batchnorm=False, skip=None, crfb=False):

    def layer(input_tensor):

        if crfb:
            block_type = 'crfb'
        else:
            block_type = 'encStride'

        conv_name, bn_name, up_name, merge_name = handle_block_kiu_names(stage, cols, type_=block_type)
        activation_name = handle_activation_kiu_names(stage, cols, type_=block_type, type_activation=activation)

        x = input_tensor

        if skip is not None:
            _, _, _, fm_s = skip.get_shape().as_list()
            _, _, _, fm_x = x.get_shape().as_list()
            if fm_s == fm_x:
                x = Add(name=merge_name)([x, skip])
            elif int(fm_s / fm_x) == 2:
                x = Add(name=merge_name)([Concatenate()([x, x]), skip])
            elif int(fm_x / fm_s) == 2:
                x = Add(name=merge_name)([x, Concatenate()([skip, skip])])

        for i in range(n_convs_per_block):
            activation_used = activation
            if i == n_convs_per_block-1:
                activation_used = None
            x = Conv2DBlock(filters, kernel_size, activation=activation_used, use_batchnorm=use_batchnorm,
                        conv_name=conv_name + str(i+1), bn_name=bn_name + str(i+1),
                        activation_name=activation_name + str(i+1))(x)

        x = Activation(activation, name=activation_name)(x)
        x = Conv2DBlock(filters, downsample_rate, strides=downsample_rate, activation=activation, 
                        use_batchnorm=use_batchnorm, conv_name=conv_name+"_stride", bn_name=bn_name+"_stride", 
                        activation_name=activation_name+"_stride")(x)

        if dropout > 0:
            x = Dropout(dropout)(x)

        return x
    return layer

def Upsample2DKite_block(filters, stage, cols, kernel_size=(3,3), upsample_rate=(2,2), dropout=0,
                     n_convs_per_block=2, activation='relu', use_batchnorm=False, skip=None, crfb=False):

    def layer(input_tensor):

        if crfb:
            block_type = 'crfb'
        else:
            block_type = 'decUpSam'

        conv_name, bn_name, up_name, merge_name = handle_block_kiu_names(stage, cols, type_=block_type)
        activation_name = handle_activation_kiu_names(stage, cols, type_=block_type, type_activation=activation)

        x = input_tensor

        if skip is not None:
            _, _, _, fm_s = skip.get_shape().as_list()
            _, _, _, fm_x = x.get_shape().as_list()
            if fm_s == fm_x:
                x = Add(name=merge_name)([x, skip])
            elif int(fm_s / fm_x) == 2:
                x = Add(name=merge_name)([Concatenate()([x, x]), skip])
            elif int(fm_x / fm_s) == 2:
                x = Add(name=merge_name)([x, Concatenate()([skip, skip])])

        for i in range(n_convs_per_block):
            activation_used = activation
            if i == n_convs_per_block-1:
                activation_used = None
            x = Conv2DBlock(filters, kernel_size, activation=activation_used, use_batchnorm=use_batchnorm,
                        conv_name=conv_name + str(i+1), bn_name=bn_name + str(i+1),
                        activation_name=activation_name + str(i+1))(x)

        x = UpSampling2D(size=upsample_rate, name=up_name, interpolation="bilinear")(x)

        if dropout > 0:
            x = Dropout(dropout)(x)

        x = Activation(activation, name=activation_name)(x)

        return x
    return layer

def Transpose2DKite_block(filters, stage, cols, kernel_size=(3,3), n_convs_per_block=2, upsample_rate=(2,2), dropout=0,
                        activation='relu', use_batchnorm=False, skip=None, crfb=False):

    def layer(input_tensor):

        if crfb:
            block_type = 'crfb'
        else:
            block_type = 'decTranspose'

        conv_name, bn_name, up_name, merge_name = handle_block_kiu_names(stage, cols, type_=block_type)
        activation_name = handle_activation_kiu_names(stage, cols, type_=block_type, type_activation=activation)

        x = input_tensor

        if skip is not None:
            _, _, _, fm_s = skip.get_shape().as_list()
            _, _, _, fm_x = x.get_shape().as_list()
            if fm_s == fm_x:
                x = Add(name=merge_name)([x, skip])
            elif int(fm_s / fm_x) == 2:
                x = Add(name=merge_name)([Concatenate()([x, x]), skip])
            elif int(fm_x / fm_s) == 2:
                x = Add(name=merge_name)([x, Concatenate()([skip, skip])])

        for i in range(n_convs_per_block):
            x = Conv2DBlock(filters, kernel_size, activation=activation, use_batchnorm=use_batchnorm,
                        conv_name=conv_name + str(i+2), bn_name=bn_name + str(i+2),
                        activation_name=activation_name + str(i+2))(x)
        
        transpose_kernel_size = tuple(2 * x for x in upsample_rate)
        x = Conv2DTranspose(filters, transpose_kernel_size, activation=activation, strides=upsample_rate,
                            padding='same', name=up_name+'_transpose', use_bias=not(use_batchnorm))(x)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name+'_transpose')(x)
        x = Activation(activation, name=activation_name+'_transpose')(x)

        if dropout > 0:
            x = Dropout(dropout)(x)
                     
        return x
    return layer

def CRFB2DKite_block(filters, stage, cols, kernel_size=(3,3), n_convs_per_block=1, activation='relu', 
                        use_batchnorm=False, decoder_block_type='upsampling', encoder_block_type='downsampling'):

    def layer(input_tensor):

        from_kite = input_tensor[0]
        from_u = input_tensor[1]

        # print("from_kite shape: ", from_kite.get_shape().as_list())
        # print("from_u shape   : ", from_u.get_shape().as_list())

        _, res_kite, _, _, = from_kite.get_shape().as_list()
        _, res_u, _, _, = from_u.get_shape().as_list()

        resize_rate = max([res_kite/res_u, res_u/res_kite])

        upsample_rate = to_tuple(int(resize_rate))
        downsample_rate = to_tuple(int(resize_rate))

        # print("upsample_rate: ", upsample_rate)
        # print("downsample_rate: ", downsample_rate)

        conv_name, bn_name, up_name, merge_name = handle_block_kiu_names(stage, cols, type_='crfb')
        activation_name = handle_activation_kiu_names(stage, cols, type_='crfb', type_activation=activation)

        if decoder_block_type == 'transpose':
            up_block = Transpose2DKite_block
        elif decoder_block_type == 'upsampling':
            up_block = Upsample2DKite_block

        if encoder_block_type == 'downsampling':
            down_block = Downsample2DKite_block
        elif encoder_block_type == 'stride':
            down_block = Stride2DKite_block

        ## From Kite-Net to U-Net
        RKi = down_block(filters, stage, 0, activation=activation, kernel_size=kernel_size,
                            downsample_rate=downsample_rate, n_convs_per_block=n_convs_per_block, 
                            use_batchnorm=use_batchnorm, crfb=True)(from_kite)
        toUNet = Add()([RKi, from_u])

        ## From U-Net to Kite-Net
        RU = up_block(filters, stage, 1, activation=activation, kernel_size=kernel_size,
                            upsample_rate=upsample_rate, n_convs_per_block=n_convs_per_block, 
                            use_batchnorm=use_batchnorm, crfb=True)(from_u)
        toKiNet = Add()([RU, from_kite])
                     
        return toKiNet, toUNet
    return layer


def CRFBRes2DKite_block(filters2kite, filters2unet, stage, cols, kernel_size=(3,3), n_convs_per_block=1, activation='relu', 
                        use_batchnorm=False, decoder_block_type='upsampling', encoder_block_type='downsampling'):

    def layer(input_tensor):

        from_kite = input_tensor[0]
        from_u = input_tensor[1]

        # print("from_kite shape: ", from_kite.get_shape().as_list())
        # print("from_u shape   : ", from_u.get_shape().as_list())

        _, res_kite, _, _, = from_kite.get_shape().as_list()
        _, res_u, _, _, = from_u.get_shape().as_list()

        resize_rate = max([res_kite/res_u, res_u/res_kite])

        upsample_rate = to_tuple(int(resize_rate))
        downsample_rate = to_tuple(int(resize_rate))

        # print("upsample_rate: ", upsample_rate)
        # print("downsample_rate: ", downsample_rate)

        conv_name, bn_name, up_name, merge_name = handle_block_kiu_names(stage, cols, type_='crfb')
        activation_name = handle_activation_kiu_names(stage, cols, type_='crfb', type_activation=activation)

        if decoder_block_type == 'transpose':
            up_block = Transpose2DKite_block
        elif decoder_block_type == 'upsampling':
            up_block = Upsample2DKite_block

        if encoder_block_type == 'downsampling':
            down_block = Downsample2DKite_block
        elif encoder_block_type == 'stride':
            down_block = Stride2DKite_block

        ## From Kite-Net to U-Net
        RKi = down_block(filters2unet, stage, 0, activation=activation, kernel_size=kernel_size,
                            downsample_rate=downsample_rate, n_convs_per_block=n_convs_per_block, 
                            use_batchnorm=use_batchnorm, crfb=True)(from_kite)
        toUNet = Add()([RKi, from_u])

        ## From U-Net to Kite-Net
        RU = up_block(filters2kite, stage, 1, activation=activation, kernel_size=kernel_size,
                            upsample_rate=upsample_rate, n_convs_per_block=n_convs_per_block, 
                            use_batchnorm=use_batchnorm, crfb=True)(from_u)
        toKiNet = Add()([RU, from_kite])
                     
        return toKiNet, toUNet
    return layer

def down_block(filters, stage, cols, kernel_size=(3,3), n_convs_per_block=2, activation='relu', 
                use_batchnorm=False, dropout=0, skip=None):
    
    def layer(input_tensor):
        conv_name, bn_name, up_name, merge_name = handle_block_names(stage, cols, type_='encoder')
        activation_name = handle_activation_names(stage, cols, type_='encoder', type_activation=activation)

        x = input_tensor
        
        if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
            # print("\nskip = {}".format(skip))
            if type(skip) is list:
                merge_list = []
                merge_list.append(x)
                for l in skip:
                    merge_list.append(l)
                x = Concatenate(name=merge_name)(merge_list)
            else:
                x = Concatenate(name=merge_name)([x, skip])

        if dropout > 0:
            x = Dropout(dropout)(x)

        for i in range(n_convs_per_block):
            x = Conv2DBlock(filters, kernel_size, activation=activation, use_batchnorm=use_batchnorm, 
                        conv_name=conv_name + str(i+1), bn_name=bn_name + str(i+1), 
                        activation_name=activation_name + str(i+1)) (x)

        return x
    return layer

def down_block3D(filters, stage, cols, kernel_size=(3,3,3), n_convs_per_block=2, activation='relu', use_batchnorm=False):
    
    def layer(input_tensor):
        conv_name, bn_name,_,_ = handle_block_names(stage, cols, type_='encoder')
        activation_name = handle_activation_names(stage, cols, type_='encoder', type_activation=activation)

        x = input_tensor
        for i in range(n_convs_per_block):
            x = Conv3DBlock(filters, kernel_size, activation=activation, use_batchnorm=use_batchnorm, 
                        conv_name=conv_name + str(i+1), bn_name=bn_name + str(i+1), 
                        activation_name=activation_name + str(i+1)) (x)

        return x
    return layer

def SpectralNormalizationConv2D(filters, stage, cols, kernel_size=3, strides=(1, 1), n_convs_per_block=2, activation='relu'):
    
    def layer(input_tensor):
        conv_name, bn_name,_,_ = handle_block_names(stage, cols, type_='SNDiscriminator')
        activation_name = handle_activation_names(stage, cols, type_='SNDiscriminator', type_activation=activation)

        x = input_tensor
        for i in range(n_convs_per_block):
            x = SpectralNormalization(Conv2D(
                filters, kernel_size, strides=strides, padding="same", activation=activation
                )) (x)
        return x
    return layer

def SpectralNormalizationConv3D(filters, stage, cols, kernel_size=(3,3,3), strides=(1,1,1), n_convs_per_block=2, activation='relu'):
    
    def layer(input_tensor):
        conv_name, bn_name,_,_ = handle_block_names(stage, cols, type_='SNDiscriminator3D')
        activation_name = handle_activation_names(stage, cols, type_='SNDiscriminator3D', type_activation=activation)

        x = input_tensor
        for i in range(n_convs_per_block):
            x = SpectralNormalization(Conv3D(
                filters, kernel_size, strides=strides, padding="same", activation=activation
                )) (x)
        return x
    return layer

def upRelu_attention(filters, transpose=False, activation='relu', use_batchnorm=False, conv_name='c_up',
           bn_name='b_up', activation_name='relu_up'):

    def layer(input_tensor):
        if transpose:
            x = Conv2DTranspose(filters, kernel_size=4, padding='same') (input_tensor)
        else:
            x = UpSampling2D(size=2, interpolation="bilinear") (input_tensor)

        x = Conv2DBlock(filters, kernel_size=3, activation=activation, use_batchnorm=use_batchnorm,
                     conv_name=conv_name, bn_name=bn_name, activation_name=activation_name) (x)
        return x
    return layer

def upRelu3D_attention(filters, transpose=False, activation='relu', use_batchnorm=False, conv_name='c_up',
           bn_name='b_up', activation_name='relu_up'):

    def layer(input_tensor):
        if transpose:
            x = Conv3DTranspose(filters, kernel_size=4, padding='same') (input_tensor)
        else:
            x = UpSampling3D(size=2) (input_tensor)

        x = Conv3DBlock(filters, kernel_size=3, activation=activation, use_batchnorm=use_batchnorm,
                     conv_name=conv_name, bn_name=bn_name, activation_name=activation_name) (x)
        return x
    return layer

def attention_block(filters, skip, stage, cols, attention_activation='relu', activation='relu', upsample_rate=(2,2)):

    def layer(input_tensor):
        conv_name, bn_name, up_name,_ = handle_block_names(stage, cols, type_='attention')
        add_name = handle_operation_names(stage, cols, type_='attention', type_operation='add')
        mul_name = handle_operation_names(stage, cols, type_='attention', type_operation='mul')
        activation_name = handle_activation_names(stage, cols, type_='attention', type_activation=activation)
        attention_activation_name = handle_activation_names(stage, cols, type_='attention', type_activation=attention_activation)
        sigmoid_name = handle_activation_names(stage, cols, type_='attention', type_activation='sigmoid')

        x_up = upRelu_attention(filters, activation=attention_activation, conv_name=conv_name+'_before', bn_name=bn_name+'_before',
                       activation_name=attention_activation_name+'_before', use_batchnorm=True) (input_tensor)

        x1 = Conv2D(filters, kernel_size=1, padding='same', name=conv_name+'_skip') (skip)
        x1 = BatchNormalization(name=bn_name+'1') (x1)
        x2 = Conv2D(filters, kernel_size=1, padding='same', name=conv_name+'_up') (x_up)
        x2 = BatchNormalization(name=bn_name+'2') (x2)

        x = Add(name=add_name) ([x1,x2])
        x = Activation(activation, name=activation_name) (x)
        x = Conv2D(1, kernel_size=1, padding='same', name=conv_name) (x)
        x = BatchNormalization(name=bn_name+'3') (x)
        x = Activation('sigmoid', name=sigmoid_name) (x)
        x = Multiply(name=mul_name) ([skip,x])

        return x
    return layer

def attention_block3D(filters, skip, stage, cols, attention_activation='relu', activation='relu', upsample_rate=(2,2)):

    def layer(input_tensor):
        conv_name, bn_name, up_name,_ = handle_block_names(stage, cols, type_='attention')
        add_name = handle_operation_names(stage, cols, type_='attention', type_operation='add')
        mul_name = handle_operation_names(stage, cols, type_='attention', type_operation='mul')
        activation_name = handle_activation_names(stage, cols, type_='attention', type_activation=activation)
        attention_activation_name = handle_activation_names(stage, cols, type_='attention', type_activation=attention_activation)
        sigmoid_name = handle_activation_names(stage, cols, type_='attention', type_activation='sigmoid')

        x_up = upRelu3D_attention(filters, activation=attention_activation, conv_name=conv_name+'_before', bn_name=bn_name+'_before',
                       activation_name=attention_activation_name+'_before', use_batchnorm=True) (input_tensor)

        x1 = Conv3D(filters, kernel_size=1, padding='same', name=conv_name+'_skip') (skip)
        x1 = BatchNormalization(name=bn_name+'1') (x1)
        x2 = Conv3D(filters, kernel_size=1, padding='same', name=conv_name+'_up') (x_up)
        x2 = BatchNormalization(name=bn_name+'2') (x2)

        x = Add(name=add_name) ([x1,x2])
        x = Activation(activation, name=activation_name) (x)
        x = Conv3D(1, kernel_size=1, padding='same', name=conv_name) (x)
        x = BatchNormalization(name=bn_name+'3') (x)
        x = Activation('sigmoid', name=sigmoid_name) (x)
        x = Multiply(name=mul_name) ([skip,x])

        return x
    return layer

def DeepSupervision(classes, activation='sigmoid'):

    def layer(seg_branches):
        concat_list = []
        for k,i in enumerate(seg_branches):
            temp = Conv2D(classes, kernel_size=1, padding='same', name='ds_conv_'+str(k+1)) (i)
            if activation != None:
                temp = Activation(activation, name='ds_' + activation + '_'+str(k+1)) (temp)
            concat_list.append(temp)

        return concat_list
    return layer

def spectral_normed_weight(w, u=None, num_iters=1, update_collection=None, with_sigma=False ):
    # For Power iteration method, usually num_iters = 1 will be enough
    w_shape = w.shape.as_list()
    w_new_shape = [ np.prod(w_shape[:-1]), w_shape[-1] ]
    w_reshaped = tf.reshape(w, w_new_shape, name='w_reshaped')

    if u is None:
        u = tf.get_variable("u_vec", [w_new_shape[0], 1], initializer=params.weight_initializer, trainable=False)
   
    # power iteration
    u_ = u
    for _ in range(num_iters):
        # ( w_new_shape[1], w_new_shape[0] ) * ( w_new_shape[0], 1 ) -> ( w_new_shape[1], 1 )
        v_ = _l2normalize(tf.matmul(tf.transpose(w_reshaped), u_))
        # ( w_new_shape[0], w_new_shape[1] ) * ( w_new_shape[1], 1 ) -> ( w_new_shape[0], 1 )
        u_ = _l2normalize(tf.matmul(w_reshaped, v_))

    u_final = tf.identity(u_, name='u_final') # ( w_new_shape[0], 1 )
    v_final = tf.identity(v_, name='v_final') # ( w_new_shape[1], 1 )

    u_final = tf.stop_gradient(u_final)
    v_final = tf.stop_gradient(v_final)

    sigma = tf.matmul(tf.matmul(tf.transpose(u_final), w_reshaped), v_final, name="est_sigma")

    update_u_op = tf.assign(u, u_final)
    with tf.control_dependencies([update_u_op]):
        sigma = tf.identity(sigma)
        w_bar = tf.identity(w / sigma, 'w_bar')

    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar

def _l2normalize(v, eps=1e-12):
    with tf.name_scope('l2normalize'):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def sn_conv2d(inputs, out_dim, k_size, strides, padding="SAME", w_init=None, use_bias=True,  spectral_normed=True, name="conv2d", layer_reuse=False):
   
    with tf.variable_scope(name, reuse=layer_reuse):
        w = tf.get_variable("w", shape=[k_size, k_size, inputs.get_shape()[-1], out_dim], dtype=tf.float32, initializer=w_init)
       
        if spectral_normed:
            w = spectral_normed_weight(w)
       
        conv = Conv2D(inputs, w, strides=[1, strides, strides, 1], padding=padding.upper())
       
        if use_bias:
            biases = tf.get_variable("b", [out_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases, name="conv_add_b")
       
        return conv

