# https://github.com/MrGiovanni/UNetPlusPlus/blob/master/keras/segmentation_models/xnet/model.py

from builders.DiscriminatorSNGAN import build_discriminator_sngan
from utils import freeze_model
from backbones import get_backbone

def DiscriminatorSNGAN(
        encoder_filters=(32,64,128,256,512),
        downsample_rates=(2,2,2,2,2),
        filter_sizes=(5,5,3,3,3),
        n_downsamples=5,
        n_convs_per_block=2,
        conv_activation='relu',
        encoder_block_type='downsampling',
        input_shape=(None,None,3),
        type='2D'
        ):

    if type == '2D':
        model = build_discriminator_sngan(
            encoder_filters=encoder_filters,
            downsample_rates=downsample_rates,
            filter_sizes=filter_sizes,
            n_downsamples=n_downsamples,
            n_convs_per_block=n_convs_per_block,
            conv_activation=conv_activation,
            encoder_block_type=encoder_block_type,
            input_shape=input_shape
            )
        model._name = model_name = 'DiscriminatorSNGAN'

    return model 
