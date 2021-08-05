import tensorflow as tf
from tensorflow.keras.layers import Add, Conv2D, BatchNormalization
from tensorflow.keras.layers import Activation, UpSampling2D, Multiply

def handle_block_names(stage, cols, type_block='decoder', type_act='relu'):
    conv_name = '{}_stage{}-{}_conv'.format(type_block, stage, cols)
    bn_name = '{}_stage{}-{}_bn'.format(type_block, stage, cols)
    act_name = '{}_stage{}-{}_{}'.format(type_block, stage, cols, type_act)
    return conv_name, bn_name, act_name

def conv_relu(filters, kernel_size, use_batchnorm=False, conv_name='conv',
              bn_name='bn', act_name='relu', act_function='relu'):
  
    def layer(input_tensor):
        x = Conv2D(filters, kernel_size, padding='same', name=conv_name) (input_tensor)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name) (x)
        x = Activation(act_function, name=act_name) (x)

        return x
    return layer

def conv_block(filters, stage, cols, kernel_size=3, use_batchnorm=True,
               amount=3, type_act='relu', type_block='encoder'):
    
    def layer(x):
        act_function = tf.identity if type_act == 'identity' else type_act
        conv_name, bn_name, act_name = handle_block_names(stage, cols, type_block=type_block, type_act=type_act)
        for i in range(amount):
            temp = '_'+str(i+1)
            x = conv_relu(filters, kernel_size=kernel_size, use_batchnorm=use_batchnorm, 
                          conv_name=conv_name+temp, bn_name=bn_name+temp,
                          act_name=act_name+temp, act_function=act_function) (x)
        return x
    return layer

def z_mu_sigma(filters, stage, cols, use_batchnorm=True, type_block='z'):
    def layer(x):
        mu = conv_block(filters, stage, cols, use_batchnorm=use_batchnorm, amount=1,
                        kernel_size=1, type_act='identity', type_block='mu') (x)
        sigma = conv_block(filters, stage, cols, use_batchnorm=use_batchnorm, amount=1,
                           kernel_size=1, type_act='softplus', type_block='sigma') (x)
                           
        z = Multiply(name='z_stage{}-{}_mul'.format(stage,cols)) ([sigma,
                                                                   tf.random.normal(tf.shape(mu), 0, 1, dtype=tf.float32)])
        z = Add(name='z_stage{}-{}_add'.format(stage,cols)) ([mu, z])
        return z, mu, sigma
    return layer

def increase_resolution(filters, stage, cols, times):
    def layer(x):
        for i in range(times):
            x = UpSampling2D(name='z_post_stage{}-{}_up_{}'.format(stage, cols, str(times+1))) (x)
            x = conv_block(filters, stage, cols, amount=1, type_block='z_post') (x)
        return x
    return layer