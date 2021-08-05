# https://github.com/MrGiovanni/UNetPlusPlus/blob/master/keras/segmentation_models/unet/builder.py
from tensorflow import keras
import tensorflow as tf 
import losses as loss
import numpy as np

from tensorflow.keras import Input
from helpers import conv_block, z_mu_sigma
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Mean
from tensorflow.keras.layers import Activation, UpSampling2D, MaxPool2D, Concatenate

# Instantiate the optimizer for both networks
# (learning_rate=0.0002, beta_1=0.5 are recommended)
generator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)
discriminator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)

# Define the loss functions for the generator.
def generator_loss(misleading_labels, fake_logits, fake_imgs, real_imgs, loss_weights=[0.25,0.75,0.75,0.5]):
    bce = keras.losses.BinaryCrossentropy(from_logits=True)
    fcl = loss.categorical_focal_loss(alpha=[loss_weights], gamma=2)
    real_dem = layers.Lambda(lambda x : x[:,:,:,1:])(real_imgs)
    fake_dem = layers.Lambda(lambda x : x[:,:,:,1:])(fake_imgs)
    return bce(misleading_labels, fake_logits) + fcl(real_dem, fake_dem)

# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_logits, fake_logits):
    real_loss = tf.reduce_mean(real_logits)
    fake_loss = tf.reduce_mean(fake_logits)
    return fake_loss - real_loss

## https://keras.io/examples/generative/dcgan_overriding_train_step/
class ProbUNet2Prior_Y1Y2GAN(keras.Model):
    def __init__(
            self, 
            discriminator,
            num_filters,
            latent_dim,
            discriminator_extra_steps=3, 
            cost_function=None,
            n_label=1,
            resolution_lvl=5,
            img_shape=(None, None, 1),
            seg_shape=(None, None, 1),
            downsample_signal=(2,2,2,2,2)
            ):
        super(ProbUNet2Prior_Y1Y2GAN, self).__init__()
        
        ## For the Probability UNet
        self.num_filters = num_filters
        self.resolution_lvl = resolution_lvl
        self.downsample_signal = downsample_signal
        self.n_label = n_label
        self.generator = self.unet(img_shape, latent_dim)
        self.prior = self.latent_space_net(img_shape, None, latent_dim)
        self.posterior = self.latent_space_net(img_shape, seg_shape, latent_dim)

        ## For the UNet & GAN
        self.d_steps = discriminator_extra_steps
        self.focal_val_loss = cost_function
        self.discriminator = discriminator

    def compile(self, prior_opt, posterior_opt, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(ProbUNet2Prior_Y1Y2GAN, self).compile()
        self.prior_opt = prior_opt
        self.posterior_opt = posterior_opt
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def call(self, real_images):
        if len(real_images) > 1:
            real_brain_mri_y1 = real_images[0]
            real_brain_mri_y2 = real_images[1]
        z_prior, mu_prior, sigma_prior = self.prior(real_brain_mri_y1, training=False)
        pred_dem = self.generator([real_brain_mri_y1, z_prior], training=False)
        return pred_dem

    def predict(self, real_images, batch_size=32):
        if isinstance(real_images, tuple) and len(real_images) > 1:
            print(" --> TWO tuple..")
            real_brain_mri_y1 = real_images[0]
            real_brain_mri_y2 = real_images[1]
        else:
            print(" --> Only ONE..")
            real_brain_mri_y1 = real_images
        n, _, _, _ = real_brain_mri_y1.shape
        num_iter = int(np.ceil(n / batch_size))
        pred_dem = None
        for i in range(num_iter):
            batch_imgs = real_brain_mri_y1[i * batch_size:(i + 1) * batch_size,:,:,:]
            z_prior, mu_prior, sigma_prior = self.prior(batch_imgs, training=False)
            temp_result = self.generator([batch_imgs, z_prior], training=False)
            if pred_dem is None:
                pred_dem = temp_result
            else:
                pred_dem = np.concatenate((pred_dem, temp_result), axis=0)
        return pred_dem

    def test_step(self, real_tuples):
        if isinstance(real_tuples, tuple):
            real_brain_mri = real_tuples[0]
            real_dem = real_tuples[1]
        
        if len(real_brain_mri) > 1:
            real_brain_mri_y1 = real_brain_mri[0]
            real_brain_mri_y2 = real_brain_mri[1]
            z_prior, mu_prior, sigma_prior = self.prior(real_brain_mri_y1, training=False)
            pred_dem = self.generator([real_brain_mri_y1, z_prior], training=False)
        else:
            z_prior, mu_prior, sigma_prior = self.prior(real_brain_mri, training=False)
            pred_dem = self.generator([real_brain_mri, z_prior], training=False)
        val_loss = tf.reduce_mean(tf.reduce_sum(self.focal_val_loss(real_dem, pred_dem)))
        return {"loss": val_loss}

    def train_step(self, real_tuples):
        if isinstance(real_tuples, tuple):
            # print("real_tuples: ", real_tuples)
            real_brain_mri = real_tuples[0]
            real_dem = real_tuples[1]

        if len(real_brain_mri) > 1:
            real_brain_mri_y1 = real_brain_mri[0]
            real_brain_mri_y2 = real_brain_mri[1]

            # Get the batch size
            batch_size = tf.shape(real_brain_mri_y1)[0]
        else:            
            # Get the batch size
            batch_size = tf.shape(real_brain_mri)[0]

        # Train the Prior and Posterior
        with tf.GradientTape(persistent=True) as tape_:
            _, mu_prior, sigma_prior = self.prior(real_brain_mri_y1, training=True)
            _, mu_posterior, sigma_posterior = self.posterior([real_brain_mri_y2, real_dem], training=True)
            kl_loss = self.kl_score(mu_posterior, sigma_posterior, mu_prior, sigma_prior)

        grad_prior = tape_.gradient(kl_loss, self.prior.trainable_weights)
        self.prior_opt.apply_gradients(zip(grad_prior, self.prior.trainable_weights))
        grad_posterior = tape_.gradient(kl_loss, self.posterior.trainable_weights)
        self.posterior_opt.apply_gradients(zip(grad_posterior, self.posterior.trainable_weights))
        
        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                z_posterior, _,_ = self.posterior([real_brain_mri_y2,real_dem])
                pred_dem = self.generator([real_brain_mri_y1, z_posterior], training=True)
                # Concat real brain MRI with predicted DEM
                fake_pairs = tf.concat([real_brain_mri_y1, real_brain_mri_y2, pred_dem], -1)
                real_pairs = tf.concat([real_brain_mri_y1, real_brain_mri_y2, real_dem], -1)

                # Combine them with real images
                combined_images = tf.concat([fake_pairs, real_pairs], axis=0)

                # Assemble labels discriminating real from fake images
                labels = tf.concat(
                    [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
                )
                # Add random noise to the labels - important trick!
                labels += 0.05 * tf.random.uniform(tf.shape(labels))

                # Get the logits for the combined_images
                predictions = self.discriminator(combined_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_loss = self.d_loss_fn(labels, predictions)

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
        
        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator
        with tf.GradientTape() as tape:
            z_posterior, _,_ = self.posterior([real_brain_mri_y2,real_dem])

            # Generate fake images using the generator
            pred_dem = self.generator([real_brain_mri_y1, z_posterior], training=True)
            fcl_loss = tf.reduce_mean(tf.reduce_sum(self.focal_val_loss(real_dem, pred_dem)))

            # Concat real brain MRI with predicted DEM
            fake_pairs = tf.concat([real_brain_mri_y1, real_brain_mri_y2, pred_dem], -1)
            real_pairs = tf.concat([real_brain_mri_y1, real_brain_mri_y2, real_dem], -1)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(fake_pairs, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(misleading_labels, gen_img_logits, fake_imgs=fake_pairs, real_imgs=real_pairs)

            # fcl_loss = self.focal_val_loss(real_dem, pred_dem)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss, "fcl_loss": fcl_loss}

    def latent_space_net(self, img_shape, seg_shape, latent_dim): 
        if seg_shape is not None:
            # Posterior inputs
            inputs = [Input(shape=img_shape), Input(shape=seg_shape)]
            input_ = Concatenate(name='input_con') (inputs)
            name = 'prob_unet_posterior'
        else:
            # Prior input
            inputs = Input(shape=img_shape)
            input_ = inputs
            name = 'prob_unet_prior'
        
        # Encoder blocks
        for i in range(self.resolution_lvl):
            if i == 0:
                x = conv_block(self.num_filters[i], 0, i, amount=2, type_block='encoder_latent') (input_)
            else:
                x = MaxPool2D(pool_size=self.downsample_signal[i], 
                              name='encoder_latent_stage0-{}_pool'.format(i)) (x)
                x = conv_block(self.num_filters[i], 0, i, amount=2, type_block='encoder_latent') (x)
        
        # Z sample
        z, mu, sigma = z_mu_sigma(latent_dim, 0, self.resolution_lvl+1) (x)
        return keras.Model(inputs, [z, mu, sigma], name=name)

    # FOR PROB-UNET
    def unet(self, img_shape, latent_dim):
        lvl_div = np.power(2, self.resolution_lvl-1)
        z_sample = Input(shape=(None, None, latent_dim))
        inputs = Input(shape=img_shape)
        skip_connections = [None] * self.resolution_lvl
        
        # Encoder blocks
        for i in range(self.resolution_lvl):
            if i == 0:
                x = conv_block(self.num_filters[i], 0, i, amount=2, type_block='encoder') (inputs)
            else:
                x = MaxPool2D(pool_size=self.downsample_signal[i], 
                              name='encoder_stage0-{}_pool'.format(i)) (x)
                x = conv_block(self.num_filters[i], 0, i, amount=2, type_block='encoder') (x)
            
            skip_connections[i] = x
        skip_connections = skip_connections[:-1]
        
        # Decoder blocks
        for i in reversed(range(self.resolution_lvl-1)):
            x = UpSampling2D(size=self.downsample_signal[i], 
                            name='decoder_stage0-{}_up'.format(i)) (x)
            x = Concatenate(name='decoder_stage0-{}_con'.format(i)) ([x, skip_connections[i]])
            x = conv_block(self.num_filters[i], 0, i, amount=2, type_block='decoder') (x)
        
        # Concatenate U-Net and Z sample
        broadcast_z = tf.tile(z_sample, (1, lvl_div, lvl_div, 1))
        
        x = Concatenate(name='final_con') ([x, broadcast_z])
        x = conv_block(self.num_filters[0], 0, i, amount=2, type_block='final') (x)
        if self.n_label == 1:
            x = conv_block(1, 0, i, amount=1, kernel_size=1, type_block='final_sig',
                        type_act='sigmoid', use_batchnorm=False) (x)
            model = keras.Model([inputs, z_sample], x, name='prob_unet')
        elif self.n_label > 1:
            x = conv_block(self.n_label, 0, i, amount=1, kernel_size=1, type_block='final_sig',
                        type_act='softmax', use_batchnorm=False) (x)
            model = keras.Model([inputs, z_sample], x, name='prob_unet')
        return model

    def summary(self):
        print("")
        print("-- SUMMARY of the PRIOR --")
        self.prior.summary()
        print("")
        print("-- SUMMARY of the POSTERIOR --")
        self.posterior.summary()
        print("")
        print("-- SUMMARY of the GENERATOR --")
        self.generator.summary()
        print("")
        print("-- SUMMARY of the DISCRIMINATOR --")
        self.discriminator.summary()

    def kl_score(self, mu0, sigma0, mu1, sigma1):
        # Calculate kl loss
        sigma0_f = K.square(K.flatten(sigma0))
        sigma1_f = K.square(K.flatten(sigma1))
        logsigma0 = K.log(sigma0_f + 1e-10)
        logsigma1 = K.log(sigma1_f + 1e-10)
        mu0_f = K.flatten(mu0)
        mu1_f = K.flatten(mu1)

        return tf.reduce_mean(
            0.5*tf.reduce_sum(tf.divide(sigma0_f + tf.square(mu1_f - mu0_f), sigma1_f + 1e-10) 
            + logsigma1 - logsigma0 - 1))

    
    def kl_score(self, mu0, sigma0, mu1, sigma1):
        # Calculate kl loss
        sigma0_f = K.square(K.flatten(sigma0))
        sigma1_f = K.square(K.flatten(sigma1))
        logsigma0 = K.log(sigma0_f + 1e-10)
        logsigma1 = K.log(sigma1_f + 1e-10)
        mu0_f = K.flatten(mu0)
        mu1_f = K.flatten(mu1)

        return tf.reduce_mean(
            0.5*tf.reduce_sum(tf.divide(sigma0_f + tf.square(mu1_f - mu0_f), sigma1_f + 1e-10) 
            + logsigma1 - logsigma0 - 1))

