import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from model import Generator, Discriminator

class Trainer:
    def __init__(self):

        self.num_scales = 8
        self.num_iters = 1
        self.num_filters = [32*pow(2, (s//4)) for s in range(self.num_scales)] # num_filters double for every 4 scales
        self.max_size = 1000
        self.min_size = 25
        self.noise_amp = 0.1

        self.checkpoint_dir = "./training_checkpoints"
        self.G_dir = self.checkpoint_dir + '/G'
        self.D_dir = self.checkpoint_dir + '/D'
        self.learning_rate = 5e-4
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5, beta_2=0.999)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5, beta_2=0.999)
        self.dis_metric = tf.keras.metrics.Mean()
        self.gen_metric = tf.keras.metrics.Mean()
        self.rec_metric = tf.keras.metrics.Mean()
        self.Z_fixed = []
        self.noise = []

        self.create_directory(self.checkpoint_dir)
        self.generators = []
        self.discriminators = []
        for scale in range(self.num_scales):
            self.generators.append(Generator(num_filters=self.num_filters[scale]))
            self.discriminators.append(Discriminator(num_filters=self.num_filters[scale]))


    def save_model(self, scale):
        """ Save weights and NoiseAmp """
        G_dir = self.G_dir + f'{scale}'
        D_dir = self.D_dir + f'{scale}'
        if os.path.exists(G_dir) is None:
            os.makedirs(G_dir)
        if os.path.exists(D_dir) is None:
            os.makedirs(D_dir)

        self.generators[scale].save_weights(G_dir + '/G', save_format='tf')
        self.discriminators[scale].save_weights(D_dir + '/D', save_format='tf')
        np.save(self.checkpoint_dir + '/NoiseAmp', self.noise)


    def train(self, training_image):
        """ Training """
        # load image first
        input_image = self.load_image(training_image, image_size=self.max_size)
        # Normalize image below
        input_image = input_image / 127.5 - 1 
        
        # This is to build a pyramid of input images in different sizes
        real_input_images = [input_image]
        for i in range(1, self.num_scales):
            real_input_images.append(self.imresize(input_image, min_size=self.min_size, scale_factor=pow(0.75, i)))
        real_input_images.reverse()
        # finish building pyramids

        for scale in range(self.num_scales):
            print(scale)
            prev_image = tf.zeros_like(real_input_images[scale])
            train_step = self.wrapper_func()
            for step in range(self.num_iters):
                z_fixed, prev_image, self.noise_amp = train_step(real_input_images, prev_image, self.noise_amp, scale, step)
            self.noise.append(self.noise_amp)
            self.save_model(scale)
            self.Z_fixed.append(z_fixed)

    def wrapper_func(self):
        @tf.function
        def train_step(real_input_images, prev_image, noise_amp, scale, step):
            real_input = real_input_images[scale]
            z_rand = tf.random.normal(real_input.shape)

            if scale == 0:
                z_rec = tf.random.normal(real_input.shape)
            else:
                z_rec = tf.zeros_like(real_input)

            for i in range(6):
                if i == 0 and tf.equal(step, 0):
                    if scale == 0:
                        """ Coarsest scale is purely generative """
                        prev_noise = tf.zeros_like(real_input)
                        prev_image = tf.zeros_like(real_input)
                        noise_amp = 1.0
                    else:
                        """ Finer scale takes noise and image generated from previous scale as input """
                        prev_noise = self.generate_from_coarsest(scale, real_input_images, 'rand')
                        prev_image = self.generate_from_coarsest(scale, real_input_images, 'rec')
                        """ Compute the standard deviation of noise """
                        stand_dev = tf.sqrt(tf.reduce_mean(tf.square(real_input - prev_image)))
                        noise_amp = self.noise_amp * stand_dev
                else:
                    prev_noise = self.generate_from_coarsest(scale, real_input_images, 'rand')

                Z_rand = z_rand if scale == 0 else noise_amp * z_rand
                Z_rec = noise_amp * z_rec
                
                if i < 3:
                    with tf.GradientTape() as tape:
                        """ Only record the training variables """
                        fake_rand = self.generators[scale](prev_noise, Z_rand)

                        dis_loss = self.dicriminator_wgan_loss(self.discriminators[scale], real_input, fake_rand, 1)
    
                    dis_gradients = tape.gradient(dis_loss, self.discriminators[scale].trainable_variables)
                    self.discriminator_optimizer.apply_gradients(zip(dis_gradients, self.discriminators[scale].trainable_variables))
                else:
                    with tf.GradientTape() as tape:
                        """ Only record the training variables """
                        fake_rand = self.generators[scale](prev_noise, Z_rand)
                        fake_rec = self.generators[scale](prev_image, Z_rec)

                        gen_loss = self.generator_wgan_loss(self.discriminators[scale], fake_rand)
                        rec_loss = self.reconstruction_loss(real_input, fake_rec)
                        gen_loss = gen_loss + 10 * rec_loss

                    gen_gradients = tape.gradient(gen_loss, self.generators[scale].trainable_variables)
                    self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generators[scale].trainable_variables))

            return z_rec, prev_image, noise_amp
        return train_step


    def generate_from_coarsest(self, scale, real_images, mode):
        """ Use random/fixed noise to generate from coarsest scale"""
        fake_image = tf.zeros_like(real_images[0])
        if mode == 'rec':
            for i in range(scale):
                z_fixed = self.noise[i] * self.Z_fixed[i]
                fake_image = self.generators[i](fake_image, z_fixed)
                fake_image = self.imresize(fake_image, new_shapes=real_images[i+1].shape)
        elif mode == 'rand':
            for i in range(scale):
                z_rand = tf.random.normal(real_images[i].shape)
                z_rand = self.noise[i] * z_rand
                fake_image = self.generators[i](fake_image, z_rand)
                fake_image = self.imresize(fake_image, new_shapes=real_images[i+1].shape)
    
        return fake_image


    def build_real_image_pyramids(self, input_image):
        """ Create the pyramid of scales """
        reals = [input_image]
        for i in range(1, self.num_scales):
            reals.append(self.imresize(input_image, min_size=self.min_size, scale_factor=pow(0.75, i)))
        reals.reverse()
        for real in reals:
            print(real.shape)
        return reals


    def generator_wgan_loss(self, discriminator, fake):
        """ Ladv(G) = -E[D(fake)] """
        return -tf.reduce_mean(discriminator(fake))


    def reconstruction_loss(self, real, fake_rec):
        """ Lrec = || G(z*) - real ||^2 """
        return tf.reduce_mean(tf.square(fake_rec - real))

 
    def dicriminator_wgan_loss(self, discriminator, real, fake, batch_size=1):
        """ Ladv(D) = E[D(fake)] - E[D(real)] + GradientPenalty"""
        dis_loss = tf.reduce_mean(discriminator(fake)) - tf.reduce_mean(discriminator(real))

        alpha = tf.random.uniform(shape=[batch_size,1,1,1], minval=0., maxval=1.)# real.shape
        interpolates = alpha * real + ((1 - alpha) * fake)
        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            dis_interpolates = discriminator(interpolates)
        gradients = tape.gradient(dis_interpolates, [interpolates])[0]

        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[3])) # compute pixelwise gradient norm; per image use [1, 2, 3]
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        dis_loss = dis_loss + 0.1 * gradient_penalty
        return dis_loss       
            
    # Util functions below
            
    def load_image(self, image, image_size):
        """Load an image from directory into a tensor shape of [1,H,W,C] and value between [0, 255]
        image : Directory of image
        image_size : An integer number
        """
        image = tf.io.read_file(image)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32)
        # image = tf.image.convert_image_dtype(image, tf.float32)   # to [0, 1]

        if image_size:
            image = tf.image.resize(image, (image_size, image_size),
                                    method=tf.image.ResizeMethod.BILINEAR,
                                    antialias=True,
                                    preserve_aspect_ratio=True
                                    )
        return image[tf.newaxis, ...]
    
    def create_directory(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
            print(f'Directory {dir} createrd')
        else:
            print(f'Directory {dir} already exists')  

        return dir

    def imresize(self, image, min_size=0, scale_factor=None, new_shapes=None):
        """ Expect input shapes [B, H, W, C] """
        if new_shapes:
            new_height = new_shapes[1]
            new_width = new_shapes[2]

        elif scale_factor:
            new_height = tf.maximum(min_size, 
                                    tf.cast(image.shape[1]*scale_factor, tf.int32))
            new_width = tf.maximum(min_size, 
                                tf.cast(image.shape[2]*scale_factor, tf.int32))

        image = tf.image.resize(
                    image, 
                    (new_height, new_width),
                    method=tf.image.ResizeMethod.BILINEAR,
                    antialias=True
                )
        return image

    