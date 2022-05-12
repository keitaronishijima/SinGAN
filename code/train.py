import numpy as np
from PIL import Image
from model import Generator, Discriminator
import tensorflow as tf

class Trainer:
    def __init__(self):
        
        self.max_size = 1000
        self.min_size = 25
        self.noise_amp = 0.1
        self.epoch_num = 4
        self.num_scales = 8
        self.num_iters = 1
        self.reconstruction_weight = 10
        self.num_filters = [pow(2, (s//4)) * 32 for s in range(self.num_scales)] # num_filters double for every 4 scales
        self.checkpoint_dir = "./training_checkpoints"
        self.G_dir = './training_checkpoints/G'
        self.D_dir = './training_checkpoints/D'
        self.learning_rate = 5e-4
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5, beta_2=0.999)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5, beta_2=0.999)
        self.z_s = []
        self.noise = []
        self.generators = []
        self.discriminators = []
        for scale in range(self.num_scales):
            self.generators.append(Generator(num_filters=self.num_filters[scale]))
            self.discriminators.append(Discriminator(num_filters=self.num_filters[scale]))

    def train(self, training_image):
        # load image first
        input_image = tf.image.resize(tf.cast(tf.image.decode_png(tf.io.read_file(training_image), channels=3), tf.float32), (self.max_size, self.max_size))[tf.newaxis, ...]
        # Normalize image
        input_image = input_image / 127.5 - 1 
        
        # This is to build a pyramid of input images in different sizes
        real_input_images = [input_image]
        for i in range(1, self.num_scales):
            real_input_images.append(self.resize_image(input_image, min_size=self.min_size, scale_factor=pow(0.75, i)))
        real_input_images.reverse()
        # finish building pyramids

        for scale_idx in range(self.num_scales):
            print("Current scale is", scale_idx)
            prev_image = tf.zeros_like(real_input_images[scale_idx])
            for step in range(self.num_iters):
                z, prev_image, self.noise_amp = self.train_one_itr(real_input_images, prev_image, self.noise_amp, scale_idx, step)
            self.noise.append(self.noise_amp)
            self.z_s.append(z)
            self.save_model(scale_idx)

    def train_one_itr(self, real_input_images, prev_image, noise_amp, scale, step):
        real_input = real_input_images[scale]
        z_rand = tf.random.normal(real_input.shape)

        z_rec = tf.zeros_like(real_input)

        for epoch in range(self.epoch_num):
            if epoch == 0 and tf.equal(step, 0):
                prev_image = self.generate_small(scale, real_input_images, 'recreate')
                prev_noise = self.generate_small(scale, real_input_images, 'random')
                stand_dev = tf.sqrt(tf.reduce_mean(tf.square(real_input - prev_image)))
                noise_amp = self.noise_amp * stand_dev
            else:
                prev_noise = self.generate_small(scale, real_input_images, 'random')

            if scale == 0:
                Z_rand = z_rand
            else:
                Z_rand = noise_amp * z_rand
            Z_rec = noise_amp * z_rec
            if epoch < 2:
                with tf.GradientTape() as tape:
                    fake_rand_img = self.generators[scale](prev_noise, Z_rand)
                    dis_loss = self.dicriminator_loss(self.discriminators[scale], real_input, fake_rand_img)
                dis_gradients = tape.gradient(dis_loss, self.discriminators[scale].trainable_variables)
                self.discriminator_optimizer.apply_gradients(zip(dis_gradients, self.discriminators[scale].trainable_variables))
            else:
                with tf.GradientTape() as tape:
                    fake_rand_img = self.generators[scale](prev_noise, Z_rand)
                    fake_rec_img = self.generators[scale](prev_image, Z_rec)
                    rec_loss = self.reconstruction_loss(real_input, fake_rec_img)
                    gen_loss = self.generator_loss(self.discriminators[scale], fake_rand_img)
                    gen_loss = gen_loss + self.reconstruction_weight * rec_loss
                gen_gradients = tape.gradient(gen_loss, self.generators[scale].trainable_variables)
                self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generators[scale].trainable_variables))

        return z_rec, prev_image, noise_amp

    def generate_small(self, scale, real_images, mode):
        fake_image = tf.zeros_like(real_images[0])
        if mode == 'recreate':
            for i in range(scale):
                z = self.z_s[i] * self.noise[i]
                fake_image = self.generators[i](fake_image, z)
                fake_image = self.resize_image(fake_image, new_shapes=real_images[i+1].shape)
        elif mode == 'random':
            for i in range(scale):
                z_rand = tf.random.normal(real_images[i].shape)
                z_rand = z_rand * self.noise[i]
                fake_image = self.generators[i](fake_image, z_rand)
                fake_image = self.resize_image(fake_image, new_shapes=real_images[i+1].shape)
    
        return fake_image

    def generator_loss(self, discriminator, fake):
        return -tf.reduce_mean(discriminator(fake))
    
    def dicriminator_loss(self, discriminator, real_image, fake_image):
        dis_loss = tf.reduce_mean(discriminator(fake_image)) - tf.reduce_mean(discriminator(real_image))
        return dis_loss   

    def reconstruction_loss(self, real, fake_rec):
        return tf.reduce_mean(tf.square(fake_rec - real))    
            
    # Utility functions below

    def resize_image(self, image, min_size=0, scale_factor=None, new_shapes=None):
        if new_shapes:
            new_h = new_shapes[1]
            new_w = new_shapes[2]
        if scale_factor is not None:
            new_h = np.maximum(min_size, 
                                    tf.cast(image.shape[1]*scale_factor, tf.int32))
            new_w = np.maximum(min_size, 
                                tf.cast(image.shape[2]*scale_factor, tf.int32))

        image = tf.image.resize(
                    image, 
                    (new_h, new_w),
                )
        return image
    
    def save_model(self, scale_index):
        G_dir = self.G_dir + f'{scale_index}'
        D_dir = self.D_dir + f'{scale_index}'

        self.generators[scale_index].save_weights(G_dir + '/G', save_format='tf')
        self.discriminators[scale_index].save_weights(D_dir + '/D', save_format='tf')
        
        # Saving noise array in a file
        np.save(self.checkpoint_dir + '/NoiseAmp', self.noise)

    