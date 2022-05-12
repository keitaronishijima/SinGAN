import os
import numpy as np
import tensorflow as tf

from PIL import Image
from model import Generator


class Inferencer:
    def __init__(self):

        self.model = []
        self.NoiseAmp = []
        self.load_model("../training_checkpoints")
        self.num_samples = 10
        self.inject_scale = 0
        self.result_dir = "../results"


    def load_model(self, checkpoint_dir):
        """ Load generators and NoiseAmp from checkpoint_dir """
        self.NoiseAmp = np.load(checkpoint_dir + '/NoiseAmp.npy')
        dir = os.walk(checkpoint_dir)
        for path, dir_list, _ in dir:
            for dir_name in dir_list:
                network = dir_name[0]                
                scale = int(dir_name[1])
                if network == 'G':
                    generator = Generator(num_filters=32*pow(2, (scale//4)))
                    generator.load_weights(os.path.join(path, dir_name) + '/G').expect_partial()    # Silence the warning
                    self.model.append(generator)


    def inference(self, mode, reference_image, image_size=250):
        """ Use SinGAN to do inference
        mode : Inference mode
        reference_image : Input image name
        image_size : Size of output image
        """
        reference_image = self.load_image(reference_image, image_size=image_size)
        # Normalize image between -1 to 1
        reference_image = reference_image / 127.5 - 1 
        reals = self.create_real_pyramid(reference_image, num_scales=len(self.model))

        dir = self.create_dir(os.path.join(self.result_dir, mode))
        if mode == 'random_sample':
            z_fixed = tf.random.normal(reals[0].shape)
            for n in range(self.num_samples):
                fake = self.SinGAN_generate(reals, z_fixed, inject_scale=self.inject_scale)
                self.imsave(fake, dir + f'/random_sample_{n}.jpg') 

        elif (mode == 'harmonization') or (mode == 'editing') or (mode == 'paint2image'):
            fake = self.SinGAN_inject(reals, inject_scale=self.inject_scale)
            self.imsave(fake, dir + f'/inject_at_{self.inject_scale}.jpg') 

        else:
            print('Inference mode must be: random_sample, harmonization, paint2image, editing')


    def SinGAN_inject(self, reals, inject_scale=1):
        """ Inject reference image on given scale (inject_scale should > 0)"""
        fake = reals[inject_scale]

        for scale in range(inject_scale, len(reals)):
            fake = self.imresize(fake, new_shapes=reals[scale].shape)
            z = tf.random.normal(fake.shape)
            z = z * self.NoiseAmp[scale]
            fake = self.model[scale](fake, z)
    
        return fake


    @tf.function
    def SinGAN_generate(self, reals, z_fixed, inject_scale=0):
        """ Use fixed noise to generate before start_scale """
        fake = tf.zeros_like(reals[0])
    
        for scale, generator in enumerate(self.model):
            fake = self.imresize(fake, new_shapes=reals[scale].shape)
            
            if scale > 0:
                z_fixed = tf.zeros_like(fake)

            if scale < inject_scale:
                z = z_fixed
            else:
                z = tf.random.normal(fake.shape)
            z = z * self.NoiseAmp[scale]
            fake = generator(fake, z)
            print(scale)

        return fake


    def create_real_pyramid(self, real_image, num_scales):
        """ Create the pyramid of scales """
        reals = [real_image]
        for i in range(1, num_scales):
            reals.append(self.imresize(real_image, scale_factor=pow(0.75, i)))
        
        """ Reverse it to coarse-fine scales """
        reals.reverse()
        for real in reals:
            print(real.shape)
        return reals
    
    def load_image(self, image, image_size=None):
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
    
    def create_dir(self, dir):
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


    def imsave(self, image, path_to_image):
        """ Expected input values [-1, 1] """
        image = (image + 1) * 127.5
        image = Image.fromarray(np.array(image).astype(np.uint8).squeeze())
        image.save(path_to_image)
