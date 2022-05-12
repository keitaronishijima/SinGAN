from doctest import script_from_examples
import os
import numpy as np
import tensorflow as tf

from PIL import Image
from model import Generator


class Inferencer:
    def __init__(self):

        self.model = []
        self.noise = []
        self.load_model("../training_checkpoints")
        self.num_samples = 10
        self.inject_scale = 0
        self.result_dir = "../results"


    def load_model(self, checkpoint_dir):
        """ Load generators and NoiseAmp from checkpoint_dir """
        self.noise = np.load(checkpoint_dir + '/NoiseAmp.npy')
        directory = os.walk(checkpoint_dir)
        for path, dir_list, _ in dir:
            for name in dir_list:
                network = name[0]                
                scale = int(name[1])
                if network == 'G':
                    generator = Generator(num_filters=32*pow(2, (scale//4)))
                    generator.load_weights(os.path.join(path, name) + '/G').expect_partial() 
                    self.model.append(generator)


    def inference(self, mode, reference_image, image_size=250):
        """ Use SinGAN to do inference
        mode : Inference mode
        reference_image : Input image name
        image_size : Size of output image
        """
        rimg = self.load_image(reference_image, image_size=image_size) / 127.5 - 1 
        reimgs = self.make_pyramid(rimg, len(self.model))

        dir =  "../results"
        if mode == 'random_sample':
            z_fixed = tf.random.normal(reimgs[0].shape)
            for n in range(self.num_samples):
                fake = self.SinGAN_generate(reimgs, z_fixed, inject_scale=self.inject_scale)
                self.imsave(fake, dir + f'/random_sample_{n}.jpg') 

        elif (mode == 'harmonization') or (mode == 'editing') or (mode == 'paint2image'):
            fake = self.SinGAN_inject(reimgs, inject_scale=self.inject_scale)
            self.imsave(fake, dir + f'/inject_at_{self.inject_scale}.jpg') 

        else:
            print('Inference mode must be: random_sample, harmonization, paint2image, editing')


    def SinGAN_inject(self, reals, inject_scale=1):
        """ Inject reference image on given scale (inject_scale should > 0)"""
        fake = reals[inject_scale]

        for scale in range(inject_scale, len(reals)):
            fake = self.imresize(fake, new_shapes=reals[scale].shape)
            z = tf.random.normal(fake.shape)
            z = z * self.noise[scale]
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
            z = z * self.noise[scale]
            fake = generator(fake, z)
            print(scale)

        return fake


    def make_pyramid(self, timg, scales):
        """ Create the pyramid of scales """
        pyramid = [timg]
        for scale in range(1, scales):
            scaledimg = self.imresize(script_from_examples, scale_factor=pow(0.75, scale))
            pyramid.append(scaledimg)
        
        """ Reverse it to coarse-fine scales """
        pyramid.reverse()
        for img in pyramid:
            print("Image shape")
            print(img.shape)
        return pyramid
    
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


    def imsave(self, image, path_to_image):
        """ Expected input values [-1, 1] """
        image = (image + 1) * 127.5
        image = Image.fromarray(np.array(image).astype(np.uint8).squeeze())
        image.save(path_to_image)
