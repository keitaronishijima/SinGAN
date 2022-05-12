from doctest import script_from_examples
from logging.handlers import TimedRotatingFileHandler
import os
import numpy as np
import tensorflow as tf

from PIL import Image
from model import Generator


class Inferencer:
    def __init__(self):

        self.model = []
        self.noise = []
        self.modelload("../training_checkpoints")
        self.num_samples = 10
        self.inject_scale = 0
        self.result_dir = "../results"


    def modelload(self, checkpoint_dir):
        self.noise = np.load(checkpoint_dir + '/NoiseAmp.npy')
        directory = os.walk(checkpoint_dir)
        for path, dir_list, _ in directory:
            for name in dir_list:
                network = name[0]                
                scale = int(name[1])
                if network == 'G':
                    generator = Generator(num_filters=32*pow(2, (scale//4)))
                    generator.load_weights(os.path.join(path, name) + '/G').expect_partial() 
                    self.model.append(generator)


    def inference(self, task, rimg, size=250):
        rimg = self.imloader(rimg, size=size) / 127.5 - 1 
        reimgs = self.make_pyramid(rimg, len(self.model))
        
        if (task == 'harmonization') or (task == 'editing') or (task == 'paint2image'):
            genimg = self.gIn(reimgs, inject_scale=self.inject_scale)
            self.saveimg(genimg, '../results' + f'/inject_at_{self.inject_scale}.jpg') 
        elif task == 'random_sample':
            for i in range(self.num_samples):
                genimg = self.generate(reimgs, tf.random.normal(reimgs[0].shape), inject_scale=self.inject_scale)
                self.saveimg(genimg, '../results' + f'/random_sample_number_{i}.jpg') 
        else:
            print('Please select task as one of random_sample, harmonization, paint2image or editing')


    def gIn(self, rimgs, iscale=1):
        genimg = rimgs[iscale]
        tscales = len(rimgs)
        for scale in range(iscale, tscales):
            genimg = self.resizeimg(genimg, nshapes=rimgs[scale].shape)
            i = tf.random.normal(genimg.shape) * self.noise[scale]
            genimg = self.model[scale](genimg, i)
        return genimg


    @tf.function
    def generate(self, reals, z_fixed, inject_scale=0):
        genimg = tf.zeros_like(reals[0])
        for scale, generator in enumerate(self.model):
            genimg = self.resizeimg(genimg, nshapes=reals[scale].shape)  
            if scale > 0:
                r = tf.zeros_like(genimg)
            if scale < inject_scale:
                i = r
            else:
                i = tf.random.normal(genimg.shape)
            i = i * self.noise[scale]
            genimg = generator(genimg, i)
            print("Scale:")
            print(scale)

        return genimg


    def make_pyramid(self, timg, scales):
        pyramid = [timg]
        for scale in range(1, scales):
            scaledimg = self.resizeimg(timg, sfactor=pow(0.75, scale))
            pyramid.append(scaledimg)
        pyramid.reverse()
        for img in pyramid:
            print("Image shape")
            print(img.shape)
        return pyramid
    
    def imloader(self, img, size=None):
        img = tf.cast(tf.image.decode_png(tf.io.read_file(img), channels=3), tf.float32)
    
        if size is not None:
            img = tf.image.resize(img, (size, size),method=tf.image.ResizeMethod.BILINEAR,antialias=True,preserve_aspect_ratio=True)
        return img[tf.newaxis, ...]
    
    def mkdir(self, dir):
        if os.path.exists(dir):
            print("This directory already exists")  
        else:
            os.makedirs(dir)
            print("Your directory was created")  
        return dir

    def resizeimg(self, image, msize=0, sfactor=None, nshapes=None):
        if nshapes:
            h, w = nshapes[1],nshapes[2]
        if sfactor is not None:
            h = np.maximum(msize, tf.cast(image.shape[1]*sfactor, tf.int32))
            w = np.maximum(msize, tf.cast(image.shape[2]*sfactor, tf.int32))
        nimg = tf.image.resize(image, (h, w),)
        return nimg


    def saveimg(self, img, path):
        img = Image.fromarray(np.array((img + 1) * 127.5).astype(np.uint8).squeeze())
        img.save(path)
