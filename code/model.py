import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, ReLU, LeakyReLU
from tensorflow.keras.layers import Add, Lambda, ZeroPadding2D
from ConvBlock import ConvBlock

class Generator(Model):
    def __init__(self, num_filters):
        super(Generator, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.padding = ZeroPadding2D(5)
        self.head = ConvBlock(num_filters)
        self.convblock1 = ConvBlock(num_filters)
        self.convblock2 = ConvBlock(num_filters)
        self.convblock3 = ConvBlock(num_filters)
        self.tail = Conv2D(filters=3,
                           kernel_size=3,
                           strides=1,
                           padding='valid',
                           activation='tanh',
                           kernel_initializer=self.initializer)

    def call(self, prev, noise):
        prev_pad = self.padding(prev)
        noise_pad = self.padding(noise)
        out = Add()([prev_pad, noise_pad])
        out = self.head(out)
        out = self.convblock1(out)
        out = self.convblock2(out)
        out = self.convblock3(out)
        out = self.tail(out)
        out = Add()([out, prev])

        return out


class Discriminator(Model):
    def __init__(self, num_filters):
        super(Discriminator, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.head = ConvBlock(num_filters)
        self.convblock1 = ConvBlock(num_filters)
        self.convblock2 = ConvBlock(num_filters)
        self.convblock3 = ConvBlock(num_filters)
        self.tail = Conv2D(filters=1,
                           kernel_size=3,
                           strides=1,
                           padding='valid',
                           kernel_initializer=self.initializer)

    def call(self, out):
        out = self.head(out)
        out = self.convblock1(out)
        out = self.convblock2(out)
        out = self.convblock3(out)
        out = self.tail(out)

        return out