import tensorflow as tf
from ConvBlock import ConvBlock

class Generator(tf.keras.models.Model):
    def __init__(self, num_filters):
        super(Generator, self).__init__()
        self.initializer = tf.random_normal_initializer(0.0, 0.02)
        self.padding_layer = tf.keras.layers.ZeroPadding2D(5)
        self.conv0 = ConvBlock(num_filters)
        self.conv1 = ConvBlock(num_filters)
        self.conv2 = ConvBlock(num_filters)
        self.conv3 = ConvBlock(num_filters)
        self.bottom = tf.keras.layers.Conv2D(filters=3,
                           kernel_size=3,
                           strides=1,
                           padding='valid',
                           activation='tanh',
                           kernel_initializer=self.initializer)

    def call(self, prev_image, noise_image):
        prev_padding = self.padding_layer(prev_image)
        noise_padding = self.padding_layer(noise_image)
        out = tf.keras.layers.Add()([prev_padding, noise_padding])
        out = self.conv0(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bottom(out)
        out = tf.keras.layers.Add()([out, prev_image])

        return out


class Discriminator(tf.keras.models.Model):
    def __init__(self, num_filters):
        super(Discriminator, self).__init__()
        self.initializer = tf.random_normal_initializer(0.0, 0.02)
        self.conv0 = ConvBlock(num_filters)
        self.conv1 = ConvBlock(num_filters)
        self.conv2 = ConvBlock(num_filters)
        self.conv3 = ConvBlock(num_filters)
        self.bottom = tf.keras.layers.Conv2D(filters=1,
                           kernel_size=3,
                           strides=1,
                           padding='valid',
                           kernel_initializer=self.initializer)

    def call(self, out):
        out = self.conv0(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bottom(out)

        return out