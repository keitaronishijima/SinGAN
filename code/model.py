import tensorflow as tf
from ConvBlock import ConvBlock

class Generator(tf.keras.models.Model):
    def __init__(self, num_filters):
        super(Generator, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.padding_layer = tf.keras.layers.ZeroPadding2D(5)
        self.head = ConvBlock(num_filters)
        self.convblock1 = ConvBlock(num_filters)
        self.convblock2 = ConvBlock(num_filters)
        self.convblock3 = ConvBlock(num_filters)
        self.tail = tf.keras.layers.Conv2D(filters=3,
                           kernel_size=3,
                           strides=1,
                           padding='valid',
                           activation='tanh',
                           kernel_initializer=self.initializer)

    def call(self, prev_image, noise_image):
        prev_pad = self.padding_layer(prev_image)
        noise_pad = self.padding_layer(noise_image)
        out = tf.keras.layers.Add()([prev_pad, noise_pad])
        out = self.head(out)
        out = self.convblock1(out)
        out = self.convblock2(out)
        out = self.convblock3(out)
        out = self.tail(out)
        out = tf.keras.layers.Add()([out, prev_image])

        return out


class Discriminator(tf.keras.models.Model):
    def __init__(self, num_filters):
        super(Discriminator, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.head = ConvBlock(num_filters)
        self.convblock1 = ConvBlock(num_filters)
        self.convblock2 = ConvBlock(num_filters)
        self.convblock3 = ConvBlock(num_filters)
        self.tail = tf.keras.layers.Conv2D(filters=1,
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