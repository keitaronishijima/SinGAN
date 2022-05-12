import tensorflow as tf
from batch_norm import batch_norm

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters):
        super(ConvBlock, self).__init__()
        self.num_filters = num_filters
        self.initializer = tf.random_normal_initializer(0.0, 0.02)
        self.conv2D = tf.keras.layers.Conv2D(filters=self.num_filters,kernel_size=3,strides=1,padding='valid',use_bias=False,kernel_initializer=self.initializer)
        self.normalizer = batch_norm()
        
    def call(self, x):
        out = self.conv2D(x)
        out = self.normalizer(out)
        out = tf.keras.layers.LeakyReLU(alpha=0.2)(out)

        return out
