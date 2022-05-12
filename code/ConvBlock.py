import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, ReLU, LeakyReLU
from batch_norm import batch_norm
from tensorflow.keras.layers import Conv2D, ReLU, LeakyReLU
from tensorflow.keras.layers import Add, Lambda, ZeroPadding2D

class ConvBlock(Layer):
    def __init__(self, num_filters):
        super(ConvBlock, self).__init__()
        self.num_filters = num_filters
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.conv2D = Conv2D(filters=self.num_filters,
                             kernel_size=3,
                             strides=1,
                             padding='valid',
                             use_bias=False,
                             kernel_initializer=self.initializer)
        self.instance_norm = batch_norm()
        
    def call(self, x):
        x = self.conv2D(x)
        x = self.instance_norm(x)
        x = LeakyReLU(alpha=0.2)(x)

        return x
