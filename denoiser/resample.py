import math

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.experimental.numpy import sinc

def kernel(zeros=56):
    win = tf.signal.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = tf.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    return tf.reshape(sinc(t) * winodd, (-1, 1, 1))


class Upsample2(layers.Layer):
    def __init__(self, zeros=56, **kwargs):
        super(Upsample2, self).__init__(**kwargs)
        self.zeros = zeros
        self.kernel = kernel(zeros)

    def get_config(self):
        config = super(Upsample2, self).get_config()
        config.update({"zeros": self.zeros})
        return config

    def call(self, inputs):
        out = tf.nn.conv1d(inputs, self.kernel, stride=1, padding='SAME')
        y = tf.stack([inputs, out], axis=-2)
        return tf.reshape(y, tf.stack([tf.shape(inputs)[0], -1, tf.shape(inputs)[2]]))


class Downsample2(layers.Layer):
    def __init__(self, zeros=56, **kwargs):
        super(Downsample2, self).__init__(**kwargs)
        self.zeros = zeros
        self.kernel = kernel(zeros)

    def get_config(self):
        config = super(Downsample2, self).get_config()
        config.update({"zeros": self.zeros})
        return config

    def call(self, inputs):
        xeven = inputs[:, ::2, :]
        xodd = inputs[:, 1::2, :]
        out = tf.nn.conv1d(xodd, self.kernel, stride=1, padding='SAME')
        return (xeven + out) * 0.5