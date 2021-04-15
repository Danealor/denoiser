# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.experimental.numpy import sinc

def log1p10(x):
    return tf.math.log1p(x) / np.log(10)

def exp10m1(x):
    return tf.math.expm1(x * np.log(10))


def hz_to_mel(f):
    return 2595 * log1p10(f / 700)

def mel_to_hz(m):
    return 700 * exp10m1(m / 2595)


@tf.function
def mel_frequencies(n_mels, fmin, fmax):
    low = hz_to_mel(fmin)
    high = hz_to_mel(fmax)
    mels = tf.linspace(low, high, n_mels)
    return mel_to_hz(mels)


class LowPassFilters(layers.Layer):
    """
    Bank of low pass filters.

    Args:
        cutoffs (list[float]): list of cutoff frequencies, in [0, 1] expressed as `f/f_s` where
            f_s is the samplerate.
        width (int): width of the filters (i.e. kernel_size=2 * width + 1).
            Default to `2 / min(cutoffs)`. Longer filters will have better attenuation
            but more side effects.
    Shape:
        - Input: `(*, T)`
        - Output: `(F, *, T` with `F` the len of `cutoffs`.
    """

    def __init__(self, cutoffs, width: int = None, **kwargs):
        super().__init__(**kwargs)
        self.cutoffs = cutoffs
        if width is None:
            width = int(2 / min(cutoffs))
        self.width = width
        window = tf.signal.hamming_window(2 * width + 1, periodic=False)
        t = tf.range(-width, width + 1, dtype=tf.float32)
        sincs = sinc(2 * cutoffs[:,tf.newaxis] * t[tf.newaxis,:])
        filters = 2 * cutoffs[:,tf.newaxis] * sincs * window[tf.newaxis,:]
        self.filters = tf.transpose(filters)[:,tf.newaxis,:]

    def get_config(self):
        config = super().get_config()
        config.update({"cutoffs": self.cutoffs})
        config.update({"width": self.width})
        return config

    def call(self, input):
        out = tf.nn.conv1d(input, self.filters, stride=1, padding='SAME')
        return tf.transpose(out, (2,0,1))[...,tf.newaxis]

    def __repr__(self):
        return "LossPassFilters(width={},cutoffs={})".format(self.width, self.cutoffs)
