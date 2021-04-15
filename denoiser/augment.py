# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import tensorflow as tf
from tensorflow.keras import layers

from . import dsp


class Remix(layers.Layer):
    """Remix.
    Mixes different noises with clean speech within a given batch
    """

    def call(self, inputs, grouped=False):
        if grouped:
            # Shuffle the respective noise tensors of each group together
            length = tf.shape(inputs[0])[1]
            indices = tf.range(0, length)
            indices = tf.random.shuffle(indices)
            def remix(sources):
                noise, clean = sources
                return tf.stack([tf.gather(noise, indices), clean])
            return tuple(remix(group) for group in inputs)
        else:
            noise, clean = inputs
            return tf.stack([tf.random.shuffle(noise), clean])

class RevEcho(layers.Layer):
    """
    Hacky Reverb but runs on GPU without slowing down training.
    This reverb adds a succession of attenuated echos of the input
    signal to itself. Intuitively, the delay of the first echo will happen
    after roughly 2x the radius of the room and is controlled by `first_delay`.
    Then RevEcho keeps adding echos with the same delay and further attenuation
    until the amplitude ratio between the last and first echo is 1e-3.
    The attenuation factor and the number of echos to adds is controlled
    by RT60 (measured in seconds). RT60 is the average time to get to -60dB
    (remember volume is measured over the squared amplitude so this matches
    the 1e-3 ratio).

    At each call to RevEcho, `first_delay`, `initial` and `RT60` are
    sampled from their range. Then, to prevent this reverb from being too regular,
    the delay time is resampled uniformly within `first_delay +- 10%`,
    as controlled by the `jitter` parameter. Finally, for a denser reverb,
    multiple trains of echos are added with different jitter noises.

    Args:
        - initial: amplitude of the first echo as a fraction
            of the input signal. For each sample, actually sampled from
            `[0, initial]`. Larger values means louder reverb. Physically,
            this would depend on the absorption of the room walls.
        - rt60: range of values to sample the RT60 in seconds, i.e.
            after RT60 seconds, the echo amplitude is 1e-3 of the first echo.
            The default values follow the recommendations of
            https://arxiv.org/ftp/arxiv/papers/2001/2001.08662.pdf, Section 2.4.
            Physically this would also be related to the absorption of the
            room walls and there is likely a relation between `RT60` and
            `initial`, which we ignore here.
        - first_delay: range of values to sample the first echo delay in seconds.
            The default values are equivalent to sampling a room of 3 to 10 meters.
        - repeat: how many train of echos with differents jitters to add.
            Higher values means a denser reverb.
        - jitter: jitter used to make each repetition of the reverb echo train
            slightly different. For instance a jitter of 0.1 means
            the delay between two echos will be in the range `first_delay +- 10%`,
            with the jittering noise being resampled after each single echo.
        - keep_clean: fraction of the reverb of the clean speech to add back
            to the ground truth. 0 = dereverberation, 1 = no dereverberation.
        - sample_rate: sample rate of the input signals.
    """

    def __init__(self, proba=0.5, initial=0.3, rt60=(0.3, 1.3), first_delay=(0.01, 0.03),
                 repeat=3, jitter=0.1, keep_clean=0.1, sample_rate=16000):
        super().__init__()
        self.proba = proba
        self.initial = initial
        self.rt60 = rt60
        self.first_delay = first_delay
        self.repeat = repeat
        self.jitter = jitter
        self.keep_clean = keep_clean
        self.sample_rate = sample_rate

    def get_config(self):
        config = super().get_config()
        config.update({"proba": self.proba})
        config.update({"initial": self.initial})
        config.update({"rt60": self.rt60})
        config.update({"first_delay": self.first_delay})
        config.update({"repeat": self.repeat})
        config.update({"jitter": self.jitter})
        config.update({"keep_clean": self.keep_clean})
        config.update({"sample_rate": self.sample_rate})
        return config

    def _reverb(self, source, initial, first_delay, rt60):
        """
        Return the reverb for a single source.
        """
        length = tf.shape(source)[-2]
        reverb = tf.zeros_like(source)
        for _ in range(self.repeat):
            frac = 1  # what fraction of the first echo amplitude is still here
            echo = initial * source
            while frac > 1e-3:
                # First jitter noise for the delay
                jitter = 1 + self.jitter * tf.random.uniform((), -1, 1)
                delay = min(
                    1 + int(jitter * first_delay * self.sample_rate),
                    length)
                # Delay the echo in time by padding with zero on the left
                echo = tf.pad(echo[..., :-delay, :], [[0,0],[delay,0],[0,0]])
                reverb += echo

                # Second jitter noise for the attenuation
                jitter = 1 + self.jitter * tf.random.uniform((), -1, 1)
                # we want, with `d` the attenuation, d**(rt60 / first_ms) = 1e-3
                # i.e. log10(d) = -3 * first_ms / rt60, so that
                attenuation = 10**(-3 * jitter * first_delay / rt60)
                echo *= attenuation
                frac *= attenuation
        return reverb

    def call(self, wav):
        if tf.random.uniform(()) >= self.proba:
            return wav
        noise, clean = wav
        # Sample characteristics for the reverb
        initial = tf.random.uniform(()) * self.initial
        first_delay = tf.random.uniform((), *self.first_delay)
        rt60 = tf.random.uniform((), *self.rt60)

        reverb_noise = self._reverb(noise, initial, first_delay, rt60)
        # Reverb for the noise is always added back to the noise
        noise += reverb_noise
        reverb_clean = self._reverb(clean, initial, first_delay, rt60)
        # Split clean reverb among the clean speech and noise
        clean += self.keep_clean * reverb_clean
        noise += (1 - self.keep_clean) * reverb_clean

        return tf.stack([noise, clean])


class BandMask(layers.Layer):
    """BandMask.
    Maskes bands of frequencies. Similar to Park, Daniel S., et al.
    "Specaugment: A simple data augmentation method for automatic speech recognition."
    (https://arxiv.org/pdf/1904.08779.pdf) but over the waveform.
    """

    def __init__(self, maxwidth=0.2, bands=120, sample_rate=16_000, **kwargs):
        """__init__.

        :param maxwidth: the maximum width to remove
        :param bands: number of bands
        :param sample_rate: signal sample rate
        """
        super().__init__(**kwargs)
        self.maxwidth = maxwidth
        self.bands = bands
        self.sample_rate = sample_rate

        self.bandwidth = int(abs(maxwidth) * bands)
        self.mels = dsp.mel_frequencies(bands, 40, sample_rate/2) / sample_rate

    def get_config(self):
        config = super().get_config()
        config.update({"maxwidth": self.maxwidth})
        config.update({"bands": self.bands})
        config.update({"sample_rate": self.sample_rate})
        return config

    def call(self, wav):
        low = tf.random.uniform((), maxval=bands, dtype=tf.dtypes.int32)
        high = tf.random.uniform((), minval=low, maxval=min(self.bands, low + self.bandwidth), dtype=tf.dtypes.int32)
        filters = dsp.LowPassFilters(tf.gather(self.mels, [low, high]))
        low, midlow = filters(wav)
        # band pass filtering
        out = wav - midlow + low
        return out


class Shift(layers.Layer):
    """Shift."""

    def __init__(self, shift=8192, same=False, **kwargs):
        """__init__.

        :param shift: randomly shifts the signals up to a given factor
        :param same: shifts both clean and noisy files by the same factor
        """
        super().__init__(**kwargs)
        self.shift = shift
        self.same = same

    def get_config(self):
        config = super().get_config()
        config.update({"shift": self.shift})
        config.update({"same": self.same})
        return config

    def call(self, wav):
        sources, batch, length, channels = tf.shape(wav)
        length = length - self.shift
        if self.shift > 0:
            offsets = tf.random.uniform(
                (1 if same else sources, batch),
                maxval=shift,
                dtype=tf.dtypes.int32)
            offsets = tf.tile(offsets, [sources if same else 1,1])
            indices = tf.range(length) + offsets[...,tf.newaxis]
            wav = tf.gather(audio, indices, axis=2, batch_dims=2)
        else:
            offsets = tf.zeros((sources, batch))
        return wav, offsets
