# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Original copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import tensorflow as tf
from tensorflow.keras import layers, losses


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = tf.signal.stft(x, win_length, hop_size, fft_size, window)

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return tf.clip_by_value(tf.abs(x_stft), 1e-7, tf.dtypes.float32.max)


class SpectralConvergenceLoss(layers.Layer):
    """Spectral convergence loss layer."""

    def __init__(self, **kwargs):
        """Initilize spectral convergence loss layer."""
        super().__init__(**kwargs)

    def call(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return tf.norm(y_mag - x_mag, ord="fro") / tf.norm(y_mag, ord="fro")


class LogSTFTMagnitudeLoss(layers.Layer):
    """Log STFT magnitude loss module."""

    def __init__(self, **kwargs):
        """Initilize los STFT magnitude loss module."""
        super().__init__(**kwargs)

    def call(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return losses.MAE(tf.log(y_mag), tf.log(x_mag))


class STFTLoss(layers.Layer):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super().__init__(**kwargs)
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = window
        
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        if hasattr(tf.signal, window):
            self.window_fn = getattr(tf.signal, window)
        else:
            raise ValueError(f"Invalid window function '{window}'.")

    def get_config(self):
        config = super().get_config()
        config.update({"fft_size": self.fft_size})
        config.update({"shift_size": self.shift_size})
        config.update({"win_length": self.win_length})
        config.update({"window": self.window})
        return config

    def call(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window_fn)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window_fn)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(layers.Layer):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window", **kwargs):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super().__init__(**kwargs)
        self.fft_sizes = fft_sizes
        self.hop_size = hop_sizes
        self.win_lengths = win_lengths
        self.window = window
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

        self.stft_losses = [STFTLoss(fs, ss, wl, window) for fs, ss, wl 
                            in zip(fft_sizes, hop_sizes, win_lengths)]

    def get_config(self):
        config = super().get_config()
        config.update({"fft_sizes": self.fft_sizes})
        config.update({"hop_sizes": self.hop_sizes})
        config.update({"win_lengths": self.win_lengths})
        config.update({"window": self.window})
        return config

    def call(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss


class TotalMultiResolutionSTFTLoss(losses.Loss):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window", factor_sc=0.1, factor_mag=0.1, **kwargs):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        """
        super().__init__(**kwargs)
        self.fft_sizes = fft_sizes
        self.hop_size = hop_sizes
        self.win_lengths = win_lengths
        self.window = window
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

        self.mrstft_loss = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_lengths, window)

    def get_config(self):
        config = super().get_config()
        config.update({"fft_sizes": self.fft_sizes})
        config.update({"hop_sizes": self.hop_sizes})
        config.update({"win_lengths": self.win_lengths})
        config.update({"window": self.window})
        config.update({"factor_sc": self.factor_sc})
        config.update({"factor_mag": self.factor_mag})
        return config

    def call(self, y_true, y_pred):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        # Combine to mono if not already and squeeze out channel dim
        y_true, y_pred = map(lambda y: tf.reduce_mean(y, axis=-1, keepdims=False), y_true, y_pred)

        # Calculate sc and mag loss
        sc_loss, mag_loss = self.mrstft_loss(y_pred, y_true)

        # Calculate weighted total loss
        return self.factor_sc*sc_loss + self.factor_mag*mag_loss