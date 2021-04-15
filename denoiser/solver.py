# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

import json
import logging
from pathlib import Path
import os
import time

import tensorflow as tf

from . import augment
from .custom import SumLoss
from .metrics import PESQ, STOI
from .stft_loss import TotalMultiResolutionSTFTLoss
from .utils import map_cond, tensor_to_tuple

logger = logging.getLogger(__name__)


class Solver(tf.keras.Model):
    def __init__(self, model, args):
        super().__init__()

        self.model = model
        self.args = args

        # data augment
        if args.remix:
            self.remix = augment.Remix()
        if args.bandmask:
            self.bandmask = augment.BandMask(args.bandmask, sample_rate=args.sample_rate)
        if args.shift:
            self.shift = augment.Shift(args.shift, args.shift_same)
        if args.revecho:
            self.revecho = augment.RevEcho(args.revecho)

    def _reset(self):
        """_reset."""
        load_from = None
        load_best = False
        keep_history = True

        args = self.args

        # Reset
        checkpoint_file = Path(args.checkpoint_file)
        continue_from = Path(args.continue_from)

        if args.checkpoint and checkpoint_file.exists() and not args.restart:
            load_from = args.checkpoint_file
        elif args.continue_from:
            load_from = args.continue_from
            load_best = args.continue_best
            keep_history = False

        if load_from:
            logger.info(f'Loading checkpoint model: {load_from}')
            self.load_weights(load_from)

    def compile_from_args(self):
        args = self.args

        def make_loss():
            # Configure model loss
            if args.loss == 'l1':
                loss = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)
            elif args.loss == 'l2':
                loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
            elif args.loss == 'huber':
                loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
            else:
                raise ValueError(f"Invalid loss {args.loss}")

            # MultiResolution STFT loss
            if args.stft_loss:
                mrstftloss = TotalMultiResolutionSTFTLoss(factor_sc=args.stft_sc_factor,
                                                          factor_mag=args.stft_mag_factor,
                                                          reduction=losses.Reduction.SUM)
                loss = SumLoss(loss, mrstftloss, reduction=losses.Reduction.SUM)

            return loss

        # Duplicate loss for double elements
        loss = [make_loss(), make_loss()]

        # Configure model optimizer
        if args.optim == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.9, beta_2=args.beta2)
        else:
            raise ValueError(f"Invalid optimizer {args.optimizer}")

        # Configure metrics
        metrics = []
        if args.pesq:
            metrics.append(PESQ(sample_rate=args.sample_rate))
        metrics.append(STOI(sample_rate=args.sample_rate))

        self.compile(optimizer, loss, metrics)

    def fit_from_args(self, data):
        tr_loader = data['tr_loader']
        cv_loader = data['cv_loader']
        tt_loader = data['tt_loader']

        args = self.args

        # Checkpoints
        self._reset()

        # Callbacks
        callbacks = []
        if args.checkpoint:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(Path(args.checkpoint_file)))
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(Path(args.best_file), save_best_only=True))

        self.fit(x=tr_loader, 
                 validation_data=tt_loader, 
                 epochs=args.epochs, 
                 validation_freq=args.eval_every,
                 callbacks=callbacks)

    def _extract_and_pack(self, sources):
        examples = []
        samples = []
        sample_lengths = []
        for source in sources:
            e, s = source.get('examples'), source.get('samples')
            if e is not None: 
                examples.append(e)
            if s is not None: 
                sample, length = s
                samples.append(sample)
                sample_lengths.append(length)
        
        return map_cond(tf.stack, examples, samples, sample_lengths)

    def _preprocess(self, examples, samples, sample_lengths):
        def split(sources):
            noisy, clean = tensor_to_tuple(sources, 2)
            noise = noisy - clean
            return tf.stack([noise, clean])

        def combine(sources):
            noise, clean = tensor_to_tuple(sources, 2)
            noisy = clean + noise
            return tf.stack([noisy, clean])

        # Split noisy into noise and clean
        examples, samples = map_cond(split, examples, samples)

        if self.args.remix:
            if examples:
                examples = self.remix(examples)
            if samples:
                samples, sample_lengths = self.remix((samples, sample_lengths), grouped=True)

        if self.args.bandmask:
            examples, samples = map_cond(self.bandmask, examples, samples)

        if self.args.shift:
            if examples:
                examples, _ = self.shift(examples)
            if samples:
                samples, offsets = self.shift(samples)
                sample_lengths -= offsets

        if self.args.revecho:
            examples, samples = map_cond(self.revecho, examples, samples)

        # Combine noise and clean back into noisy
        examples, samples = map_cond(combine, examples, samples)

        return examples, samples, sample_lengths

    def _postprocess(self, examples, samples, sample_lengths):
        if samples is not None:
            sources, batch, length, channels = tensor_to_tuple(tf.shape(samples), 4)

            # Truncate to shortest sample, in case of shuffling/shifting
            sample_lengths = tf.math.reduce_min(sample_lengths, axis=0)
            # Truncate further down to nearest valid length
            if hasattr(self.model, 'valid_length'):
                sample_lengths = self.model.valid_length(sample_lengths, greater=False)

            # Convert to Ragged Tensor, with ragged dimension on length axis (axis=2)
            sample_lengths = tf.tile(sample_lengths, (sources,))
            samples = tf.reshape(samples, (sources * batch, length, channels))
            samples = tf.RaggedTensor.from_tensor(samples, sample_lengths)
            samples = tf.RaggedTensor.from_uniform_row_length(samples, tf.cast(batch, tf.int64))

        return examples, samples

    def _execute(self, sources, training=False):
        noisy, clean = tensor_to_tuple(sources, 2)
        estimate = self.model(noisy, training=True)
        return tf.stack([clean, estimate]) # swap to the (y_true, y_pred) order

    @staticmethod
    def _unpack(*elements):
        y_true = []
        y_pred = []
        for elem in elements:
            if elem is not None:
                y_true.append(elem[0])
                y_pred.append(elem[1])
            else:
                y_true.append(None)
                y_pred.append(None)
        return y_true, y_pred

    def train_step(self, inputs):
        examples, samples, sample_lengths = self._extract_and_pack(inputs)
        examples, samples, sample_lengths = self._preprocess(examples, samples, sample_lengths)

        # Execute in model
        with tf.GradientTape():
            examples, samples = map_cond(lambda sources: 
                self._execute(sources, training=True), examples, samples)
            examples, samples = self._postprocess(examples, samples, sample_lengths)
            loss = self.compiled_loss(*self._unpack(examples, samples))
            total_loss = sum(loss for loss in (example_loss, sample_loss) if loss is not None)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # No metrics are computed in the train step except for loss.
        return {'loss': total_loss}

    def test_step(self, inputs):
        examples, samples, sample_lengths = self._extract_and_pack(inputs)
        examples, samples, sample_lengths = self._preprocess(examples, samples, sample_lengths)
        
        # Execute in model
        examples, samples = map_cond(lambda sources: self._execute(sources, training=False), examples, samples)
        examples, samples = self._postprocess(examples, samples, sample_lengths)
        example_loss, sample_loss = map_cond(lambda sources: self.compiled_loss(*sources), examples, samples)
        total_loss = sum(loss for loss in (example_loss, sample_loss) if loss is not None)

        # Update the metrics.
        map_cond(lambda sources: self.compiled_metrics.update_state(*sources), examples, samples)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {'loss': total_loss} + {m.name: m.result() for m in self.metrics}

    def predict_step(self, inputs):
        examples, samples, sample_lengths = self._extract_and_pack(inputs)
        examples, samples, sample_lengths = self._preprocess(examples, samples, sample_lengths)
        
        # Execute in model
        examples, samples = map_cond(self.model, examples, samples)
        examples, samples = self._postprocess(examples, samples, sample_lengths)

        return tuple(s for s in (examples, samples) if s)
