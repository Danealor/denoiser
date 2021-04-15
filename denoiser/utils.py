# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import functools
import logging
from contextlib import contextmanager
import inspect
import time
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def capture_init(init):
    """capture_init.

    Decorate `__init__` with this, and you can then
    recover the *args and **kwargs passed to it in `self._init_args_kwargs`
    """
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self._init_args_kwargs = (args, kwargs)

    return __init__


def deserialize_model(package, strict=False):
    """deserialize_model.

    """
    klass = package['class']
    if strict:
        model = klass(*package['args'], **package['kwargs'])
    else:
        sig = inspect.signature(klass)
        kw = package['kwargs']
        for key in list(kw):
            if key not in sig.parameters:
                logger.warning("Dropping inexistant parameter %s", key)
                del kw[key]
        model = klass(*package['args'], **kw)
    model.load_state_dict(package['state'])
    return model


def copy_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}


def serialize_model(model):
    args, kwargs = model._init_args_kwargs
    state = copy_state(model.state_dict())
    return {"class": model.__class__, "args": args, "kwargs": kwargs, "state": state}


@contextmanager
def swap_state(model, state):
    """
    Context manager that swaps the state of a model, e.g:

        # model is in old state
        with swap_state(model, new_state):
            # model in new state
        # model back to old state
    """
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state)
    try:
        yield
    finally:
        model.load_state_dict(old_state)


def pull_metric(history, name):
    out = []
    for metrics in history:
        if name in metrics:
            out.append(metrics[name])
    return out


class LogProgress:
    """
    Sort of like tqdm but using log lines and not as real time.
    Args:
        - logger: logger obtained from `logging.getLogger`,
        - iterable: iterable object to wrap
        - updates (int): number of lines that will be printed, e.g.
            if `updates=5`, log every 1/5th of the total length.
        - total (int): length of the iterable, in case it does not support
            `len`.
        - name (str): prefix to use in the log.
        - level: logging level (like `logging.INFO`).
    """
    def __init__(self,
                 logger,
                 iterable,
                 updates=5,
                 total=None,
                 name="LogProgress",
                 level=logging.INFO):
        self.iterable = iterable
        self.total = total or len(iterable)
        self.updates = updates
        self.name = name
        self.logger = logger
        self.level = level

    def update(self, **infos):
        self._infos = infos

    def __iter__(self):
        self._iterator = iter(self.iterable)
        self._index = -1
        self._infos = {}
        self._begin = time.time()
        return self

    def __next__(self):
        self._index += 1
        try:
            value = next(self._iterator)
        except StopIteration:
            raise
        else:
            return value
        finally:
            log_every = max(1, self.total // self.updates)
            # logging is delayed by 1 it, in order to have the metrics from update
            if self._index >= 1 and self._index % log_every == 0:
                self._log()

    def _log(self):
        self._speed = (1 + self._index) / (time.time() - self._begin)
        infos = " | ".join(f"{k.capitalize()} {v}" for k, v in self._infos.items())
        if self._speed < 1e-4:
            speed = "oo sec/it"
        elif self._speed < 0.1:
            speed = f"{1/self._speed:.1f} sec/it"
        else:
            speed = f"{self._speed:.1f} it/sec"
        out = f"{self.name} | {self._index}/{self.total} | {speed}"
        if infos:
            out += " | " + infos
        self.logger.log(self.level, out)


def colorize(text, color):
    """
    Display text with some ANSI color in the terminal.
    """
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def bold(text):
    """
    Display text in bold in the terminal.
    """
    return colorize(text, "1")


def get_device():
    gpus = tf.config.list_logical_devices('GPU')
    if gpus:
        return gpus[0]
    return tf.config.list_logical_devices()[0]


def map_cond(func, *elements):
    return tuple(func(x) if x is not None else None for x in elements)


@tf.function
def tensor_to_tuple(tensor, num_elements):
    return tuple(tensor[i] for i in range(num_elements))


class LengthCalc:
    def __init__(self, depth, kernel_size, stride):
        self.first = LengthCalc.calc_in(1, depth, kernel_size, stride)
        self.diff = stride**depth

    @staticmethod
    def calc_out(input, depth, kernel_size, stride):
        x = input
        n = depth
        k = kernel_size
        s = stride

        return (x + s*(1-s**n)//(1-s)*(1-k//s))/(s**n)
    
    @staticmethod
    def calc_in(output, depth, kernel_size, stride):
        x = output
        n = depth
        k = kernel_size
        s = stride

        return s**n*x - s*(1-s**n)//(1-s)*(1-k//s)

    def first_lower(self, input):
        output = (input - self.first) // self.diff
        if isinstance(output, tf.Tensor):
            return tf.where(output < 0, tf.zeros_like(output), output*self.diff + self.first)
        elif isinstance(output, np.ndarray):
            return np.where(output < 0, np.zeros_like(output), output*self.diff + self.first)
        else:
            if output < 0: return 0
            return output*self.diff + self.first

    def first_greater(self, input):
        output = (input - self.first - 1) // self.diff + 1
        if isinstance(output, tf.Tensor):
            return tf.where(output < 0, tf.zeros_like(output), output)*self.diff + self.first
        elif isinstance(output, np.ndarray):
            return np.where(output < 0, np.zeros_like(output), output)*self.diff + self.first
        else:
            return max(0,output)*self.diff + self.first