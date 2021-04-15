# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez and adiyoss

import json
import logging
import os
import re

import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.data import Dataset

logger = logging.getLogger(__name__)


def match_dns(noisy, clean):
    """match_dns.
    Match noisy and clean DNS dataset filenames.

    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    """
    logger.debug("Matching noisy and clean for dns dataset")
    noisydict = {}
    extra_noisy = []
    for path, size in noisy:
        match = re.search(r'fileid_(\d+)\.wav$', path)
        if match is None:
            # maybe we are mixing some other dataset in
            extra_noisy.append((path, size))
        else:
            noisydict[match.group(1)] = (path, size)
    noisy[:] = []
    extra_clean = []
    copied = list(clean)
    clean[:] = []
    for path, size in copied:
        match = re.search(r'fileid_(\d+)\.wav$', path)
        if match is None:
            extra_clean.append((path, size))
        else:
            noisy.append(noisydict[match.group(1)])
            clean.append((path, size))
    extra_noisy.sort()
    extra_clean.sort()
    clean += extra_clean
    noisy += extra_noisy


def match_files(noisy, clean, matching="sort"):
    """match_files.
    Sort files to match noisy and clean filenames.
    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    :param matching: the matching function, at this point only sort is supported
    """
    if matching == "dns":
        # dns dataset filenames don't match when sorted, we have to manually match them
        match_dns(noisy, clean)
    elif matching == "sort":
        noisy.sort()
        clean.sort()
    else:
        raise ValueError(f"Invalid value for matching {matching}")


def load_files(json_dir, matching="sort"):
    noisy_json = os.path.join(json_dir, 'noisy.json')
    clean_json = os.path.join(json_dir, 'clean.json')
    with open(noisy_json, 'r') as f:
        noisy = json.load(f)
    with open(clean_json, 'r') as f:
        clean = json.load(f)

    match_files(noisy, clean, matching)
    sources = (noisy, clean)
    source_paths = tuple([path for path, size in source] for source in sources)
    files_ds = Dataset.from_tensor_slices(source_paths)
    return files_ds


def apply(func):
    def apply_all(*sources):
        if len(sources)==1: return func(sources[0])
        res = tuple(func(*source) if isinstance(source, tuple) else func(source) for source in sources)
        if all(isinstance(r, Dataset) for r in res):
            res = Dataset.zip(res)
        return res
    return apply_all


def read_audio(file_path):
    audio = tfio.audio.AudioIOTensor(file_path,dtype='int16').to_tensor()
    return tf.cast(audio, tf.float32) / (2**16/2)


def generate_audio(files_ds, shuffle=False):
    if shuffle:
        # Shuffle all file paths, reshuffling enabled
        # This prevents training on local pockets of the same speaker
        files_ds = files_ds.shuffle(len(files_ds))
    return files_ds.map(apply(read_audio))


def extract_examples(audio, length, stride):
    indices = Dataset.range(0,tf.cast(tf.shape(audio)[0]-length+1,dtype='int64'),stride)
    return indices.map(lambda i: audio[i:i+length,...])


def extract_examples_static(audio, length, stride):
    "Same as `extract_examples`, except returns `tf.Tensor` instead of `Dataset`"
    indices = (tf.range(length) + tf.range(0,tf.maximum(0,tf.shape(audio)[0]-length+1),stride)[:,tf.newaxis])
    return tf.gather(audio, indices)


def generate_examples(audio_ds, length, stride, batch=None):
    func = lambda audio: extract_examples(audio, length, stride)
    examples_ds = audio_ds.flat_map(apply(func))
    if batch:
        examples_ds = examples_ds.batch(batch)
    return examples_ds


def generate_samples(audio_ds, batch=None):
    func = lambda t: (t,tf.shape(t)[0])
    audio_length_ds = audio_ds.map(apply(func))
    if batch:
        audio_length_ds = audio_length_ds.padded_batch(batch)
    return audio_length_ds


def generate_both(audio_ds, example_length, example_stride, batch=None):
    def func_examples(audio):
        return extract_examples_static(audio, example_length, example_stride)
    def func_sample(t):
        return (t,tf.shape(t)[0])
    def flatten(t):
        return tf.reshape(t, tf.concat([[tf.shape(t)[0] * tf.shape(t)[1]], tf.shape(t)[2:]], axis=0))

    combined_ds = audio_ds.map(apply(lambda audio: (func_examples(audio), func_sample(audio))))
    if batch:
        combined_ds = combined_ds.padded_batch(batch) \
                                 .map(apply(lambda e,s: (flatten(e), s)))
    return combined_ds

def generate_dataset(json_dir, length, stride, batch_size, matching='sort', element_type='both', named=True):
    files_ds = load_files(json_dir, matching)
    audio_ds = generate_audio(files_ds, shuffle=True)
    if element_type == 'examples':
        return generate_examples(audio_ds, length, stride, batch_size)\
                 .map(apply(lambda e: dict(examples=e)))
    elif element_type == 'samples':
        return generate_samples(audio_ds, batch_size)\
                 .map(apply(lambda s: dict(samples=s)))
    elif element_type == 'both':
        return generate_both(audio_ds, length, stride, batch_size)\
                 .map(apply(lambda e,s: dict(examples=e, samples=s)))
    else:
        raise ValueError(f"element_type must be one of 'examples', 'samples', or 'both';" +
                         f" instead found '{element_type}'")