# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

from collections import namedtuple
from functools import reduce
import json
from pathlib import Path
import os
import sys

from tensorflow import get_static_value
import tensorflow_io as tfio


Info = namedtuple("Info", ["length", "sample_rate", "channels"])


def get_info(path):
    data = tfio.audio.AudioIOTensor(path)
    return Info(get_static_value(data.shape[0]), 
                get_static_value(data.rate), 
                get_static_value(data.shape[1]))


def find_audio_files(path, exts=[".wav"], progress=True):
    audio_files = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))
    meta = []
    for idx, file in enumerate(audio_files):
        info = get_info(file)
        meta.append((file, int(info.length)))
        if progress:
            print("Loading audio metadata..." + format((1 + idx) / len(audio_files), " 3.1%"), end='\r')
    meta.sort()
    if progress:
        print("Loading audio metadata... Done! ")
    return meta


if __name__ == "__main__":
    dash = next(i for i,j in enumerate(sys.argv) if j=='-')
    dirs = sys.argv[1:dash] if dash else sys.argv[1:]
    meta = [item for dir in dirs for item in find_audio_files(dir)]
    if dash:
        with open(sys.argv[dash+1], 'w') as f:
            json.dump(meta, f, indent=4)
    else:
        json.dump(meta, sys.stdout, indent=4)
