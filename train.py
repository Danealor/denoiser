#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

import logging
import os

import hydra

from denoiser.executor import start_ddp_workers

logger = logging.getLogger(__name__)


def run(args):
    import tensorflow as tf
    from denoiser.data import generate_dataset
    from denoiser.demucs import Demucs
    from denoiser.solver import Solver
    from denoiser.utils import get_device
    
    # Initialize seed
    tf.random.set_seed(args.seed)

    # Choose distribution strategy
    if args.ddp:
        if args.ddp_strategy == 'tpu':
            strategy = tf.distribute.TPUStrategy()
        else:
            strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy(get_device().name)

    # Build and compile
    with strategy.scope():
        model = Demucs(**args.demucs)
        model.build()
        solver = Solver(model, args)
        solver.compile_from_args()

    if args.show:
        logger.info(model)
        mb = sum(v.shape.num_elements() for v in model.variables) * 4 / 2**20
        logger.info('Size: %.1f MB', mb)
        if hasattr(model, 'valid_length'):
            field = model.valid_length(1)
            logger.info('Field: %.1f ms', field / args.sample_rate * 1000)
        return

    length = int(args.segment * args.sample_rate)
    stride = int(args.stride * args.sample_rate)
    # Demucs requires a specific sequence length to avoid 0 padding during training
    if hasattr(model, 'valid_length'):
        length = model.valid_length(length)
        
    # Building datasets
    kwargs = {"length": length,
              "stride": stride,
              "batch_size": args.batch_size,
              "element_type": args.element_type,
              "matching": args.dset.matching}

    tr_dataset = generate_dataset(args.dset.train, **kwargs)
    if args.dset.valid:
        cv_dataset = generate_dataset(args.dset.valid, **kwargs)
    else:
        cv_dataset = None
    if args.dset.test:
        tt_dataset = generate_dataset(args.dset.test, **kwargs)
    else:
        tt_dataset = None

    data = {"tr_loader": tr_dataset, "cv_loader": cv_dataset, "tt_loader": tt_dataset}

    solver.fit_from_args(data)


def _main(args):
    global __file__
    # Updating paths in config
    for key, value in args.dset.items():
        if isinstance(value, str) and key not in ["matching"]:
            args.dset[key] = hydra.utils.to_absolute_path(value)
    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("denoise").setLevel(logging.DEBUG)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    run(args)


@hydra.main(config_path="conf/config.yaml")
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()
