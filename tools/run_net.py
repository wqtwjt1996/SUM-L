#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from epic.test_net import test as test_epic


def get_func(cfg):
    return None, test_epic

def main():
    """
    Main function to spawn the train and test process.
    """

    args = parse_args()
    cfg = load_config(args)

    train_func, test_func = get_func(cfg)

    is_first_job = True

    # Perform training.
    # Qitong's config on Dec. 17th, 2021.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train_func, args=args, is_first_job=is_first_job)
        is_first_job = False

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test_func, args=args, is_first_job=is_first_job)


if __name__ == "__main__":
    main()
