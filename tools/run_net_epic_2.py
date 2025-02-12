#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import sys
sys.path = [p for p in sys.path if "Ego-Exo-2" not in p]
sys.path.append(os.getcwd())
print(sys.path)
"""Wrapper to train and test a video classification model."""
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from demo_net import demo
from epic.test_net import test as test_epic
from epic.my_train_net_lemma_tpv_2 import train as train_epic


def get_func(cfg):
    train_func = train_epic
    if cfg.TRAIN.DATASET in ["epickitchen", "epickitchen_lemma_txt_2"]:
        train_func = train_epic

    test_func = test_epic
    if cfg.TEST.DATASET in ["epickitchen", "epickitchenhandobj"]:
        test_func = test_epic

    return train_func, test_func

def main():
    """
    Main function to spawn the train and test process.
    """

    args = parse_args()
    cfg = load_config(args)

    train_func, test_func = get_func(cfg)

    is_first_job = True

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train_func, args=args, is_first_job=is_first_job)

    is_first_job = False

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test_func, args=args, is_first_job=is_first_job)

    if cfg.DEMO.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=demo)


if __name__ == "__main__":
    main()
