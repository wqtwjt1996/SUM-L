#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""
from fvcore.common.config import CfgNode

def add_custom_config(_C):
    # Add your own customized configs.

    _C.TASK = ""

    # if using epic-55 or epic-100
    _C.DATA.EPIC_100 = False

    # multi-class
    _C.MODEL.MULTI_CLASS = False
    _C.MODEL.NUM_CLASSES_LIST = []
    
    # Train/val split files
    _C.TRAIN.TRAIN_DATA_LIST = "train.csv"
    _C.TRAIN.VAL_DATA_LIST = "val.csv"


    # Test split files
    _C.TEST.DATA_LIST = "test.csv"
    #  Test saving path
    _C.TEST.SAVE_PREDICT_PATH = "predicts.pkl"

    _C.TEST.NO_ACT = False

    _C.TEST.HEAD_TARGETS = True

    _C.TEST.LOAD_BEST = False

    # Model
    _C.MODEL.FREEZE_STAGES = 0

    _C.MODEL.FREEZE_BN = False

    # Loss
    _C.LOSS = CfgNode()
    # cross-entropy loss weight
    _C.LOSS.CE_LOSS_WEIGHT = 1.0
