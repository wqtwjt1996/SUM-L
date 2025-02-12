"""

    Created on 12/3/21

    @author: Qitong Wang

    Description:

"""
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

import numpy as np
import pprint
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.utils.logging as logging
import slowfast.models.losses as losses

# from models.build import get_loss_func
import slowfast.models.optimizer as optim
from slowfast.models import build_model

from slowfast.datasets import loader

from slowfast.utils.meters import RecMeter_fpv, RecMeter_bih

logger = logging.get_logger(__name__)

def train_epoch_bi(train_loader, model, optimizer, train_meter, cur_epoch, cfg):
    """
    Qitong Wang on Oct. 31st, 2021.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    for cur_iter, (fpv_inputs, tpv_inputs, labels, _, meta) in enumerate(train_loader):

        # construct unpaired data
        fpv_inputs = [ele[:int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)] for ele in fpv_inputs]
        tpv_inputs = [ele[:int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)] for ele in tpv_inputs]

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        # Transfer the data to the current GPU device.
        if isinstance(fpv_inputs, (list,)) and isinstance(tpv_inputs, (list,)):
            for i in range(len(fpv_inputs)):
                fpv_inputs[i] = fpv_inputs[i].cuda(non_blocking=True)
            if not cfg.DATA.FPV_ONLY:
                for i in range(len(tpv_inputs)):
                    tpv_inputs[i] = tpv_inputs[i].cuda(non_blocking=True)
        else:
            fpv_inputs = fpv_inputs.cuda(non_blocking=True)
            if not cfg.DATA.FPV_ONLY:
                tpv_inputs = tpv_inputs.cuda(non_blocking=True)
        labels = labels.cuda()

        if cfg.DATA.FPV_ONLY:
            if cur_iter == 0:
                logger.info("LEMMA with FPV only...")
            all_preds, addi_loss = model(fpv_inputs, None, cur_epoch, labels, meta)
        elif cfg.DATA.TPV_ONLY:
            if cur_iter == 0:
                logger.info("LEMMA with TPV only...")
            all_preds, addi_loss = model(None, tpv_inputs, cur_epoch, labels, meta)
        else:
            if cur_iter == 0:
                logger.info("LEMMA with FPV and TPV...")
            all_preds, addi_loss = model(fpv_inputs, tpv_inputs, cur_epoch, labels, meta)

        loss = 0.0
        for k, v in addi_loss.items():
            misc.check_nan_losses(v)
            loss += v

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()


        if cfg.NUM_GPUS > 1:
            loss = du.all_reduce([loss])[0]
        loss = loss.item()

        train_meter.iter_toc()
        # Update and log stats.
        train_meter.update_stats_small(loss, lr,
            tpv_inputs[0].size(0) * cfg.NUM_GPUS
        )
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    # train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

def eval_epoch_bi(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Qitong on Nov. 2nd, 2021.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    # print(model.state_dict().keys())
    # import time
    # time.sleep(500)
    val_meter.iter_tic()
    for cur_iter, (fpv_inputs, tpv_inputs, labels, _, meta) in enumerate(val_loader):
        # Transfer the data to the current GPU device.
        if isinstance(fpv_inputs, (list,)):
            for i in range(len(fpv_inputs)):
                fpv_inputs[i] = fpv_inputs[i].cuda(non_blocking=True)
        else:
            fpv_inputs = fpv_inputs.cuda(non_blocking=True)
        if isinstance(tpv_inputs, (list,)):
            for i in range(len(tpv_inputs)):
                tpv_inputs[i] = tpv_inputs[i].cuda(non_blocking=True)
        else:
            tpv_inputs = tpv_inputs.cuda(non_blocking=True)
        labels = labels.cuda()
        preds_fpv, loss_con = model(fpv_inputs, None, -1, None, meta)

        (
            tp_f, tn_f, fp_f, fn_f,
            pred_labels_f, act_pred_labels_f, obj_pred_labels_f,
            gt_act_labels_f, gt_obj_labels_f
        ) = metrics.eval_pred(preds_fpv, labels, meta, cfg)

        if cfg.NUM_GPUS > 1:
            torch.distributed.all_reduce(tp_f)
            torch.distributed.all_reduce(fp_f)
            torch.distributed.all_reduce(fn_f)
            pred_labels_f = torch.stack([i for item in du.all_gather(pred_labels_f) for i in item])
            act_pred_labels_f = torch.stack([i for item in du.all_gather(act_pred_labels_f) for i in item])
            obj_pred_labels_f = torch.stack([i for item in du.all_gather(obj_pred_labels_f) for i in item])
            labels = torch.stack([i for item in du.all_gather(labels) for i in item])

        (
            pred_labels_f, act_pred_labels_f, obj_pred_labels_f,
            gt_act_labels_f, gt_obj_labels_f,
            labels
        ) = (
            pred_labels_f.cpu(), act_pred_labels_f.cpu(), obj_pred_labels_f.cpu(),
            gt_act_labels_f.cpu(), gt_obj_labels_f.cpu(),
            labels.cpu()
        )

        tp_f, fp_f, fn_f = \
            (tp_f.item(), fp_f.item(), fn_f.item())

        val_meter.iter_toc()
        val_meter.update_stats(
            tp_f, fp_f, fn_f, pred_labels_f, act_pred_labels_f, obj_pred_labels_f,
            gt_act_labels_f, gt_obj_labels_f,
            labels, None, None,
            fpv_inputs[0].size(0) * cfg.NUM_GPUS
        )
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()

def train_bi(cfg):
    """
    Qitong Wang on Oct. 31st, 2021.
    """
    # Set random seed from settings.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup log_utils format.
    logging.setup_logging()

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model_utils and print model_utils statistics.
    model = build_model(cfg)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info('Load from last checkpoint.')
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        checkpoint_epoch = cu.load_checkpoint(
            last_checkpoint, model, cfg.NUM_GPUS > 1, None
        )
        start_epoch = 0
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != '':
        logger.info('Load from given checkpoint file {}.'.format(cfg.TRAIN.CHECKPOINT_FILE_PATH))
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == 'caffe2',
        )
        start_epoch = 0
    else:
        start_epoch = 0

    # Create the video train and val loaders.
    if cfg.TEST.TEST_ONLY == False:
        train_loader = loader.construct_loader(cfg, 'train')
        val_loader = loader.construct_loader(cfg, 'val')

        train_meter = RecMeter_bih(len(train_loader), cfg, 'train')
        val_meter = RecMeter_bih(len(val_loader), cfg, 'val')

        # Perform the training loop.
        logger.info("Start epoch: {}".format(start_epoch + 1))
        # print(cfg.OUTPUT_DIR)

    if cfg.TEST.TEST_ONLY == True:
        test_loader = loader.construct_loader(cfg, 'test')
        test_meter = RecMeter_fpv(len(test_loader), cfg, 'test')
        eval_epoch_bi(test_loader, model, test_meter, 0, cfg)
        print('Done.')
        exit(1)

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        # Train for one epoch.
        train_epoch_bi(train_loader, model, optimizer, train_meter, cur_epoch, cfg)

        # Compute precise BN stats.
        if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
            logger.info('Updating precise BN stats')
            calculate_and_update_precise_bn_bi(
                train_loader, model, cfg.BN.NUM_BATCHES_PRECISE
            )
            logger.info('Precise BN update finished')

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cfg, cur_epoch):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)

        logger.info('Checkpoint saved at {}'.format(cur_epoch))

def calculate_and_update_precise_bn_bi(loader, model, num_iters=200):

    def _gen_loader():
        for fpv_inputs, tpv_inputs, _, _, meta in loader:
            if isinstance(fpv_inputs, (list,)):
                for i in range(len(fpv_inputs)):
                    fpv_inputs[i] = fpv_inputs[i].cuda(non_blocking=True)
            else:
                fpv_inputs = fpv_inputs.cuda(non_blocking=True)
            if isinstance(tpv_inputs, (list,)):
                for i in range(len(tpv_inputs)):
                    tpv_inputs[i] = tpv_inputs[i].cuda(non_blocking=True)
            else:
                tpv_inputs = tpv_inputs.cuda(non_blocking=True)
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
            yield fpv_inputs, tpv_inputs, meta

    # Update the bn stats.
    update_bn_stats_bi(model, _gen_loader(), num_iters)