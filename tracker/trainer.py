# COPYRIGHT 2021. Fred Fung. Boston University.
import math
import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm

from utils.logging import get_logger, TqdmToLogger
from utils import check_keys


class TrackerTrainer:
    def __init__(self, train_cfg):
        self.train_cfg = train_cfg
        self._logger = get_logger(__name__)

    def train_tracker_on_datasets(self, tracker, train_dataset, test_dataset, rank):
        self._prepare_for_training(tracker, rank)
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=self.train_cfg.TRAIN.BATCH_SIZE,
                                      sampler=train_sampler, num_workers=self.train_cfg.TRAIN.NUM_WORKERS)
        test_sampler = DistributedSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=self.train_cfg.TRAIN.BATCH_SIZE,
                                     sampler=test_sampler, num_workers=self.train_cfg.TRAIN.NUM_WORKERS)

        start_epoch = self.train_cfg.TRAIN.INIT.START_EPOCH
        best_test_loss = self.train_cfg.TRAIN.LOSS_CLIP_VALUE
        for epoch in range(start_epoch, self.train_cfg.TRAIN.EPOCH):
            best_test_loss = self._train_one_epoch(tracker, train_dataloader, test_dataloader, best_test_loss, epoch,
                                                   rank)

    def _prepare_for_training(self, tracker, rank):
        if rank == 0:
            if not os.path.exists(os.path.join(self.train_cfg.LOG_DIR, self.train_cfg.EXPR_NAME)):
                os.makedirs(os.path.join(self.train_cfg.LOG_DIR, self.train_cfg.EXPR_NAME, "summary"))
                os.makedirs(os.path.join(self.train_cfg.LOG_DIR, self.train_cfg.EXPR_NAME, "checkpoints"))
            self.tb_writer = SummaryWriter(os.path.join(self.train_cfg.LOG_DIR, self.train_cfg.EXPR_NAME, "summary"))
            with open(os.path.join(self.train_cfg.LOG_DIR, self.train_cfg.EXPR_NAME, "config.yaml"), "w") as f:
                f.write(self.train_cfg.dump())
        self.global_step = 0
        if self.train_cfg.TRAIN.BACKBONE_TRAIN:
            tracker.module.backbone.train()
            tracker.module.adjustment_layer.train()
        else:
            tracker.module.backbone.eval()
            tracker.module.adjustment_layer.eval()
            for param in tracker.module.backbone.parameters():
                param.requires_grad = False
            for param in tracker.module.adjustment_layer.parameters():
                param.requires_grad = False

        if self.train_cfg.LANG.FINETUNE_BERT:
            raise NotImplementedError("Fine-tuning BERT Model is not implemented.")
        else:
            tracker.module.nl_multi_rpn.bert_model.eval()
            for param in tracker.module.nl_multi_rpn.bert_model.parameters():
                param.requires_grad = False

        self.optimizer = torch.optim.SGD(
            params=tracker.module.get_trainable_weights(self.train_cfg.TRAIN.BACKBONE_TRAIN),
            lr=self.train_cfg.TRAIN.LR.BASE_LR,
            momentum=self.train_cfg.TRAIN.LR.MOMENTUM)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                       patience=2,
                                                                       factor=self.train_cfg.TRAIN.LR.WEIGHT_DECAY)

        if self.train_cfg.TRAIN.INIT.RESUME:
            device = torch.cuda.current_device()
            ckpt = torch.load(self.train_cfg.TRAIN.INIT.RESUME, map_location=lambda storage, loc: storage.cuda(device))
            check_keys(tracker, ckpt['state_dict'], rank)
            tracker.load_state_dict(ckpt['state_dict'], strict=False)
            check_keys(self.optimizer, ckpt['optimizer'], rank)
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.global_step = ckpt['global_step']
            if rank == 0:
                self._logger.info("RELOADED FROM {}.".format(self.train_cfg.TRAIN.INIT.RESUME))
                self._logger.info("CURRENT EPOCH {}.".format(ckpt['epoch']))
                self._logger.info("CURRENT GLOBAL_STEP {}.".format(self.global_step))
                if ckpt['epoch'] != self.train_cfg.TRAIN.INIT.START_EPOCH:
                    self._logger.warning('START EPOCH AND RELOADED EPOCH DOES NOT MATCH.')
        elif self.train_cfg.TRAIN.INIT.PRETRAINED:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(self.train_cfg.TRAIN.INIT.PRETRAINED,
                                         map_location=lambda storage, loc: storage.cuda(device))
            resnet_keys = [key for key in pretrained_dict.keys() if ('backbone' in key or "adjustment_layer" in key)]
            resnet_weights = {"module." + key: pretrained_dict[key] for key in resnet_keys}
            check_keys(tracker, resnet_weights, rank)
            tracker.load_state_dict(resnet_weights, strict=False)
            if rank == 0:
                self._logger.info(
                    "LOADED PRE-TRAINED RESNET WEIGHTS AND ADJUSTMENT LAYER WEIGHTS FROM {}.".format(
                        self.train_cfg.TRAIN.INIT.PRETRAINED_RESNET_ONLY))
        else:
            if rank == 0:
                self._logger.warning('NOT LOADING ANY WEIGHTS. INITIALIZED FROM SCRATCH.')
        if rank == 0:
            self._logger.info("TRAINING PREPARATION DONE.")

    def _train_one_epoch(self, tracker, train_dataloader, test_dataloader, best_test_loss, epoch, rank):
        if rank == 0:
            if self.train_cfg.TRAIN.BACKBONE_TRAIN:
                self._logger.info('TRAINING BACKBONE.')
            if self.train_cfg.LANG.FINETUNE_BERT:
                self._logger.info('TRAINING BERT.')
            self._logger.info('EPOCH: ' + str(epoch))
        if rank == 0:
            pbar = tqdm(total=len(train_dataloader), leave=False, desc="Training Epoch %d" % epoch, file=TqdmToLogger(),
                        mininterval=1, maxinterval=100, )

        for batch in train_dataloader:
            with torch.autograd.enable_grad():
                self.optimizer.zero_grad()
                losses = tracker.module.compute_loss(batch)
                loss = losses['total_loss']
                if not (math.isnan(loss.data.item())
                        or math.isinf(loss.data.item())
                        or loss.data.item() > self.train_cfg.TRAIN.LOSS_CLIP_VALUE):
                    loss.backward()
                    self.optimizer.step()
            self.global_step += 1
            if rank == 0:
                pbar.update()
            if self.global_step % self.train_cfg.TRAIN.PRINT_FREQ == 0:
                if rank == 0:
                    self._logger.info("EPOCH: %d, STEP: %d, LR: %.6f, TRAIN_LOSS: %.6f" % (
                        epoch, self.global_step, self.optimizer.param_groups[0]['lr'], losses["total_loss"]))
                    for k, v in losses.items():
                        self.tb_writer.add_scalar('train_loss/' + k, v, self.global_step)
                    self._logger.info("Running Evaluation.")
                    for k, v in tracker.named_parameters():
                        self.tb_writer.add_scalar("weights/%s/avg" % k, torch.mean(v), self.global_step)
                test_info = {}
                for k in losses.keys():
                    test_info[k] = []
                if rank == 0:
                    pbar_eval = tqdm(total=len(test_dataloader), leave=False, desc="Evaluation", file=TqdmToLogger(),
                                     mininterval=1, maxinterval=100)
                for test_data in test_dataloader:
                    with torch.autograd.no_grad():
                        test_loss = tracker.module.compute_loss(test_data)
                    if rank == 0:
                        pbar_eval.update()
                    for k, v in sorted(test_loss.items()):
                        test_info[k].append(v.data.item())
                if rank == 0:
                    pbar_eval.close()
                for k, v in sorted(test_info.items()):
                    test_info[k] = np.mean(test_info[k])
                if rank == 0:
                    for k, v in test_info.items():
                        self.tb_writer.add_scalar('test_loss/' + k, v, self.global_step)
                if test_info["total_loss"] < best_test_loss:
                    best_test_loss = test_info["total_loss"]
                    if rank == 0:
                        checkpoint_file = os.path.join(self.train_cfg.LOG_DIR, self.train_cfg.EXPR_NAME, "checkpoints",
                                                       'CKPT-E%d-S%d.pth' % (epoch, self.global_step))
                        torch.save(
                            {'epoch': epoch, 'global_step': self.global_step, 'state_dict': tracker.state_dict(),
                             'optimizer': self.optimizer.state_dict()}, checkpoint_file)
                        self._logger.info(
                            'NEW BEST LOSS FOUND: %f, CHECKPOINT SAVED AT: %s' % (best_test_loss, checkpoint_file))
                self.lr_scheduler.step(test_info['total_loss'])
        if rank == 0:
            pbar.close()
        return best_test_loss
