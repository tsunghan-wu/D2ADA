#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime

# model, dataset, utils
from agent.base_agent import BaseTrainer
from agent.adaptseg_agent import AdaptSegTrainer
from dataloader import get_dataset
from utils.common import initialization, timediff, get_parser


class UDATrainer(AdaptSegTrainer):
    def __init__(self, args, logger):
        super().__init__(args, logger, 1)

    def train(self):
        # prepare dataset
        src_dataset = \
            get_dataset(name=self.args.src_dataset, data_root=self.args.src_data_dir,
                        datalist=self.args.src_datalist, imageset='train')
        trg_dataset = \
            get_dataset(name=self.args.trg_dataset, data_root=self.args.trg_data_dir,
                        datalist=self.args.trg_datalist, imageset='train')
        val_dataset = get_dataset(name=self.args.val_dataset, data_root=self.args.val_data_dir,
                                  datalist=self.args.val_datalist, imageset='val')
        self.src_loader = self.get_trainloader(src_dataset)
        self.trg_loader = self.get_trainloader(trg_dataset)
        self.val_dataset_loader = self.get_valloader(val_dataset)
        self.checkpoint_file = os.path.join(self.model_save_dir, 'checkpoint.tar')

        total_itrs = int(self.args.total_itrs)
        val_period = int(self.args.val_period)
        self.train_uda_impl(total_itrs, val_period)


class SupTrainer(BaseTrainer):
    def __init__(self, args, logger):
        super().__init__(args, logger, 1)

    def train(self):
        # prepare dataset
        train_dataset = \
            get_dataset(name=self.args.trg_dataset, data_root=self.args.trg_data_dir,
                        datalist=self.args.trg_datalist, imageset='train')
        val_dataset = get_dataset(name='cityscapes', data_root=self.args.val_data_dir,
                                  datalist=self.args.val_datalist, imageset='val')
        self.train_dataset_loader = self.get_trainloader(train_dataset)
        self.val_dataset_loader = self.get_valloader(val_dataset)
        self.checkpoint_file = os.path.join(self.model_save_dir, 'checkpoint.tar')

        total_itrs = int(self.args.total_itrs)
        val_period = int(self.args.val_period)
        self.train_impl(total_itrs, val_period)


def main(args):
    # initialization
    logger = initialization(args)
    t_start = datetime.now()
    # Training
    if args.warmup == 'sup_warmup':
        trainer = SupTrainer(args, logger)
    elif args.warmup == 'uda_warmup':
        trainer = UDATrainer(args, logger)
    trainer.train()
    # Evaluate on validation set
    fname = os.path.join(args.model_save_dir, 'checkpoint.tar')
    trainer.load_checkpoint(fname)
    result = trainer.validate(update_ckpt=False)
    t_end = datetime.now()
    # End Experiment
    t_end = datetime.now()
    logger.info(f"{'%'*20} Experiment Report {'%'*20}")
    logger.info("0. Methods: Fully Supervision")
    logger.info(f"1. Takes: {timediff(t_start, t_end)}")
    logger.info(f"2. Log dir: {args.model_save_dir} (with selection json & model checkpoint)")
    logger.info("3. Validation mIoU (Be sure to submit to google form)")
    logger.info(result)
    logger.info(f"{'%'*20} Experiment End {'%'*20}")


if __name__ == '__main__':
    parser = get_parser(mode='warmup')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
