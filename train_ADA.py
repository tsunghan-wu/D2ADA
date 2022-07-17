#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic
import os
import sys
from datetime import datetime

# torch
import torch

# custom
from agent.base_agent import BaseTrainer
from dataloader import get_dataset, get_active_dataset
from active_selection import get_density_selector, get_uncertain_selector
from utils.common import initialization, finalization, get_parser


class Trainer(BaseTrainer):
    def __init__(self, args, logger):
        super().__init__(args, logger, 1)

    def train(self, active_set, init_checkpoint=None):
        # prepare datasets / dataloaders / checkpoint filename
        train_dataset = active_set.get_trainset()
        val_dataset = get_dataset(name=self.args.val_dataset, data_root=self.args.val_data_dir,
                                  datalist=self.args.val_datalist, imageset='val')
        self.train_dataset_loader = self.get_trainloader(train_dataset)
        self.val_dataset_loader = self.get_valloader(val_dataset)
        self.checkpoint_file = \
            os.path.join(self.model_save_dir, f'checkpoint{active_set.selection_iter:02d}.tar')

        # Training if no initial checkpoint is provided.
        if active_set.selection_iter == 0:
            assert self.args.init_checkpoint is not None
            from shutil import copyfile
            copyfile(self.args.init_checkpoint, self.checkpoint_file)
            return
        else:
            total_iterations = int(self.args.finetune_itrs)
            val_period = int(self.args.val_period)
            self.train_impl(total_iterations, val_period)


def main(args):
    # initialization
    logger = initialization(args)
    t_start = datetime.now()
    val_result = {}
    # Active Learning dataset
    active_set = get_active_dataset(args)
    density_selector = get_density_selector(args)       # density-aware method
    uncertainty_selector = get_uncertain_selector(args)   # uncertainty-based method

    print('active learning iteration...')
    # Active Learning iteration
    # Note: iteration 0 --> warmup iteration.
    for selection_iter in range(args.init_iteration, args.max_iterations + 1):
        active_set.selection_iter = selection_iter
        if args.datalist_path is not None:
            active_set.load_datalist(args.datalist_path)
        # 1. Supervision Finetuning
        logger.info(f"AL {selection_iter}: Start Training ({selection_iter}% training data)")
        trainer = Trainer(args, logger)
        if selection_iter >= 1:
            prevckpt_fname = os.path.join(args.model_save_dir, f'checkpoint{selection_iter-1:02d}.tar')
            trainer.load_checkpoint(prevckpt_fname)
        trainer.train(active_set)

        # 2. Load best checkpoint + Evaluation
        fname = os.path.join(args.model_save_dir, f'checkpoint{selection_iter:02d}.tar')
        trainer.load_checkpoint(fname)
        val_return = trainer.validate(update_ckpt=False)
        logger.info(f"AL {selection_iter}: Get best validation result")
        val_result[selection_iter] = val_return
        torch.cuda.empty_cache()

        # 3. Active Learning Round

        # Budget Allocation (Dynamic Scheduling Policy)
        Lambda = args.alpha * 2 ** (args.beta * (1 - selection_iter))
        B_d = args.active_selection_size * Lambda
        B_u = args.active_selection_size * (1 - Lambda)
        logger.info(f"AL {selection_iter}: Select Next Batch")

        # Selection
        density_selector.select_next_batch(trainer, active_set, B_d)
        uncertainty_selector.select_next_batch(trainer, active_set, B_u)
        active_set.dump_datalist()
    # finalization
    finalization(t_start, val_result, logger, args)


if __name__ == '__main__':
    parser = get_parser(mode='sup_active')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
