# torch
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim

# model, dataset, utils
from dataloader.utils import DataProvider
from models import get_model, freeze_bn
from utils.miou import MeanIoU
from utils.scheduler import PolyLR
from utils.loss import FocalLoss


class BaseTrainer(object):
    def __init__(self, args, logger, selection_iter):
        self.args = args
        self.logger = logger
        self.model_save_dir = args.model_save_dir
        self.best_iou = 0
        self.device = torch.device('cuda:0')
        self.local_rank = 0

        # prepare model
        self.num_classes = args.num_classes

        self.net = get_model(model=args.model, num_classes=self.num_classes,
                             output_stride=args.output_stride, separable_conv=args.separable_conv)

        self.net.to(self.device)
        # Optimizer
        self.optimizer = optim.SGD(params=[
            {'params': self.net.backbone.parameters(), 'lr': self.args.train_lr},
            {'params': self.net.classifier.parameters(), 'lr': 10 * self.args.train_lr},
        ], lr=self.args.train_lr, momentum=0.9, weight_decay=self.args.weight_decay)

        # Scheduler
        if hasattr(self.args, "finetune_itrs"):
            total_itrs = self.args.finetune_itrs
        else:
            total_itrs = self.args.total_itrs
        self.scheduler = PolyLR(self.optimizer, total_itrs, power=0.9)
        # Criterion
        if self.args.loss_type == 'focal_loss':
            self.loss_fun = FocalLoss(ignore_index=self.args.ignore_idx, size_average=True)
        elif self.args.loss_type == 'cross_entropy':
            self.loss_fun = torch.nn.CrossEntropyLoss(ignore_index=self.args.ignore_idx, reduction='mean')
        print("Class init done", flush=True)

    def get_trainloader(self, dataset):
        data_provider = DataProvider(dataset=dataset, batch_size=self.args.train_batch_size,
                                     shuffle=True, num_workers=self.args.train_batch_size,
                                     pin_memory=True, drop_last=True)
        return data_provider

    def get_valloader(self, dataset):

        data_provider = DataProvider(dataset=dataset, batch_size=self.args.val_batch_size,
                                     shuffle=False, num_workers=self.args.val_batch_size,
                                     pin_memory=True, drop_last=False)
        return data_provider

    def train(self):
        raise NotImplementedError

    def train_impl(self, total_itrs, val_period):
        self.net.train()
        if self.args.freeze_bn is True:
            freeze_bn(self.net)
        for iteration in tqdm(range(total_itrs)):
            batch = self.train_dataset_loader.__next__()
            # training
            images = batch['images']
            labels = batch['labels']
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward + backward + optimize
            # torch.cuda.synchronize()
            preds = self.net(images)
            if isinstance(preds, tuple):  # for multi-level
                preds = preds[1]

            loss = self.loss_fun(preds, labels)
            loss.backward()
            self.optimizer.step()
            # update learning rate scheduler
            self.scheduler.step()

            if iteration % val_period == (val_period - 1) and iteration > 80000:
                if self.local_rank == 0:
                    self.logger.info('**** EVAL ITERATION %06d ****' % (iteration))
                self.validate()
                self.net.train()
                if self.args.freeze_bn is True:
                    freeze_bn(self.net)

    def validate(self, update_ckpt=True):
        self.net.eval()
        iou_helper = MeanIoU(self.num_classes, self.args.ignore_idx)
        iou_helper._before_epoch()

        with torch.no_grad():
            N = self.val_dataset_loader.__len__()
            for iteration in range(N):
                batch = self.val_dataset_loader.__next__()
                images = batch['images']
                labels = batch['labels']
                # validation
                images = images.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)
                # forward
                # torch.cuda.synchronize()
                outputs = self.net(images)
                preds = outputs.detach().max(dim=1)[1]
                # calculate IoU Score
                output_dict = {
                    'outputs': preds,
                    'targets': labels
                }
                iou_helper._after_step(output_dict)
            # Prepare Logging
            iou_table = []
            # SYNTHIA: miou, miou*, per-class iou
            # GTA5   : miou, per-class iou
            if self.args.src_dataset == "SYNTHIA":
                class_16_ignore = [9, 14, 16]
                ious = iou_helper._after_epoch(class_16_ignore)
                val_miou = np.mean(ious)
                ious_13 = ious[:3] + ious[6:]
                val_miou_star = np.mean(ious_13)
                iou_table.append(f'{val_miou:.2f},{val_miou_star:.2f}')
            else:
                ious = iou_helper._after_epoch()
                val_miou = np.mean(ious)
                iou_table.append(f'{val_miou:.2f}')
            # Append per-class ious
            for class_iou in ious:
                iou_table.append(f'{class_iou:.2f}')
            iou_table_str = ','.join(iou_table)
            # save model if performance is improved
            del iou_table
            del output_dict
            print(iou_table_str, flush=True)
            if update_ckpt is False:
                return iou_table_str

            if self.local_rank == 0:
                self.logger.info('[Validation Result]')
                self.logger.info('%s' % (iou_table_str))
                if self.best_iou < val_miou:
                    self.best_iou = val_miou
                    self.save_checkpoint()

                self.logger.info('Current val miou is %.3f %%, while the best val miou is %.3f %%'
                                 % (val_miou, self.best_iou))
            return iou_table_str

    def save_checkpoint(self):
        checkpoint = {
                        'model_state_dict': self.net.state_dict(),
                        'opt_state_dict': self.optimizer.state_dict()
                     }
        torch.save(checkpoint, self.checkpoint_file)

    def load_checkpoint(self, fname, load_optimizer=False):
        map_location = self.device
        checkpoint = torch.load(fname, map_location=map_location)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        if load_optimizer is True:
            self.optimizer.load_state_dict(checkpoint['opt_state_dict'])
