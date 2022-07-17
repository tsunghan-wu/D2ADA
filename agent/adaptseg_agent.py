# torch
import torch
import torch.optim as optim
import torch.nn.functional as F
import pytorch_warmup as warmup
# model, dataset, utils
from .base_agent import BaseTrainer
from models import freeze_bn
from models.discriminator import FCDiscriminator
from utils.scheduler import PolyLR
from tqdm import tqdm


class AdaptSegTrainer(BaseTrainer):
    def __init__(self, args, logger, selection_iter):
        super().__init__(args, logger, selection_iter)
        # initialize discriminator
        self.model_D = FCDiscriminator(num_classes=args.num_classes)
        self.model_D.to(self.device)

        # Optimizer
        self.optimizer_D = optim.Adam(self.model_D.parameters(), lr=args.train_lr_D, betas=(0.9, 0.99))
        # Scheduler
        self.scheduler = PolyLR(self.optimizer, self.args.total_itrs, power=0.9)
        self.scheduler_D = PolyLR(self.optimizer_D, self.args.total_itrs, power=0.9)
        self.warmup_scheduler = warmup.LinearWarmup(self.optimizer, warmup_period=5000)
        self.warmup_scheduler_D = warmup.LinearWarmup(self.optimizer_D, warmup_period=5000)
        # adversarial loss
        self.adv_loss_fun = torch.nn.BCEWithLogitsLoss()
        print("Class init done", flush=True)

    def train(self):
        raise NotImplementedError

    def train_uda_impl(self, total_itrs, val_period):
        self.net.train()
        freeze_bn(self.net)
        self.model_D.train()
        # self.model_D2.train()
        for itr in tqdm(range(total_itrs)):
            src_batch = self.src_loader.__next__()
            trg_batch = self.trg_loader.__next__()

            self.optimizer.zero_grad()
            self.optimizer_D.zero_grad()

            # train G
            # don't accumulate grads in D
            for param in self.model_D.parameters():
                param.requires_grad = False

            # train with source
            src_images = src_batch['images']
            src_labels = src_batch['labels']
            src_images = src_images.to(self.device, dtype=torch.float32)
            src_labels = src_labels.to(self.device, dtype=torch.long)

            pred = self.net(src_images)

            loss = self.loss_fun(pred, src_labels)

            # proper normalization
            loss.backward()

            # train with target
            trg_images = trg_batch['images']
            trg_labels = trg_batch['labels']
            trg_images = trg_images.to(self.device, dtype=torch.float32)
            trg_labels = trg_labels.to(self.device, dtype=torch.long)

            pred_target = self.net(trg_images)
            D_out = self.model_D(F.softmax(pred_target, dim=1))

            loss_adv_target = \
                self.adv_loss_fun(D_out, torch.FloatTensor(D_out.data.size()).fill_(0).to(self.device))

            loss = 0.0005 * loss_adv_target
            loss.backward()

            # train D

            # bring back requires_grad
            for param in self.model_D.parameters():
                param.requires_grad = True

            # train with source
            pred = pred.detach()

            D_out = self.model_D(F.softmax(pred, dim=1))

            loss_D = self.adv_loss_fun(D_out, torch.FloatTensor(D_out.data.size()).fill_(0).to(self.device))

            loss_D = loss_D / 2

            loss_D.backward()

            # train with target
            pred_target = pred_target.detach()

            D_out = self.model_D(F.softmax(pred_target, dim=1))

            loss_D = self.adv_loss_fun(D_out, torch.FloatTensor(D_out.data.size()).fill_(1).to(self.device))

            loss_D = loss_D / 2

            loss_D.backward()

            self.optimizer.step()
            self.optimizer_D.step()
            self.scheduler.step()
            self.scheduler_D.step()
            self.warmup_scheduler.dampen()
            self.warmup_scheduler_D.dampen()
            if itr % val_period == (val_period - 1):
                if self.local_rank == 0:
                    self.logger.info('**** EVAL ITERATIONS %06d ****' % (itr))
                self.validate()
                self.net.train()
                freeze_bn(self.net)
                self.model_D.train()

    def save_checkpoint(self):
        checkpoint = {
                        'model_state_dict': self.net.state_dict(),
                        'opt_state_dict': self.optimizer.state_dict(),
                        'model_D': self.model_D.state_dict(),
                     }
        torch.save(checkpoint, self.checkpoint_file)

    def load_adaptseg_discriminator(self, fname):
        map_location = self.device
        checkpoint = torch.load(fname, map_location=map_location)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.model_D.load_state_dict(checkpoint['model_D'])
