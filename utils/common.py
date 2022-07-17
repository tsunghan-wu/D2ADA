import os
import sys
import torch
import random
import logging
import argparse
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta


def seed_everything(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def initialize_logging(exp_dir):
    # mkdir
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "AL_record"), exist_ok=True)
    log_fname = os.path.join(exp_dir, 'log_train.txt')
    LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    DATE_FORMAT = '%Y%m%d %H:%M:%S'
    logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
    logger = logging.getLogger("Trainer")
    logger.info(f"{'-'*20} New Experiment {'-'*20}")
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)
    return logger


def timediff(t_start, t_end):
    t_diff = relativedelta(t_end, t_start)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)


def find_topk(a, k, axis=-1, largest=True, sorted=True):
    if axis is None:
        axis_size = a.size
    else:
        axis_size = a.shape[axis]
    assert 1 <= k <= axis_size

    a = np.asanyarray(a)
    if largest:
        index_array = np.argpartition(a, axis_size-k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k)-1, axis=axis)
    else:
        index_array = np.argpartition(a, k-1, axis=axis)
        topk_indices = np.take(index_array, np.arange(k), axis=axis)
    topk_values = np.take_along_axis(a, topk_indices, axis=axis)
    if sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        if largest:
            sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
        sorted_topk_values = np.take_along_axis(
            topk_values, sorted_indices_in_topk, axis=axis)
        sorted_topk_indices = np.take_along_axis(
            topk_indices, sorted_indices_in_topk, axis=axis)
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices


def initialization(args):
    # set random seed
    seed_everything(0)
    # Initialize Logging
    logger = initialize_logging(args.model_save_dir)
    logger.info(' '.join(sys.argv))
    logger.info(args)
    return logger


def finalization(t_start, val_result, logger, args):
    # End Experiment
    t_end = datetime.now()
    logger.info(f"{'%'*20} Experiment Report {'%'*20}")
    logging.info(f"0. AL Methods: {args.active_method}")
    logging.info(f"1. Takes: {timediff(t_start, t_end)}")
    logging.info(f"2. Log dir: {args.model_save_dir} (with selection json & model checkpoint)")
    logging.info("3. Validation mIoU (Be sure to submit to google form)")
    for selection_iter in range(1, args.max_iterations + 1):
        logging.info(f"AL {selection_iter}: {val_result[selection_iter]}")
    logger.info(f"{'%'*20} Experiment End {'%'*20}")


def worker_init_fn(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)
    torch.manual_seed(worker_id)


def get_parser(mode):
    # Training configurations
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-p', '--model_save_dir', default='./test')
    # Deeplab Options
    parser.add_argument("-m", "--model", type=str, default='deeplabv3plus_resnet101',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet',
                                 'deeplabv2_resnet101', 'deeplabv2_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--freeze_bn", dest='freeze_bn', action='store_true',
                        help='Freeze BatchNorm Layer while training (defulat: True)')
    parser.set_defaults(freeze_bn=True)
    # dataset
    parser.add_argument('--src_dataset', default='GTA5', choices=['cityscapes', 'GTA5', 'SYNTHIA'],
                        help='source domain training dataset')
    parser.add_argument('--src_data_dir', default='./data/GTA5')
    parser.add_argument('--src_datalist', default='dataloader/init_data/GTA5/train.txt',
                        help='source domain training list')

    parser.add_argument('--trg_dataset', default='cityscapes', help='target domain dataset')
    parser.add_argument('--trg_data_dir', default='./data/Cityscapes')
    parser.add_argument('--trg_datalist', default='dataloader/init_data/cityscapes/train.txt',
                        help='target domain training list')

    parser.add_argument('--val_dataset', default='cityscapes', help='validation dataset')
    parser.add_argument('--val_data_dir', default='./data/Cityscapes')
    parser.add_argument('--val_datalist', default='dataloader/init_data/cityscapes/val.txt', help='validation list')

    # training related
    parser.add_argument('--num_classes', type=int, default=19, help='number of classes in dataset')
    parser.add_argument('--ignore_idx', type=int, default=255, help='ignore index')
    parser.add_argument('--train_batch_size', type=int, default=1, help='batch size for training (default: 1)')
    parser.add_argument('--val_batch_size', type=int, default=4, help='batch size for validation (default: 4)')
    parser.add_argument("--weight_decay", type=float, default=5e-4, help='weight decay (default: 5e-4)')
    parser.add_argument("--total_itrs", type=int, default=100000, help="epoch number (default: 100k)")
    parser.add_argument("--val_period", type=int, default=1000, help="validation frequency (default: 1000)")
    parser.add_argument("--train_lr", type=float, default=2.5e-4, help="learning rate (default: 2.5e-4)")
    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--train_lr_D", type=float, default=2.5e-4,
                        help="AdaptSeg Discriminator learning rate (default: 2.5e-4)")
    if 'active' in mode:
        parser.add_argument("--active_mode", default='region', choices=['scan', 'region'],
                            help="Region-based or scan-based AL method")
        parser.add_argument('--init_checkpoint', type=str, default=None,
                            help='Load init checkpoint file to skip 1st iterations.')
        parser.add_argument('--save_feat_dir', type=str, default=None,
                            help='Region feature directory.')
        parser.add_argument('--datalist_path', type=str, default=None,
                            help='Load datalist files (to continue the experiment).')
        parser.add_argument('--finetune_itrs', type=int, default=1.2e5, help='finetune iterations (default: 120k)')
        parser.add_argument("--finetune_lr", type=float, default=2.5e-4, help="learning rate (default: 2.5e-4)")
        parser.add_argument('--init_iteration', type=int, default=0,
                            help='Initial active learning iteration (default: 0)')
        parser.add_argument('--max_iterations', type=int, default=5,
                            help='Number of active learning iterations (default: 5)')
        parser.add_argument('--active_selection_size', type=int, default=29,
                            help='active selection size/images (default: 29)')
        # Hyper-parameters for our dynamic scheduling policy
        parser.add_argument('--alpha', type=float, default=1.0, help='Hyper-parameter in dynamic scheduling policy')
        parser.add_argument('--beta', type=float, default=1.0, help='Hyper-parameter in dynamic scheduling policy')
    if 'warmup' in mode:
        parser.add_argument("--warmup", type=str, default='uda_warmup',
                            choices=['uda_warmup', 'sup_warmup'], help="Warm-up strategies")

    return parser
