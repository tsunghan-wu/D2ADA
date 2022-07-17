import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from models import get_model
from dataloader import get_dataset
from dataloader.utils import DataProvider
from tqdm import tqdm


def inference(model, testset, device, save_dir):
    interp_target = torch.nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)

    test_dataset_loader = \
        DataProvider(dataset=testset, batch_size=4, shuffle=False,
                     num_workers=4, pin_memory=True, drop_last=False)
    model.eval()
    with torch.no_grad():
        N = test_dataset_loader.__len__()
        for iteration in tqdm(range(N)):
            batch = test_dataset_loader.__next__()
            # forward
            images = batch['images']
            images = images.to(device, dtype=torch.float32)
            outputs = model(images)
            prob = F.softmax(outputs.detach(), dim=1)
            labels = interp_target(prob).max(dim=1)[1].cpu().numpy().astype(np.uint8)
            # entorpys = torch.mean(-prob * torch.log2(prob + 1e-12), dim=1).cpu().numpy()
            B = images.size(0)
            for batch_idx in range(B):
                basename = batch['fnames'][batch_idx][0].split('/')[-1]
                label_fname = os.path.join(save_dir, basename)
                color_label_fname = os.path.join(save_dir, f"{basename.split('.')[0]}_color.png")
                label = labels[batch_idx]
                color_label = testset.decode_target(label).astype(np.uint8)
                color_label = Image.fromarray(color_label)
                label = Image.fromarray(label)
                label.save(label_fname)
                color_label.save(color_label_fname)


def get_args():
    parser = argparse.ArgumentParser(description='')
    # Deeplab Options
    parser.add_argument("-m", "--model", type=str, default='deeplabv2_resnet101',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet',
                                 'deeplabv2_resnet101', 'deeplabv2_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    parser.add_argument('--num_classes', type=int, default=19, help='number of classes in dataset')
    parser.add_argument('--trained_model_path', required=True, help='Trained Model Path (str)')
    parser.add_argument('--val_data_dir', default='./data/Cityscapes', help='cityscapes root')
    parser.add_argument('--val_datalist', default='dataloader/init_data/cityscapes/val.txt', help='validation list')
    parser.add_argument('--save_dir', default='result')
    args = parser.parse_args()
    print(' '.join(sys.argv))
    return args


if __name__ == "__main__":
    # args
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda:0')
    # get model
    model = get_model(model=args.model, num_classes=args.num_classes,
                      output_stride=args.output_stride, separable_conv=args.separable_conv)
    # Example1: model = get_model('deeplabv3plus_resnet101', 19, 16, False).to(device)
    # Example2: model = get_model('deeplabv2_resnet101', 19, 16, False).to(device)
    model = model.to(device)
    checkpoint = torch.load(args.trained_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # get test dataset
    print('Inference on cityscapes.')
    testset = get_dataset(name='cityscapes', data_root=args.val_data_dir, datalist=args.val_datalist, imageset='val')
    # inference
    inference(model, testset, device, args.save_dir)
