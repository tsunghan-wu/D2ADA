import json
import os
import dataloader.ext_transforms as et
import torch.utils.data as data
from PIL import Image
import numpy as np
from .constant import train_id_to_color, id_to_train_id
# for synthia
import imageio
imageio.plugins.freeimage.download()


class RegionCityscapesGTA5(data.Dataset):
    # Pre-defined augmentation methods
    train_transform = et.ExtCompose([
        et.ExtRandomScale((0.5, 1.5)),
        et.ExtResize((1024, 2048)),
        # et.ExtRandomCrop(size=(500, 500)),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        et.ExtResize((1024, 2048)),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, root, datalist, split='train', transform=None, return_spx=False,
                 region_dict="dataloader/init_data/cityscapes/train.dict", mask_region=True):
        self.root = os.path.expanduser(root)
        if split not in ['train', 'test', 'val', 'active-label', 'active-ulabel', 'custom-set']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val" or split="active-label" or split="active-ulabel" or split="custom-set"')
        if transform is not None:
            self.transform = transform
        else:  # Use default transform
            if split in ["train", "active-label"]:
                self.transform = self.train_transform
            elif split in ["val", "test", "active-ulabel", "custom-set"]:
                self.transform = self.val_transform

        json_dict = self._load_json(region_dict)

        self.split = split
        self.return_spx = return_spx
        self.mask_region = mask_region
        # im_idx contains the list of each image paths
        self.im_idx = []
        self.suppix = {}
        if datalist is not None:
            valid_list = np.loadtxt(datalist, dtype='str')
            for img_fname, lbl_fname, spx_fname in valid_list:
                img_fullname = os.path.join(self.root, img_fname)
                lbl_fullname = os.path.join(self.root, lbl_fname)
                spx_fullname = os.path.join(self.root, spx_fname)
                self.im_idx.append([img_fullname, lbl_fullname, spx_fullname])
                self.suppix[spx_fullname] = json_dict[spx_fname]

    @classmethod
    def encode_target(cls, target):
        return id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        return train_id_to_color[target]

    def __getitem__(self, index):
        img_fname, lbl_fname, spx_fname = self.im_idx[index]
        # Load image, label, and superpixel
        image = Image.open(img_fname).convert('RGB')
        target = Image.open(lbl_fname)
        superpixel = Image.open(spx_fname)
        image, lbls = self.transform(image, [target, superpixel])
        target, superpixel = lbls
        target = self.encode_target(target)
        # GT masking (mimic region-based annotation)
        if self.mask_region is True:
            h, w = target.shape
            target = target.reshape(-1)
            superpixel = superpixel.reshape(-1)
            if spx_fname in self.suppix:
                preserving_labels = self.suppix[spx_fname]
            else:
                preserving_labels = []
            mask = np.isin(superpixel, preserving_labels)
            target = np.where(mask, target, 255)
            target = target.reshape(h, w)
            superpixel = superpixel.reshape(h, w)
        if self.return_spx is False:
            sample = {'images': image, 'labels': target, 'fnames': self.im_idx[index]}
        else:
            sample = {'images': image, 'labels': target, 'spx': superpixel, 'fnames': self.im_idx[index]}
        return sample

    def __len__(self):
        return len(self.im_idx)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data
