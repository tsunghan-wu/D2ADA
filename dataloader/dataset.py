import os
import dataloader.ext_transforms as et
import torch.utils.data as data
from PIL import Image
import numpy as np
from .constant import train_id_to_color, id_to_train_id, syn_id_to_train_id
# for synthia
import imageio
imageio.plugins.freeimage.download()


class CityscapesGTA5(data.Dataset):
    """GTA5 Synthetic Dataset.

    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val'
        - **transform** (callable, optional): A function/transform that takes in a PIL image
                                              and returns a transformed version.
    """

    train_transform = et.ExtCompose([
        et.ExtRandomScale((0.5, 1.5)),
        et.ExtResize((1024, 2048)),
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

    def __init__(self, root, datalist, split='train', transform=None, return_spx=False):
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

        self.split = split
        self.return_spx = return_spx
        # im_idx contains the list of each image paths
        self.im_idx = []
        if datalist is not None:
            valid_list = np.loadtxt(datalist, dtype='str')
            for img_fname, lbl_fname, spx_fname in valid_list:
                img_fullname = os.path.join(self.root, img_fname)
                lbl_fullname = os.path.join(self.root, lbl_fname)
                spx_fullname = os.path.join(self.root, spx_fname)
                self.im_idx.append([img_fullname, lbl_fullname, spx_fullname])

    @classmethod
    def encode_target(cls, target):
        return id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        return train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        img_fname, lbl_fname, spx_fname = self.im_idx[index]
        image = Image.open(img_fname).convert('RGB')
        target = Image.open(lbl_fname)
        if self.return_spx is False:
            image, lbls = self.transform(image, [target])
            target = lbls[0]
        else:
            superpixel = Image.open(spx_fname)
            image, lbls = self.transform(image, [target, superpixel])
            target, superpixel = lbls
        target = self.encode_target(target)
        if self.return_spx is False:
            sample = {'images': image, 'labels': target, 'fnames': self.im_idx[index]}
        else:
            sample = {'images': image, 'labels': target, 'spx': superpixel, 'fnames': self.im_idx[index]}
        return sample

    def __len__(self):
        return len(self.im_idx)


class SYNTHIA(data.Dataset):
    """
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val'
        - **transform** (callable, optional): A function/transform that takes in a PIL image
                                              and returns a transformed version.
    """

    train_transform = et.ExtCompose([
        et.ExtResize((1024, 2048)),
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

    def __init__(self, root, datalist='./dataloader/init_data/SYNTHIA/train.txt', 
                    split='train', transform=None, return_spx=False):
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

        self.split = split
        self.return_spx = return_spx
        # im_idx contains the list of each image paths
        self.im_idx = []
        if datalist is not None:
            valid_list = np.loadtxt(datalist, dtype='str')
            for img_fname, lbl_fname, spx_fname in valid_list:
                img_fullname = os.path.join(self.root, img_fname)
                lbl_fullname = os.path.join(self.root, lbl_fname)
                spx_fullname = os.path.join(self.root, spx_fname)
                self.im_idx.append([img_fullname, lbl_fullname, spx_fullname])


    @classmethod
    def encode_target(cls, target):
        target_copy = 255 * np.ones(np.array(target).shape, dtype=np.uint8)
        for k, v in enumerate(syn_id_to_train_id):
            target_copy[target == k] = v
        return target_copy

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        return train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        img_fname, lbl_fname, spx_fname = self.im_idx[index]
        image = Image.open(img_fname).convert('RGB')
        target = np.asarray(imageio.imread(lbl_fname, format='PNG-FI'))[:,:,0]  # uint16
        target = np.array(target, dtype=np.uint8)
        target = Image.fromarray(target)
        if self.return_spx is False:
            image, lbls = self.transform(image, [target])
            target = lbls[0]
        else:
            superpixel = Image.open(spx_fname)
            image, lbls = self.transform(image, [target, superpixel])
            target, superpixel = lbls
        target = self.encode_target(target)
        if self.return_spx is False:
            sample = {'images': image, 'labels': target, 'fnames': self.im_idx[index]}
        else:
            sample = {'images': image, 'labels': target, 'spx': superpixel, 'fnames': self.im_idx[index]}
        return sample

    def __len__(self):
        return len(self.im_idx)
