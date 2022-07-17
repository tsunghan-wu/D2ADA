import os
import argparse
import numpy as np

from fast_slic import Slic
from PIL import Image


def gen_cityscapes_spx(datadir):
    spx_root_dir = os.path.join(datadir, "superpixel")
    os.makedirs(spx_root_dir, exist_ok=True)
    img_root_dir = os.path.join(datadir, "leftImg8bit")
    for root, dirs, files in os.walk(img_root_dir, topdown=False):
        for name in dirs:
            path = os.path.join(root, name)
            path = path.replace(img_root_dir, spx_root_dir)
            os.makedirs(path, exist_ok=True)

    for root, dirs, files in os.walk(img_root_dir, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            with Image.open(path) as f:
                image = np.array(f)
            slic = Slic(num_components=40, compactness=10, min_size_factor=0)
            assignment = slic.iterate(image)  # Cluster Map
            path = path.replace('leftImg8bit', 'superpixel')
            im = Image.fromarray(assignment.astype(np.uint8))
            im.save(path)


def gen_GTA5_spx(datadir):
    spx_root_dir = os.path.join(datadir, "superpixel")
    os.makedirs(spx_root_dir, exist_ok=True)
    img_root_dir = os.path.join(datadir, "images")

    for files in os.listdir(img_root_dir):
        path = os.path.join(img_root_dir, files)
        with Image.open(path) as f:
            image = np.array(f)
        slic = Slic(num_components=40, compactness=10, min_size_factor=0)
        assignment = slic.iterate(image)  # Cluster Map
        path = path.replace('images', 'superpixel')
        im = Image.fromarray(assignment.astype(np.uint8))
        im.save(path)


def gen_SYNTHIA_spx(datadir):
    spx_root_dir = os.path.join(datadir, "superpixel")
    os.makedirs(spx_root_dir, exist_ok=True)
    img_root_dir = os.path.join(datadir, "RGB")

    for files in os.listdir(img_root_dir):
        path = os.path.join(img_root_dir, files)
        with Image.open(path) as f:
            image = np.array(f)
        slic = Slic(num_components=40, compactness=10, min_size_factor=0)
        assignment = slic.iterate(image)  # Cluster Map
        path = path.replace('RGB', 'superpixel')
        im = Image.fromarray(assignment.astype(np.uint8))
        im.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cityscapes',
                        choices=['cityscapes', 'GTA5', 'SYNTHIA'])
    parser.add_argument('--datadir', default='dataset/Cityscapes/')
    args = parser.parse_args()
    if args.dataset == 'cityscapes':
        gen_cityscapes_spx(args.datadir)
    elif args.dataset == 'GTA5':
        gen_GTA5_spx(args.datadir)
    elif args.dadtaset == 'SYNTHIA':
        gen_SYNTHIA_spx(args.datadir)
