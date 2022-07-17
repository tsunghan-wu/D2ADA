import os
import os.path
import hashlib
import errno
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


def collate_fn(inputs):
    # train_keys = ['images', 'labels', 'fnames', 'spx']
    train_keys = list(inputs[0].keys())
    output_batch = {}
    for key in train_keys:
        if key in ['images', 'spx', 'labels']:
            if type(inputs[0][key]).__module__ == np.__name__:
                output_batch[key] = torch.stack([torch.from_numpy(one_batch[key]) for one_batch in inputs])
            else:
                output_batch[key] = torch.stack([one_batch[key] for one_batch in inputs])
        elif key == 'fnames':
            output_batch[key] = [one_batch[key] for one_batch in inputs]
    return output_batch


class DataProvider():
    def __init__(self, dataset, batch_size, num_workers, drop_last, shuffle,
                 pin_memory):
        # dataset
        self.dataset = dataset
        self.iteration = 0
        self.epoch = 0

        # dataloader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.dataloader = \
            DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=collate_fn,
                       shuffle=self.shuffle, num_workers=self.num_workers, drop_last=self.drop_last,
                       pin_memory=self.pin_memory)
        self.dataiter = iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def __next__(self):
        try:
            batch = self.dataiter.next()
            self.iteration += 1
            return batch

        except StopIteration:
            self.epoch += 1
            self.dataiter = iter(self.dataloader)
            batch = self.dataiter.next()
            self.iteration += 1
            return batch


def gen_bar_updater(pbar):
    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def check_integrity(fpath, md5=None):
    if md5 is None:
        return True
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str): Name to save the file under. If None, use the basename of the URL
        md5 (str): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater(tqdm(unit='B', unit_scale=True))
            )
        except OSError:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater(tqdm(unit='B', unit_scale=True))
                )


def list_dir(root, prefix=False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files
