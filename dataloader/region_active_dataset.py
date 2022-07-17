import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import ConcatDataset


class RegionActiveDataset:
    def __init__(self, args, src_label_dataset, trg_pool_dataset, trg_label_dataset):
        # Active Learning intitial selection
        self.args = args
        self.selection_iter = 0
        self.src_label_dataset = src_label_dataset
        self.trg_pool_dataset = trg_pool_dataset
        self.trg_label_dataset = trg_label_dataset

    def expand_training_set(self, sample_region, selection_count, selection_method):
        """
        Parameter: sample_region (list)
        [
            (score, scan_file_path, suppix_id),
            ...
        ]
        """
        max_selection_count = int(selection_count * 1024 * 2048)  # (number of images --> number of pixels)
        selected_count = 0
        # Active Selection
        for idx, x in enumerate(sample_region):
            _, scan_file_path, suppix_id = x
            scan_file_path = scan_file_path.split(",")
            spx_file_path = scan_file_path[2]
            suppix = Image.open(spx_file_path)
            suppix = np.array(suppix)
            selected_count += (suppix == suppix_id).sum()
            # Add into label dataset
            if scan_file_path not in self.trg_label_dataset.im_idx:
                self.trg_label_dataset.im_idx.append(scan_file_path)
                self.trg_label_dataset.suppix[spx_file_path] = [suppix_id]
            else:
                self.trg_label_dataset.suppix[spx_file_path].append(suppix_id)
            # Remove it from unlabeled dataset
            self.trg_pool_dataset.suppix[spx_file_path].remove(suppix_id)
            if len(self.trg_pool_dataset.suppix[spx_file_path]) == 0:
                self.trg_pool_dataset.suppix.pop(spx_file_path)
                self.trg_pool_dataset.im_idx.remove(scan_file_path)
            # jump out the loop when exceeding max_selection_count
            if selected_count > max_selection_count:
                fname = f'{selection_method}_selection_{self.selection_iter:02d}.pkl'
                selection_path = os.path.join(self.args.model_save_dir, fname)
                with open(selection_path, "wb") as f:
                    pickle.dump(sample_region[:idx+1], f)
                break

    def dump_datalist(self):
        datalist_path = os.path.join(self.args.model_save_dir, f'datalist_{self.selection_iter:02d}.pkl')
        with open(datalist_path, "wb") as f:
            store_data = {
                'src_label_im_idx': self.src_label_dataset.im_idx,
                'trg_label_im_idx': self.trg_label_dataset.im_idx,
                'trg_pool_im_idx': self.trg_pool_dataset.im_idx,
                'trg_label_suppix': self.trg_label_dataset.suppix,
                'trg_pool_suppix': self.trg_pool_dataset.suppix,
            }
            pickle.dump(store_data, f)

    def load_datalist(self, datalist_path=None):
        print('Load datalist', flush=True)
        # Synchronize Training Path
        if datalist_path is None:
            datalist_path = os.path.join(self.args.model_save_dir, f'datalist_{self.selection_iter:02d}.pkl')
        with open(datalist_path, "rb") as f:
            pickle_data = pickle.load(f)
        self.src_label_dataset.im_idx = pickle_data['src_label_im_idx']
        self.trg_label_dataset.im_idx = pickle_data['trg_label_im_idx']
        self.trg_pool_dataset.im_idx = pickle_data['trg_pool_im_idx']
        self.trg_label_dataset.suppix = pickle_data['trg_label_suppix']
        self.trg_pool_dataset.suppix = pickle_data['trg_pool_suppix']

    def get_trainset(self):
        return ConcatDataset([self.src_label_dataset, self.trg_label_dataset])
