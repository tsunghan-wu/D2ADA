import os
import json
from torch.utils.data import ConcatDataset


class ActiveDataset:
    def __init__(self, args, src_label_dataset, trg_pool_dataset, trg_label_dataset):
        # Active Learning intitial selection
        self.args = args
        self.selection_iter = 0
        self.src_label_dataset = src_label_dataset
        self.trg_pool_dataset = trg_pool_dataset
        self.trg_label_dataset = trg_label_dataset

    def expand_training_set(self, paths):
        self.trg_label_dataset.im_idx.extend(paths)
        for x in paths:
            self.trg_pool_dataset.im_idx.remove(x)

    def get_fraction_of_labeled_data(self):
        label_num = len(self.trg_label_dataset.im_idx)
        pool_num = len(self.trg_pool_dataset.im_idx)
        return label_num / (label_num + pool_num)

    def dump_datalist(self):
        datalist_path = os.path.join(self.args.model_save_dir, f'datalist_{self.selection_iter:02d}.json')
        with open(datalist_path, "w") as f:
            store_data = {
                'src_label_im_idx': self.src_label_dataset.im_idx,
                'trg_label_im_idx': self.trg_label_dataset.im_idx,
                'trg_pool_im_idx': self.trg_pool_dataset.im_idx,
            }
            json.dump(store_data, f)

    def load_datalist(self, convert_root=False):
        print('Load path', flush=True)
        # Synchronize Training Path
        datalist_path = os.path.join(self.args.model_save_dir, f'datalist_{self.selection_iter:02d}.json')
        with open(datalist_path, "rb") as f:
            json_data = json.load(f)
        self.src_label_dataset.im_idx = json_data['src_label_im_idx']
        self.trg_label_dataset.im_idx = json_data['trg_label_im_idx']
        self.trg_pool_dataset.im_idx = json_data['trg_pool_im_idx']

    def get_trainset(self):
        # Mode: target_finetune or joint
        if self.selection_iter == 1:
            return self.src_label_dataset
        else:
            return ConcatDataset([self.src_label_dataset, self.trg_label_dataset])
