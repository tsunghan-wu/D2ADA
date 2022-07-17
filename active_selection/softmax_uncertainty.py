import os
import json
import pandas as pd
import torch
from tqdm import tqdm
import torch.distributed as dist
from active_selection.utils import get_al_loader


"""
- Image-based Active Learning Example: SoftmaxUncertaintySelector
    - Key idea: choose the minimal max_prob image as the most valuable one.
    - select_next_batch --> calculate_scores --> active_set.expand_training_set()
    - calculate_scores: inference all poolset and then record their scores.
"""


def softmax_confidence(preds):
    prob = torch.nn.functional.softmax(preds, dim=1)
    CONF = torch.max(prob, 1)[0]
    CONF *= -1  # The small the better --> Reverse it makes it the large the better
    return CONF


def softmax_margin(preds):
    prob = torch.nn.functional.softmax(preds, dim=1)
    TOP2 = torch.topk(prob, 2, dim=1)[0]
    MARGIN = TOP2[:, 0] - TOP2[:, 1]
    MARGIN *= -1   # The small the better --> Reverse it makes it the large the better
    return MARGIN


def softmax_entropy(preds):
    # Softmax Entropy
    prob = torch.nn.functional.softmax(preds, dim=1)
    ENT = torch.mean(-prob * torch.log2(prob + 1e-12), dim=1)  # The large the better
    return ENT


class SoftmaxUncertaintySelector:

    def __init__(self, batch_size, num_workers, active_method):
        self.batch_size = batch_size
        self.num_workers = num_workers
        assert active_method in ['softmax_confidence', 'softmax_margin', 'softmax_entropy']
        if active_method == 'softmax_confidence':
            self.uncertain_handler = softmax_confidence
        if active_method == 'softmax_margin':
            self.uncertain_handler = softmax_margin
        if active_method == 'softmax_entropy':
            self.uncertain_handler = softmax_entropy

    def calculate_scores(self, trainer, pool_set):
        model = trainer.net
        model.eval()
        loader, idx = get_al_loader(trainer, pool_set, self.batch_size, self.num_workers)
        print(idx)

        scores = []
        tqdm_loader = tqdm(loader, total=len(loader))
        with torch.no_grad():
            for i_iter_test, batch in enumerate(tqdm_loader):
                images = batch['images']
                # validation
                images = images.to(trainer.device, dtype=torch.float32)
                # forward
                torch.cuda.synchronize()
                preds = model(images)
                # preds shape: (B, Class, H, W)
                uncertainty = self.uncertain_handler(preds)
                uncertainty = uncertainty.cpu().detach().numpy()  # (B, Class, H, W)

                for batch_idx in range(self.batch_size):
                    # fname = batch['file_name'][batch_idx]
                    # assert fname == pool_set.im_idx[idx]
                    score = uncertainty[batch_idx].mean()
                    scores.append(score.item())

                    idx += 1
                    if idx >= len(pool_set.im_idx):
                        break
                if idx >= len(pool_set.im_idx):
                    break
        fname = os.path.join(trainer.model_save_dir, "AL_record", f"region_val_{trainer.local_rank}.json")
        with open(fname, "w") as f:
            json.dump(scores, f)

    def select_next_batch(self, trainer, active_set, selection_count):
        self.calculate_scores(trainer, active_set.trg_pool_dataset)
        if trainer.distributed is False:
            fname = os.path.join(trainer.model_save_dir, "AL_record", "region_val_0.json")
            with open(fname, "r") as f:
                scores = json.load(f)
            # Comment: Reverse=True means the large the (former / better)
            selected_samples = list(zip(*sorted(zip(scores, active_set.trg_pool_dataset.im_idx),
                                    key=lambda x: x[0], reverse=True)))[1][:selection_count]
            active_set.expand_training_set(selected_samples)
        else:
            dist.barrier()
            if trainer.local_rank == 0:
                scores = []
                for i in range(dist.get_world_size()):
                    fname = os.path.join(trainer.model_save_dir, "AL_record", f"region_val_{i}.json")
                    with open(fname, "r") as f:
                        scores.extend(json.load(f))
                # Comment: Reverse=True means the large the (former / better)
                selected_samples = list(zip(*sorted(zip(scores, active_set.trg_pool_dataset.im_idx),
                                        key=lambda x: x[0], reverse=True)))[1][:selection_count]
                active_set.expand_training_set(selected_samples)


class RegionSoftmaxUncertaintySelector:

    def __init__(self, batch_size, num_workers, active_method):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.active_method = active_method
        assert active_method in ['softmax_confidence', 'softmax_margin', 'softmax_entropy']
        if active_method == 'softmax_confidence':
            self.uncertain_handler = softmax_confidence
        if active_method == 'softmax_margin':
            self.uncertain_handler = softmax_margin
        if active_method == 'softmax_entropy':
            self.uncertain_handler = softmax_entropy

    def calculate_scores(self, trainer, pool_set):
        model = trainer.net
        model.eval()

        loader, idx = get_al_loader(trainer, pool_set, self.batch_size, self.num_workers)
        print(idx)

        scores = []
        tqdm_loader = tqdm(loader, total=len(loader))
        with torch.no_grad():
            for i_iter_test, batch in enumerate(tqdm_loader):
                images = batch['images']
                suppixs = batch['spx']
                # validation
                images = images.to(trainer.device, dtype=torch.float32)
                # forward
                torch.cuda.synchronize()
                preds = model(images)  # (B, Class, H, W)
                uncertainty = self.uncertain_handler(preds)
                uncertainty = uncertainty.cpu().detach().numpy()  # (B, Class, H, W)

                for batch_idx in range(self.batch_size):
                    suppix = suppixs[batch_idx]
                    suppix = suppix.reshape(-1)
                    uncertain = uncertainty[batch_idx].reshape(-1)

                    # Groupby
                    # table
                    # "image-path xxx (key)" "suppix id#" "confidence score"
                    # "image-path ooo (key)" "suppix id#" "confidence score"

                    # get key (image path)
                    key = pool_set.im_idx[idx]

                    # (form a dataframe containing suppix id and uncertain value)
                    df = pd.DataFrame({'id': suppix, 'val': uncertain})
                    df1 = df.groupby('id')['val'].agg(['count', 'mean']).reset_index()  # pandas groupby method
                    # only preserve regions in unlabeled set
                    table = df1[df1['id'].isin(pool_set.suppix[key[2]])].drop(columns=['count'])
                    table['key'] = ",".join(key)
                    table = table.reindex(columns=['mean', 'key', 'id'])   # form the list
                    region_score = list(table.itertuples(index=False, name=None))
                    scores.extend(region_score)

                    idx += 1
                    if idx >= len(pool_set.im_idx):
                        break
                if idx >= len(pool_set.im_idx):
                    break
        fname = os.path.join(trainer.model_save_dir, "AL_record", f"region_val_{trainer.local_rank}.json")
        with open(fname, "w") as f:
            json.dump(scores, f)

    def select_next_batch(self, trainer, active_set, selection_count):
        self.calculate_scores(trainer, active_set.trg_pool_dataset)
        fname = os.path.join(trainer.model_save_dir, "AL_record", "region_val_0.json")
        with open(fname, "r") as f:
            scores = json.load(f)
        selected_samples = sorted(scores, reverse=True)
        active_set.expand_training_set(selected_samples, selection_count, self.active_method)
