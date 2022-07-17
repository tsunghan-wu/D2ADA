import os
import json
import torch
import pandas as pd
from tqdm import tqdm
import torch.distributed as dist
from active_selection.utils import get_al_loader


"""
- Image-based Active Learning Example: SoftmaxUncertaintySelector
    - Key idea: choose the minimal max_prob image as the most valuable one.
    - select_next_batch --> calculate_scores --> active_set.expand_training_set()
    - calculate_scores: inference all poolset and then record their scores.
"""


def H(x):
    return -1 * torch.sum(torch.log2(x) * x, dim=1)


class ImportanceWeightSelector:

    def __init__(self, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers

    def calculate_scores(self, trainer, pool_set, adaptseg_trainer):
        model = trainer.net
        model_D = adaptseg_trainer.model_D
        model.eval()
        model_D.eval()
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
                # Casting for multiple return (AdaptSeg)
                # preds shape: (B, Class, H, W)
                D_out = model_D(torch.nn.functional.softmax(preds, dim=1))
                # D_out shape: (B, 1, H/32, W/32)
                D_out = torch.sigmoid(D_out)
                D_out = torch.mean(D_out.view(D_out.shape[0], -1), dim=1)

                # D_out = D_out[:,1] / D_out.sum(dim=1)
                # w = (1 - D_out) / D_out  # Importance weight

                w = D_out / (1 - D_out)  # Importance weight
                # w = torch.mean(w.view(w.shape[0], -1), dim=1)
                preds = preds.view(preds.shape[0], -1)
                preds = torch.nn.functional.softmax(preds, dim=1)

                s = w * H(preds)  # shape: (B, 1)
                s = s.tolist()
                scores.extend(s)

        fname = os.path.join(trainer.model_save_dir, "AL_record", f"region_val_{trainer.local_rank}.json")
        with open(fname, "w") as f:
            json.dump(scores, f)

    def select_next_batch(self, trainer, active_set, selection_count, adaptseg_trainer):
        self.calculate_scores(trainer, active_set.trg_pool_dataset, adaptseg_trainer)
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


class RegionAADASelector:

    def __init__(self, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers

    def calculate_scores(self, trainer, pool_set, adaptseg_trainer):
        model = trainer.net
        model.eval()
        model_D = adaptseg_trainer.model_D
        model_D.eval()

        loader, idx = get_al_loader(trainer, pool_set, self.batch_size, self.num_workers)
        print(idx)

        scores = []
        tqdm_loader = tqdm(loader, total=len(loader))
        interp_target = torch.nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
        with torch.no_grad():
            for i_iter_test, batch in enumerate(tqdm_loader):
                images = batch['images']
                suppixs = batch['spx'].numpy()
                # validation
                images = images.to(trainer.device, dtype=torch.float32)
                # forward
                torch.cuda.synchronize()
                preds = model(images)  # (B, Class, H, W)
                D_out = model_D(torch.nn.functional.softmax(preds, dim=1))
                D_out = torch.sigmoid(D_out)
                # D_out shape: (B, 1, H/32, W/32)
                importances = interp_target(D_out).squeeze(1).cpu().detach().numpy()
                # importance = importance[:, 0, :, :]
                for batch_idx in range(self.batch_size):
                    suppix = suppixs[batch_idx]
                    suppix = suppix.reshape(-1)
                    importance = importances[batch_idx].reshape(-1)
                    # Groupby
                    # table
                    # "image-path xxx (key)" "suppix id#" "confidence score"
                    # "image-path ooo (key)" "suppix id#" "confidence score"

                    # get key (image path)
                    key = pool_set.im_idx[idx]

                    # (form a dataframe containing suppix id and uncertain value)
                    df = pd.DataFrame({'id': suppix, 'val': importance})
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

    def select_next_batch(self, trainer, active_set, selection_count, adaptseg_trainer):
        self.calculate_scores(trainer, active_set.trg_pool_dataset, adaptseg_trainer)
        if trainer.distributed is False:
            fname = os.path.join(trainer.model_save_dir, "AL_record", "region_val_0.json")
            with open(fname, "r") as f:
                scores = json.load(f)
            selected_samples = sorted(scores, reverse=True)
            active_set.expand_training_set(selected_samples, selection_count)
        else:
            dist.barrier()
            if trainer.local_rank == 0:
                scores = []
                for i in range(dist.get_world_size()):
                    fname = os.path.join(trainer.model_save_dir, "AL_record", f"region_val_{i}.json")
                    with open(fname, "r") as f:
                        scores.extend(json.load(f))

                selected_samples = sorted(scores, reverse=True)
                active_set.expand_training_set(selected_samples, selection_count)
