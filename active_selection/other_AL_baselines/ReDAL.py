# basic
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
# torch
import torch
import torch.distributed as dist
from torch_scatter import scatter_mean
# custom
from active_selection.diversity import importance_reweight
from active_selection.utils import get_al_loader


class ReDALSelector:

    def __init__(self, batch_size, num_workers, n_clusters, importance_decay, trim=False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_clusters = n_clusters
        self.importance_decay = importance_decay
        self.trim = trim

    def calculate_scores(self, trainer, pool_set):
        model = trainer.net
        model.set_return_feat()
        model.eval()
        loader, idx = get_al_loader(trainer, pool_set, self.batch_size, self.num_workers)
        print(idx)
        all_feats = np.zeros((0, 256), dtype=np.float32)
        scores = []
        tqdm_loader = tqdm(loader, total=len(loader))
        with torch.no_grad():
            for i_iter_test, batch in enumerate(tqdm_loader):
                images = batch['images']
                suppixs = batch['spx']
                edge = batch['edge'].cpu().numpy()
                suppixs = suppixs.to(trainer.device, dtype=torch.long)
                images = images.to(trainer.device, dtype=torch.float32)
                # forward
                torch.cuda.synchronize()
                feats, outputs = model.feat_forward(images)  # (B, Feat, H, W)
                outputs = torch.nn.functional.softmax(outputs, dim=1)

                for batch_idx in range(self.batch_size):
                    # Get entry
                    prob = outputs[batch_idx]
                    point_feat = feats[batch_idx]
                    suppix = suppixs[batch_idx].view(-1)
                    # Softmax Entropy
                    uncertain = torch.mean(-prob * torch.log2(prob + 1e-12), dim=0)
                    uncertain = uncertain.cpu().detach().numpy()
                    uncertain = (uncertain + 0.05 * edge[batch_idx]).reshape(-1)

                    # region feature
                    feat_size = point_feat.size(0)
                    point_feat = point_feat.view(feat_size, -1)
                    point_feat = point_feat.permute(1, 0)  # transpose --> (points, feat)
                    region_feat = scatter_mean(point_feat, suppix, dim=0).cpu().numpy()
                    key = pool_set.im_idx[idx]
                    selected_row = np.array(pool_set.suppix[key[2]])
                    feat = region_feat[selected_row]
                    all_feats = np.concatenate([all_feats, feat], axis=0)
                    # Groupby
                    suppix = suppix.cpu().numpy()
                    df = pd.DataFrame({'id': suppix, 'val': uncertain})
                    df1 = df.groupby('id')['val'].agg(['count', 'mean']).reset_index()
                    table = df1[df1['id'].isin(pool_set.suppix[key[2]])].drop(columns=['count'])
                    table['key'] = ",".join(key)
                    table = table.reindex(columns=['mean', 'key', 'id'])
                    table.astype({'mean': 'float32', 'id': 'int16'})
                    region_score = list(table.itertuples(index=False, name=None))
                    scores.extend(region_score)

                    idx += 1
                    if idx >= len(pool_set.im_idx):
                        break
                if idx >= len(pool_set.im_idx):
                    break
        # save region entropy & feature
        fname = os.path.join(trainer.model_save_dir, "AL_record", f"region_val_{trainer.local_rank}.json")
        with open(fname, "w") as f:
            json.dump(scores, f)
        npy_fname = os.path.join(trainer.model_save_dir, "AL_record", f"region_feat_{trainer.local_rank}.npy")
        np.save(npy_fname, all_feats)
        model.unset_return_feat()

    def select_next_batch(self, trainer, active_set, selection_percent):
        self.calculate_scores(trainer, active_set.trg_pool_dataset)
        print("Finish calculating scores", flush=True)
        if trainer.distributed is False:
            # load uncertainty
            fname = os.path.join(trainer.model_save_dir, "AL_record", "region_val_0.json")
            with open(fname, "r") as f:
                scores = json.load(f)
            # load region feature
            feat_fname = os.path.join(trainer.model_save_dir, "AL_record", "region_feat_0.npy")
            features = np.load(feat_fname)

            selected_samples = importance_reweight(scores, features, self.n_clusters, self.importance_decay, self.trim)
            active_set.expand_training_set(selected_samples, selection_percent)
        else:
            dist.barrier()
            if trainer.local_rank == 0:
                # load uncertainty
                scores = []
                for i in range(dist.get_world_size()):
                    fname = os.path.join(trainer.model_save_dir, "AL_record", f"region_val_{i}.json")
                    with open(fname, "r") as f:
                        scores.extend(json.load(f))
                # load region feature
                feat_lst = []
                for i in range(dist.get_world_size()):
                    npy_fname = os.path.join(trainer.model_save_dir, "AL_record", f"region_feat_{i}.npy")
                    feat_lst.append(np.load(npy_fname))
                features = np.concatenate(feat_lst, 0)
                print("Finish loading scores & feature", flush=True)
                # importance reweighting (greedy approximation)
                selected_samples = \
                    importance_reweight(scores, features, self.n_clusters, self.importance_decay, self.trim)
                active_set.expand_training_set(selected_samples, selection_percent)
