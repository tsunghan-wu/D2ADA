# basic
import os
from torch.utils.data.dataset import ConcatDataset
from active_selection.utils import get_al_loader
from utils.common import worker_init_fn
import torch
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from sklearn.metrics import pairwise_distances
from dataloader.utils import collate_fn
from torch_scatter import scatter_mean
import torch.nn.functional as F

class CoreSetSelector:

    def __init__(self, batch_size, num_workers): # (8, 8)?
        self.batch_size = batch_size
        self.num_workers = num_workers

    def calculate_scores(self, trainer, active_set):
        model = trainer.net
        model.eval()

        # ALL Dataset
        src_label_dataset = active_set.src_label_dataset
        trg_label_dataset = active_set.trg_label_dataset
        trg_pool_dataset = active_set.trg_pool_dataset
        core_list = src_label_dataset.im_idx + trg_label_dataset.im_idx
        all_list = core_list + trg_pool_dataset.im_idx
        all_dataset = ConcatDataset([src_label_dataset, trg_label_dataset, trg_pool_dataset])

        loader, idx = get_al_loader(trainer, all_dataset, self.batch_size, self.num_workers)
        print(idx)
        feature = []
        tqdm_loader = tqdm(loader, total=len(loader))
        with torch.no_grad():
            for i_iter_test, batch in enumerate(tqdm_loader):
                images = batch['images']
                
                images = images.to(trainer.device, dtype=torch.float32)
                # forward
                torch.cuda.synchronize()
                # preds = model(images)  # (B, Class, H, W) torch.Size([8, 19, 1024, 2048])
                preds = model.backbone(images)['out']
                preds = preds.cpu().numpy() # np.concatenate cannot be performed in cuda
                for batch_idx in range(self.batch_size):
                    # fname = batch['file_name'][batch_idx]
                    # assert fname == combine_lst[idx]

                    feat = preds[batch_idx].mean(axis=0).reshape(1, -1)
                    # print(feat.shape) # torch.Size([1, 8192])
                    feature.append(feat)

                    idx += 1
                    if idx >= len(all_dataset):
                        break
                if idx >= len(all_dataset):
                    break
        feat_np = np.concatenate(feature, 0)
        # print(len(feat_np), feat_np[0].shape) # 2952 (8192,)
        fname = os.path.join(trainer.model_save_dir, "AL_record", f"coreset_feat_{trainer.local_rank}.npy")
        np.save(fname, feat_np)
        return core_list, all_list

    def _updated_distances(self, cluster_centers: list, features, min_distances):
        x = features[cluster_centers, :]
        # print(x.shape) # (1, 8192)
        dist = pairwise_distances(features, x, metric='euclidean')
        # print(dist.shape) # (2952, 1)
        if min_distances is None:
            return np.min(dist, axis=1).reshape(-1, 1)
        else:
            return np.minimum(min_distances, dist)

    def _select_batch(self, features, selected_indices: list, N):
        new_batch = []
        min_distances = self._updated_distances(selected_indices, features, None)
        for _ in range(N):
            ind = np.argmax(min_distances)
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in selected_indices
            min_distances = self._updated_distances([ind], features, min_distances)
            new_batch.append(ind)

        print('Maximum distance from cluster centers is %0.5f' % max(min_distances))
        return new_batch

    def select_next_batch(self, trainer, active_set, selection_count):
        core_list, all_list = self.calculate_scores(trainer, active_set)
        if trainer.distributed is False:
            fname = os.path.join(trainer.model_save_dir, "AL_record", "coreset_feat_0.npy")
            features = np.load(fname)
        else:
            dist.barrier()
            if trainer.local_rank == 0:
                feat_lst = []
                for i in range(dist.get_world_size()):
                    fname = os.path.join(trainer.model_save_dir, "AL_record", f"coreset_feat_{i}.npy")
                    feat_lst.append(np.load(fname))
                features = np.concatenate(feat_lst, 0)
        print(f'coreset active selection: # of cores:{len(core_list)}, # of total:{len(all_list)}.')
        if trainer.local_rank == 0:
            core_size = len(core_list)
            selected_indices = self._select_batch(features, list(range(core_size)), selection_count)
            active_set.expand_training_set([all_list[i] for i in selected_indices])

'''
    RegionCoreSetSelector currently cannot work because we have (tot_lbl_regions, tot_ulbl_regions)
    = (1115160, 996160) after doing pairwise_distances, which is too large to store in memory.
'''
class RegionCoreSetSelector:

    def __init__(self, args, batch_size, num_workers): # (2, 2)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args

    def feature_extractor(self, model, dataset, device, save_dir, save_prob=False):
        if isinstance(dataset, torch.utils.data.dataset.ConcatDataset):
            dataset.datasets[0].return_spx = True       # src_label_dataset
            dataset.datasets[1].return_spx = True       # trg_label_dataset
            trg_suppix = dataset.datasets[1].suppix
        else:
            dataset.return_spx = True
            trg_suppix = dataset.suppix
        dataset_loader = \
            torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size,
                                        shuffle=False, collate_fn=collate_fn,
                                        num_workers=self.num_workers, pin_memory=True, drop_last=False)
        model.eval()
        total_regions = 0
        with torch.no_grad():
            for batch in tqdm(dataset_loader):
                # forward
                images = batch['images']
                images = images.to(device, dtype=torch.float32)
                suppixs = batch['spx'].to(device, dtype=torch.long)
                model.set_return_feat()
                feats, probs = model.feat_forward(images)
                model.unset_return_feat()
                if save_prob is True:
                    image_outputs = model(images)
                    image_probs = F.softmax(image_outputs, dim=1).cpu().numpy()
                probs = F.softmax(probs, dim=1)
                N = images.size(0)
                for batch_idx in range(N):
                    suppix = suppixs[batch_idx].view(-1)
                    spx_fname = batch['fnames'][batch_idx][2]
                    point_feat = feats[batch_idx]
                    feat_size = point_feat.size(0)
                    # region feature
                    point_feat = point_feat.view(feat_size, -1)
                    point_feat = point_feat.permute(1, 0)  # transpose --> (points, feat)
                    region_feat = scatter_mean(point_feat, suppix, dim=0).cpu().numpy()   # 40x256

                    # region class-wise probability
                    point_prob = probs[batch_idx].view(19, -1)
                    point_prob = point_prob.permute(1, 0)  # transpose --> (points, feat)
                    region_prob = scatter_mean(point_prob, suppix, dim=0).cpu().numpy()   # 40x19

                    if 'Cityscapes' in spx_fname:
                        suppix_list = trg_suppix[spx_fname]
                        region_feat = region_feat[suppix_list]
                        region_prob = region_prob[suppix_list]
                    else:
                        suppix_list = np.arange(40)
                    # save result
                    basename = batch['fnames'][batch_idx][0].split('/')[-1].split('.')[0]
                    fname = os.path.join(save_dir, f"{basename}.npz")
                    if save_prob is True:
                        image_prob = image_probs[batch_idx]
                        np.savez(fname, image_prob=image_prob, feat=region_feat,
                                 prob=region_prob, suppix_list=suppix_list)
                    else:
                        np.savez(fname, feat=region_feat, prob=region_prob, suppix_list=suppix_list)
                    total_regions += len(suppix_list)
                    del region_feat
                    del region_prob
        return total_regions

    def load_data(self, data_fnames, feat_dir, total_regions):
        feat_dim = 256
        n_class = 19
        feats = np.zeros((total_regions, feat_dim), dtype=np.float32)
        probs = np.zeros((total_regions, n_class), dtype=np.float32)
        spx_infos = np.empty((total_regions, 4), dtype=np.object)
        cur_idx = 0
        for idx, fname in enumerate(tqdm(data_fnames)):
            basename = fname[0].split("/")[-1].split(".")[0]
            abs_fname = os.path.join(feat_dir, f'{basename}.npz')
            arr = np.load(abs_fname)
            feat = arr['feat']
            prob = arr['prob']
            suppix_list = arr['suppix_list']
            num_spx = len(suppix_list)
            spx_fname = np.repeat([fname], num_spx, axis=0)
            spx_info = np.hstack([spx_fname, suppix_list.reshape(-1, 1)])
            # append
            feats[cur_idx: cur_idx+num_spx] = feat
            probs[cur_idx: cur_idx+num_spx] = prob
            spx_infos[cur_idx: cur_idx+num_spx] = spx_info
            cur_idx += num_spx
        return feats, probs, spx_infos

    def calculate_scores(self, trainer, active_set):
        model = trainer.net

        src_label_dataset = active_set.src_label_dataset
        trg_label_dataset = active_set.trg_label_dataset
        lbl_dataset = ConcatDataset([src_label_dataset, trg_label_dataset])
        lbl_dir = os.path.join(self.args.save_feat_dir, "label")
        os.makedirs(lbl_dir, exist_ok=True)

        pool_dataset = active_set.trg_pool_dataset
        pool_dir = os.path.join(self.args.save_feat_dir, "pool")
        os.makedirs(pool_dir, exist_ok=True)

        lbl_list = src_label_dataset.im_idx + trg_label_dataset.im_idx
        pool_list = pool_dataset.im_idx
        print('selection_iter', active_set.selection_iter, flush=True)
        print('starting feature extractor...', flush=True)
        lbl_total_regions = self.feature_extractor(model, lbl_dataset, torch.device('cuda:0'), lbl_dir)
        pool_total_regions = self.feature_extractor(model, pool_dataset, torch.device('cuda:0'), pool_dir)
        print('finish feature extractor', flush=True)

        # Load Feature and probability for sample selection
        print('loading data...', flush=True)
        lbl_feat, lbl_prob, lbl_list = self.load_data(lbl_list, lbl_dir, lbl_total_regions)
        pool_feat, pool_prob, pool_list = self.load_data(pool_list, pool_dir, pool_total_regions)
        print('finish loading data.', flush=True)
        return lbl_feat, lbl_list, pool_feat, pool_list

    def _updated_distances(self, cluster_centers: list, features, min_distances):
        x = features[cluster_centers, :]
        dist = pairwise_distances(features, x, metric='euclidean')
        if min_distances is None:
            return np.min(dist, axis=1).reshape(-1, 1)
        else:
            return np.minimum(min_distances, dist)

    def _select_batch(self, features, selected_indices: list, N):
        new_batch = []
        min_distances = self._updated_distances(selected_indices, features, None)
        for _ in range(N):
            ind = np.argmax(min_distances)
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in selected_indices
            min_distances = self._updated_distances([ind], features, min_distances)
            new_batch.append(ind)

        print('Maximum distance from cluster centers is %0.5f' % max(min_distances))
        return new_batch

    def select_next_batch(self, trainer, active_set, selection_count):
        lbl_feat, lbl_list, pool_feat, pool_list = self.calculate_scores(trainer, active_set)
        all_list = np.concatenate((lbl_list, pool_list)) # (total_regions, 4)
        features = np.concatenate((lbl_feat, pool_feat)) # (total_regions, 256)
        print(f'coreset active selection: # of cores:{len(lbl_list)}, # of total:{len(lbl_list) + len(pool_list)}.')
        if trainer.local_rank == 0:
            core_size = len(lbl_list)
            selected_indices = self._select_batch(features, list(range(core_size)), selection_count)
            active_set.expand_training_set([all_list[i] for i in selected_indices])
