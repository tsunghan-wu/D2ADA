import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataloader.utils import collate_fn
from torch_scatter import scatter_mean
from .utils import row_norms, kmeans_plus_plus_opt


class BADGESelector:
    def __init__(self, args, batch_size, num_workers):
        self.args = args
        self.batch_size = batch_size
        self.num_workers = num_workers

    def feature_extractor(self, model, dataset, device, save_dir):
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

    def select_next_batch(self, trainer, active_set, selection_count):
        if trainer.local_rank == 0:
            os.makedirs(self.args.save_feat_dir, exist_ok=True)
            device = torch.device('cuda:0')
            # get model
            model = trainer.net
            # inference on both source and target domain data
            trg_dir = os.path.join(self.args.save_feat_dir, "target")
            os.makedirs(trg_dir, exist_ok=True)
            trg_set = active_set.trg_pool_dataset
            trg_total_regions = self.feature_extractor(model, trg_set, device, trg_dir)
            # trg_total_regions = 119000
            # Load Feature and probability for sample selection
            trg_spx_list = active_set.trg_pool_dataset.im_idx
            trg_feat, trg_prob, trg_spx_list = self.load_data(trg_spx_list, trg_dir, trg_total_regions)

            # Compute uncertainty gradient
            trg_preds = trg_prob.argmax(1)
            trg_scores_delta = np.zeros_like(trg_prob)
            trg_scores_delta[np.arange(len(trg_scores_delta)), trg_preds] = 1

            # Uncertainty embedding
            badge_uncertainty = (trg_prob-trg_scores_delta)

            # Seed with maximum uncertainty example
            max_norm = row_norms(badge_uncertainty).argmax()

            N = selection_count * 40
            _, selected_idxs = kmeans_plus_plus_opt(badge_uncertainty, trg_feat, N, init=[max_norm])
            # selected idxs to selected samples
            selected_samples = []
            for idx in selected_idxs:
                rgb_fname, lbl_fname, spx_fname, suppix_idx = trg_spx_list[idx]
                suppix_idx = int(suppix_idx)   # string to integer (important!)
                if [rgb_fname, lbl_fname, spx_fname] in active_set.trg_pool_dataset.im_idx:
                    if suppix_idx in active_set.trg_pool_dataset.suppix[spx_fname]:
                        key = ','.join([rgb_fname, lbl_fname, spx_fname])
                        item = (0, key, suppix_idx)   # score, key, suppix_idx
                        # print(f'{rgb_fname},{lbl_fname},{spx_fname},{suppix_idx}', file=f_selection)
                        selected_samples.append(item)
            active_set.expand_training_set(selected_samples, selection_count)
