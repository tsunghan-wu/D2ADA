import os
import numpy as np
import torch
import pickle
import torch.nn.functional as F
from tqdm import tqdm
from dataloader import get_dataset
from dataloader.utils import collate_fn
from active_selection.utils import weighted_roundrobin
from torch_scatter import scatter_mean
import subprocess


class RegionDensitySelector:
    def __init__(self, args, batch_size, num_workers):
        self.args = args
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_balance_selection = True

    def feature_extractor(self, model, dataset, device, save_dir, save_prob=False):
        dataset.return_spx = True
        dataset_loader = \
            torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size,
                                        shuffle=False, collate_fn=collate_fn,
                                        num_workers=self.num_workers, pin_memory=True, drop_last=False)
        model.eval()
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
                    point_feat = feats[batch_idx]
                    feat_size = point_feat.size(0)
                    # region feature extraction (Z)
                    point_feat = point_feat.view(feat_size, -1)
                    point_feat = point_feat.permute(1, 0)  # transpose --> (points, feat)
                    region_feat = scatter_mean(point_feat, suppix, dim=0).cpu().numpy()   # 40x256

                    # region category extraction (C) --> Implementation: Get Predicted Probability
                    point_prob = probs[batch_idx].view(19, -1)
                    point_prob = point_prob.permute(1, 0)  # transpose --> (points, feat)
                    region_prob = scatter_mean(point_prob, suppix, dim=0).cpu().numpy()   # 40x19

                    # save result
                    basename = batch['fnames'][batch_idx][0].split('/')[-1].split('.')[0]
                    fname = os.path.join(save_dir, f"{basename}.npz")
                    if save_prob is True:
                        image_prob = image_probs[batch_idx]
                        np.savez(fname, image_prob=image_prob, feat=region_feat,
                                 prob=region_prob)
                    else:
                        np.savez(fname, feat=region_feat, prob=region_prob)
                    del region_feat
                    del region_prob

    def load_data(self, data_fnames, feat_dir):
        feat_dim = 256
        n_class = 19
        num_spx = 40
        n_data = len(data_fnames)
        feats = np.zeros((n_data*num_spx, feat_dim), dtype=np.float32)
        probs = np.zeros((n_data*num_spx, n_class), dtype=np.float32)
        spx_infos = np.empty((n_data*num_spx, 4), dtype=np.object)
        cur_idx = 0
        for idx, fname in enumerate(tqdm(data_fnames)):
            basename = fname[0].split("/")[-1].split(".")[0]
            abs_fname = os.path.join(feat_dir, f'{basename}.npz')
            arr = np.load(abs_fname)
            feat = arr['feat']
            prob = arr['prob']
            spx_fname = np.repeat([fname], num_spx, axis=0)
            spx_info = np.hstack([spx_fname, np.arange(num_spx).reshape(-1, 1)])
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
            trg_set = get_dataset(self.args.trg_dataset, data_root=self.args.trg_data_dir,
                                  datalist=self.args.trg_datalist)

            src_dir = os.path.join(self.args.save_feat_dir, "source")
            os.makedirs(src_dir, exist_ok=True)
            src_set = get_dataset(self.args.src_dataset, data_root=self.args.src_data_dir,
                                  datalist=self.args.src_datalist)
            # infer the model on D_S and D_T and save the information
            self.feature_extractor(model, src_set, device, src_dir)
            self.feature_extractor(model, trg_set, device, trg_dir, save_prob=False)

            # Load Feature and probability for sample selection
            src_spx_list = active_set.src_label_dataset.im_idx
            trg_spx_list = active_set.trg_pool_dataset.im_idx
            if len(active_set.trg_label_dataset.im_idx) != 0:
                trg_spx_list = np.concatenate((trg_spx_list, active_set.trg_label_dataset.im_idx))
                trg_spx_list = np.unique(trg_spx_list, axis=0)
            src_feat, src_prob, src_spx_list = self.load_data(src_spx_list, src_dir)
            trg_feat, trg_prob, trg_spx_list = self.load_data(trg_spx_list, trg_dir)
            src_pseudo_label = np.argmax(src_prob, axis=1)
            trg_pseudo_label = np.argmax(trg_prob, axis=1)

            # GMM
            # load the preprocessed file (a temporary solution)
            selected_samples = []
            GMM_dir = os.path.join(self.args.save_feat_dir, "GMM")
            os.makedirs(GMM_dir, exist_ok=True)

            adapt_classes = [x for x in range(19)]
            if self.args.src_dataset == "SYNTHIA":
                class_16_ignore = [9, 14, 16]
                for x in class_16_ignore:
                    adapt_classes.remove(x)
            for target_label in adapt_classes:
                src_cls_feat = src_feat[src_pseudo_label == target_label]
                trg_cls_feat = trg_feat[trg_pseudo_label == target_label]
                fname = os.path.join(GMM_dir, f"feat_{target_label:02d}.npz")
                np.savez(fname, src=src_cls_feat, trg=trg_cls_feat)
            # Construct GMMs for density estimation
            subprocess.call(f"python3 active_selection/gmm_scoring.py {GMM_dir} {self.args.src_dataset}", shell=True)
            print("GMM done", flush=True)
            class_weights = []
            for target_label in adapt_classes:
                src_cls_feat = src_feat[src_pseudo_label == target_label]
                trg_cls_feat = trg_feat[trg_pseudo_label == target_label]
                if trg_cls_feat.shape[0] < 2:   # GMM minimum of 2 samples is required
                    continue
                # Load constructed GMMs
                with open(os.path.join(GMM_dir, f"GMM_src_{target_label:02d}.pkl"), "rb") as f:
                    src_gm = pickle.load(f)
                with open(os.path.join(GMM_dir, f"GMM_trg_{target_label:02d}.pkl"), "rb") as f:
                    trg_gm = pickle.load(f)
                # Density Query --> log(d_T) - log(d_S) = log(d_T/d_S)
                src_gmm_prob = src_gm.score_samples(trg_cls_feat)  # sum [log (P(SRC|X))]
                trg_gmm_prob = trg_gm.score_samples(trg_cls_feat)  # sum [log (P(TRG|X))]
                diff = src_gmm_prob - trg_gmm_prob   # np.argsort --> the more negative, the more likely to be selected
                # Class balance selection (use log10 as a naive normalization function)
                if self.class_balance_selection:
                    kl_div = np.abs(diff.mean())
                    class_weights.append(int(np.log10(kl_div)))
                else:
                    class_weights.append(1)

                sorted_idx = np.argsort(diff)
                trg_cls_spx = trg_spx_list[trg_pseudo_label == target_label]
                # consider each class individually
                each_class_selection_budget = min(trg_cls_spx.shape[0], 200)
                each_class_selected_samples = []
                for i in range(each_class_selection_budget):
                    idx = sorted_idx[i]
                    rgb_fname, lbl_fname, spx_fname, suppix_idx = trg_cls_spx[idx]
                    suppix_idx = int(suppix_idx)   # string to integer
                    if [rgb_fname, lbl_fname, spx_fname] in active_set.trg_pool_dataset.im_idx:
                        if suppix_idx in active_set.trg_pool_dataset.suppix[spx_fname]:
                            key = ','.join([rgb_fname, lbl_fname, spx_fname])
                            item = (0, key, suppix_idx)   # score, key, suppix_idx
                            each_class_selected_samples.append(item)
                selected_samples.append(each_class_selected_samples)
            print(f"Class Weights: {class_weights}", flush=True)
            # Apply Weighted Round-Robin Algorithm for weighted class selection
            selected_samples = weighted_roundrobin(selected_samples, class_weights)
            active_set.expand_training_set(selected_samples, selection_count, "density")
