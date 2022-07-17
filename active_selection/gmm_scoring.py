import os
import sys
import pickle
import numpy as np
import multiprocessing as mp
from sklearn.mixture import GaussianMixture


root_dir = sys.argv[1]
dataset = sys.argv[2]


def GMM_scoring(target_label):
    print(f"------ I am class {target_label}", flush=True)
    cls_feat = np.load(os.path.join(root_dir, f"feat_{target_label:02d}.npz"))
    src_cls_feat = cls_feat['src']
    trg_cls_feat = cls_feat['trg']
    total_num = trg_cls_feat.shape[0]
    # GMM hyper-parameter: (1) Proportional to #regions (2) clipped to 1~10
    n_components = np.clip(total_num // 100, a_min=1, a_max=10)
    # Construct Gaussian Mixture Model for source feature and target feature respectively
    print(total_num, flush=True)
    if total_num > 1:
        src_gm = GaussianMixture(n_components=n_components, random_state=0, verbose=0,
                                 n_init=1, max_iter=200).fit(src_cls_feat)
        trg_gm = GaussianMixture(n_components=n_components, random_state=0, verbose=0,
                                 n_init=1, max_iter=200).fit(trg_cls_feat)

        src_gm_fname = os.path.join(root_dir, f"GMM_src_{target_label:02d}.pkl")
        trg_gm_fname = os.path.join(root_dir, f"GMM_trg_{target_label:02d}.pkl")
        with open(src_gm_fname, "wb") as f:
            pickle.dump(src_gm, f)
        with open(trg_gm_fname, "wb") as f:
            pickle.dump(trg_gm, f)


adapt_classes = [x for x in range(19)]
if dataset == "SYNTHIA":
    class_16_ignore = [9, 14, 16]
    for x in class_16_ignore:
        adapt_classes.remove(x)
with mp.Pool(5) as p:
    p.map(GMM_scoring, adapt_classes)
