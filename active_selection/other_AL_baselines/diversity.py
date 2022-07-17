# basic
from sklearn.cluster import KMeans


def importance_reweight(scores, features, n_clusters=400, importance_decay=0.95, trim=False):
    # sorted (first time)
    sorted_idx = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    features = features[sorted_idx]
    selected_samples = sorted(scores, reverse=True)
    if trim is True:
        N = features.shape[0] // 10
        features = features[:N]
        selected_samples = selected_samples[:N]
    # clustering
    m = KMeans(n_clusters=n_clusters, random_state=0)
    m.fit(features)
    clusters = m.labels_
    # importance re-weighting
    N = features.shape[0]
    importance_arr = [1 for _ in range(n_clusters)]
    for i in range(N):
        cluster_i = clusters[i]
        cluster_importance = importance_arr[cluster_i]
        scores[i][0] *= cluster_importance
        importance_arr[cluster_i] *= importance_decay
    # sorted (second time)
    selected_samples = sorted(scores, reverse=True)
    return selected_samples
