

import numpy as np
from collections import Counter


def predict_knn_labels(knn_model, sample_embeddings: np.ndarray, data, k: int = 3):
    distances, indices = knn_model.kneighbors(sample_embeddings)
    neighbor_labels = [data['labels'][i] for i in indices[0]]
    majority_label, ranked_labels = _rank_labels(neighbor_labels, k, top_n=None, weight_by_rank=False)
    return ranked_labels

def _rank_labels(neighbor_labels, k: int, top_n: int | None, weight_by_rank: bool = True):
    """
    Given labels for each neighbor (ordered closest -> farthest), return:
    - majority_label: single best label
    - ranked_labels: list of (label, weight) including ties within the top_n weight tiers
    weight_by_rank=True -> weight = k - rank; False -> weight = 1 per neighbor label.
    Labels with total weight <= 1 are dropped.
    """
    label_weights = Counter()
    for rank, doc_labels in enumerate(neighbor_labels):
        weight = (k - rank) if weight_by_rank else 1
        if isinstance(doc_labels, (list, tuple)):
            iter_labels = doc_labels
        else:
            iter_labels = [doc_labels]
        for lbl in iter_labels:
            if lbl:
                label_weights[lbl] += weight

    # Drop labels whose total weight is <= 1
    label_weights = Counter({l: w for l, w in label_weights.items() if w > 1})
    if not label_weights:
        return None, []

    ranked_all = label_weights.most_common()
    if top_n:
        distinct_weights = []
        for _, w in ranked_all:
            if not distinct_weights or distinct_weights[-1] != w:
                distinct_weights.append(w)
            if len(distinct_weights) == top_n:
                threshold_weight = distinct_weights[-1]
                break
        else:
            threshold_weight = ranked_all[-1][1]
        ranked_labels = [(l, w) for l, w in ranked_all if w >= threshold_weight]
    else:
        ranked_labels = ranked_all

    return ranked_labels[0][0], ranked_labels