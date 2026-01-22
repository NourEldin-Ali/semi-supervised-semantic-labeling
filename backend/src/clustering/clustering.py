
import numpy as np
from skcmeans.algorithms import Possibilistic
from collections import defaultdict
from kneed import KneeLocator
import numpy as np

def possibilistic_clustering(embeddings: np.ndarray, metric='cosine', k=3):
    clusterer = Possibilistic(
                    n_clusters=k,
                    metric=metric,
                    n_init=100,
                    max_iter=300,
                    random_state=43,
                )
    clusterer.fit(embeddings)
    return clusterer.memberships_


def elbow_threshold_simple(item, sample_index=0):


    # take memberships for a single sample
    vals = item[sample_index].ravel()
    vals_sorted = np.sort(vals)[::-1]

    kl = KneeLocator(
        range(len(vals_sorted)),
        vals_sorted,
        curve='convex',
        direction='decreasing'
    )

    # handle the case where no knee is found
    if kl.knee is None:
        # fallback: e.g. smallest value or some default
        threshold = vals_sorted[-1]
    else:
        threshold = vals_sorted[kl.knee]

    print(f"Elbow-based threshold for sample {sample_index} = {threshold}")
    return threshold

def get_possibilistic_clusters_after_elbow(memberships):
    sample_clusters = []   # will store the cluster sets for each sample

    for i, row in enumerate(memberships):
        threshold = elbow_threshold_simple(memberships, i)
        clusters = tuple(idx for idx, val in enumerate(row) if val >= threshold)
        sample_clusters.append(clusters)
        print(f"Sample {i}: {clusters}") 

    groups = defaultdict(list)

    for sample_idx, clusters in enumerate(sample_clusters):
        for c in clusters:
            groups[c].append(sample_idx)

    # Show result
    for c, samples in sorted(groups.items()):
        print(f"cluster {c} = samples {samples}")

    clusters_res: list[list[int]] = []
    i = 0
    for c, samples in sorted(groups.items()):
        if len(samples)>1:
            i+=1
            clusters_res.append(samples)
            print(f"cluster {c} = samples {samples}")

    return clusters_res