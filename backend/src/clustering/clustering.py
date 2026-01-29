
import numpy as np
from skcmeans.algorithms import Possibilistic
import inspect
from collections import defaultdict
from kneed import KneeLocator
import numpy as np


def _get_memberships(clusterer: Possibilistic) -> np.ndarray:
    """Return memberships in a version-tolerant way."""
    return getattr(clusterer, "memberships", getattr(clusterer, "memberships_", None))


def possibilistic_clustering(embeddings: np.ndarray, metric: str = "cosine", k: int = 3, m: float = 1.0) -> np.ndarray:
    init_kwargs = {
        "n_clusters": k,
        "metric": metric,
        "n_init": 100,
        "max_iter": 300,
        "random_state": 43,
    }
    # Some versions of skcmeans do not accept "m" for Possibilistic.
    try:
        if "m" in inspect.signature(Possibilistic.__init__).parameters:
            init_kwargs["m"] = m
    except (TypeError, ValueError):
        # Fallback if signature inspection fails; try without "m".
        pass

    clusterer = Possibilistic(**init_kwargs)
    clusterer.fit(embeddings)
    memberships = _get_memberships(clusterer)
    if memberships is None:
        raise AttributeError("Possibilistic clusterer did not expose memberships or memberships_.")
    return memberships


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
        threshold = vals_sorted[1]
    else:
        threshold = vals_sorted[kl.knee]

    print(f"Elbow-based threshold for sample {sample_index} = {threshold}")
    return threshold

def _dedupe_clusters(clusters: list[list[int]]) -> list[list[int]]:
    """Remove duplicate clusters based on identical item sets (preserve first occurrence)."""
    seen: set[tuple[int, ...]] = set()
    deduped: list[list[int]] = []
    for cluster in clusters:
        key = tuple(sorted(cluster))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cluster)
    return deduped


def get_possibilistic_clusters_after_elbow(memberships: np.ndarray) -> list[list[int]]:
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

    # Show result (including singletons)
    for c, samples in sorted(groups.items()):
        print(f"cluster {c} = samples {samples}")

    # Preserve original cluster IDs (0..k-1) and include singletons.
    n_clusters = memberships.shape[1]
    clusters_res: list[list[int]] = [[] for _ in range(n_clusters)]
    for c in range(n_clusters):
        clusters_res[c] = groups.get(c, [])

    # Drop singletons and duplicate clusters (same items).
    clusters_res = [c for c in clusters_res if len(c) > 1]
    clusters_res = _dedupe_clusters(clusters_res)

    return clusters_res
