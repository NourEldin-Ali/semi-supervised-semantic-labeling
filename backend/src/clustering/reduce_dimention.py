import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from enums.dimention_method_type import DimentionMethodType


def reduce_dimensionality(embeddings: np.ndarray, method: DimentionMethodType, n_components: int = 2, random_state: int = 42) -> np.ndarray:

    arr = embeddings.astype(np.float32, copy=False)
    if method == DimentionMethodType.PCA:
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == DimentionMethodType.TSNE:
        reducer = TSNE(n_components=n_components, random_state=random_state)
    elif method == DimentionMethodType.UMAP:
        reducer = UMAP(n_components=n_components, metric='cosine', random_state=random_state)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")

    return reducer.fit_transform(arr).astype(np.float32, copy=False)