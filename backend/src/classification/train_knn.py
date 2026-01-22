
import numpy as np
from sklearn.neighbors import NearestNeighbors

def train_knn(embeddings: np.ndarray) -> NearestNeighbors:
    knn = NearestNeighbors(n_neighbors=3, metric="cosine")
    knn.fit(embeddings)
    return knn