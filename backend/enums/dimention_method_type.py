from enum import Enum

class DimentionMethodType(Enum):
    PCA = 1
    TSNE = 2
    UMAP = 3

    @classmethod
    def from_string(cls, name: str):
        """Convert string (case-insensitive) to DimentionMethodType."""
        normalized = name.strip().lower()
        mapping = {
            "pca": cls.PCA,
            "tsne": cls.TSNE,
            "umap": cls.UMAP
        }
        if normalized not in mapping:
            raise ValueError(f"Unknown DimentionMethodType: {name}")
        return mapping[normalized]