"""
Utility functions for file handling and CSV processing.
"""
import os
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import uuid


# Base directory for storing files
UPLOAD_DIR = Path("uploads")
EMBEDDING_DIR = Path("embeddings")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

# Create directories if they don't exist
for dir_path in [UPLOAD_DIR, EMBEDDING_DIR, MODEL_DIR, OUTPUT_DIR]:
    dir_path.mkdir(exist_ok=True)


def save_uploaded_file(file_content: bytes, filename: str) -> Path:
    """Save an uploaded file to the uploads directory."""
    file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{filename}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    return file_path


def read_csv(file_path: Path) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame.

    Uses pandas defaults: comma delimiter, double-quote for fields containing commas.
    E.g. "Is there a formal, approved information..." is correctly read as one column.
    """
    try:
        df = pd.read_csv(file_path, sep=',')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, sep=',', encoding="utf-8")
    return df


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    """Save a DataFrame to CSV in the outputs directory."""
    output_path = OUTPUT_DIR / f"{uuid.uuid4()}_{filename}"
    df.to_csv(output_path, index=False)
    return output_path


def save_embeddings(embeddings: np.ndarray, filename: str) -> Path:
    """Save embeddings array to a .npy file."""
    embedding_path = EMBEDDING_DIR / f"{uuid.uuid4()}_{filename}.npy"
    np.save(embedding_path, embeddings)
    return embedding_path


def load_embeddings(file_path: Path) -> np.ndarray:
    """Load embeddings from a .npy file."""
    return np.load(file_path)


def save_model(model: Any, filename: str) -> Path:
    """Save a trained model using joblib."""
    model_path = MODEL_DIR / f"{uuid.uuid4()}_{filename}.joblib"
    joblib.dump(model, model_path)
    return model_path


def load_model(file_path: Path) -> Any:
    """Load a trained model from a joblib file."""
    return joblib.load(file_path)


def extract_text_column(df: pd.DataFrame, column: Optional[str] = None) -> List[str]:
    """
    Extract text column from DataFrame.
    If column is None, tries to infer the text column (first string column).
    """
    if column is None:
        # Try to find the first string column
        for col in df.columns:
            if df[col].dtype == 'object':
                return df[col].tolist()
        raise ValueError("No text column found in CSV. Please specify a column name.")
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in CSV. Available columns: {list(df.columns)}")
    
    return df[column].tolist()


def create_clustered_csv(df: pd.DataFrame, clusters: List[List[int]], cluster_column: str = "cluster_group") -> pd.DataFrame:
    """
    Create a new CSV with cluster groups assigned to each row.
    clusters: list of lists, where each list contains indices of items in that cluster.
    Each item can belong to multiple clusters (comma-separated cluster IDs).
    """
    # Initialize cluster column as empty strings
    result_df = df.copy()
    result_df[cluster_column] = ""
    
    # Build a mapping of item index to list of cluster IDs
    item_clusters = {}
    for cluster_id, indices in enumerate(clusters):
        for idx in indices:
            if idx < len(result_df):
                if idx not in item_clusters:
                    item_clusters[idx] = []
                item_clusters[idx].append(cluster_id)
    
    # Assign cluster IDs (comma-separated if multiple)
    for idx, cluster_ids in item_clusters.items():
        result_df.at[idx, cluster_column] = ", ".join(map(str, sorted(cluster_ids)))
    
    # Items not in any cluster remain empty (could set to -1 if preferred)
    result_df[cluster_column] = result_df[cluster_column].replace("", "-1")
    
    return result_df


def extract_clusters_from_csv(
    clustered_df: pd.DataFrame, 
    original_df: pd.DataFrame,
    cluster_column: str = "cluster_group",
    id_column: str = "id"
) -> List[List[int]]:
    """
    Extract cluster groups from a clustered CSV file.
    
    Args:
        clustered_df: DataFrame with cluster_group column
        original_df: Original DataFrame to map indices correctly
        cluster_column: Name of the column containing cluster IDs
        id_column: Name of the ID column for mapping
    
    Returns:
        List of lists, where each inner list contains the indices (from original_df) 
        of items belonging to that cluster
    """
    from collections import defaultdict
    
    # Create a mapping from ID to index in original_df
    id_to_index = {}
    for idx, row in original_df.iterrows():
        id_to_index[str(row[id_column])] = int(idx)
    
    # Build clusters: cluster_id -> list of indices in original_df
    clusters_dict = defaultdict(list)
    
    for idx, row in clustered_df.iterrows():
        cluster_groups_str = str(row[cluster_column])
        
        # Skip if no cluster assigned or is -1
        if not cluster_groups_str or cluster_groups_str.strip() == "" or cluster_groups_str == "-1":
            continue
        
        # Parse comma-separated cluster IDs
        cluster_ids = [int(cid.strip()) for cid in cluster_groups_str.split(",") if cid.strip() and cid.strip() != "-1"]
        
        # Get the item ID from clustered CSV
        item_id = str(row[id_column])
        
        # Find the index in original_df
        if item_id in id_to_index:
            original_idx = id_to_index[item_id]
            # Add this item to each cluster it belongs to
            for cluster_id in cluster_ids:
                if original_idx not in clusters_dict[cluster_id]:
                    clusters_dict[cluster_id].append(original_idx)
    
    # Convert to list of lists, sorted by cluster ID
    clusters = [clusters_dict[cluster_id] for cluster_id in sorted(clusters_dict.keys())]
    
    return clusters


def create_labeled_csv(df: pd.DataFrame, labels: Dict[str, List[str]], 
                      id_column: str, label_column: str = "labels") -> pd.DataFrame:
    """
    Add labels to DataFrame based on ID column.
    labels: Dict mapping item ID to list of labels
    """
    result_df = df.copy()
    
    # Create labels column (join list of labels as comma-separated string)
    result_df[label_column] = result_df[id_column].apply(
        lambda x: ", ".join(labels.get(str(x), [])) if str(x) in labels else ""
    )
    
    return result_df
