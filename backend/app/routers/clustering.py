"""
Router for clustering endpoints.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from pathlib import Path
import pandas as pd

from app.utils.file_utils import (
    save_uploaded_file, read_csv, save_csv, load_embeddings,
    extract_text_column, create_clustered_csv
)
from app.utils.execution_stats import run_with_stats
from app.utils.service_utils import generate_clusters

router = APIRouter()


@router.post("/clustering/generate")
async def generate_clustering_groups(
    csv_file: UploadFile = File(..., description="Original CSV file"),
    embeddings_file: UploadFile = File(..., description="Embeddings file (.npy)"),
    text_column: str = Form(default="text", description="Name of the text column"),
    k: int = Form(None, description="Number of clusters (auto-calculated as number of items - 1 if not provided)"),
    metric: str = Form(default="cosine", description="Distance metric for clustering")
):
    """
    Endpoint 2: Accept embeddings and original CSV file to generate clustering groups.
    
    Returns a new CSV file containing all cluster groups and associated items.
    The number of clusters (k) is automatically set to (number of items - 1) if not provided.
    """
    try:
        # Save and read CSV file
        csv_content = await csv_file.read()
        csv_path = save_uploaded_file(csv_content, csv_file.filename)
        df = read_csv(csv_path)
        
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Auto-calculate k if not provided or if k is -1: number of items - 1
        if k is None or k == -1:
            k = len(df) - 1
            if k < 1:
                k = 1  # Ensure at least 1 cluster if only 1 item
        
        actual_text_col = (text_column or "text").strip() or "text"
        if actual_text_col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Text column '{actual_text_col}' not found in CSV. Available columns: {list(df.columns)}"
            )
        
        # Validate text column can be extracted
        _ = extract_text_column(df, actual_text_col)
        
        # Save and load embeddings file
        embeddings_content = await embeddings_file.read()
        # Save embeddings temporarily
        import tempfile
        import numpy as np
        import traceback
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_file:
                tmp_file.write(embeddings_content)
                tmp_path = Path(tmp_file.name)
            
            embeddings = load_embeddings(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load embeddings file: {str(e)}")
        finally:
            # Clean up temp file
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
        
        # Validate embeddings shape matches DataFrame
        if len(embeddings) != len(df):
            raise HTTPException(
                status_code=400,
                detail=f"Embeddings shape {len(embeddings)} does not match CSV rows {len(df)}"
            )
        
        # Validate embeddings is 2D
        if len(embeddings.shape) != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 2D embeddings array, got shape {embeddings.shape}"
            )
        
        def _run():
            clusters = generate_clusters(embeddings, k=k, metric=metric)
            if not clusters:
                raise ValueError("No clusters found. Try adjusting clustering parameters (k, m, metric).")
            clustered_df = create_clustered_csv(df, clusters, cluster_column="cluster_group")
            output_filename = f"clustered_{Path(csv_file.filename).stem}.csv"
            output_path = save_csv(clustered_df, output_filename)
            return clusters, output_path

        try:
            (clusters, output_path), stats = run_with_stats(_run)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Clustering algorithm failed: {str(e)}\nTraceback: {traceback.format_exc()}"
            )

        cluster_summary = [
            {"cluster_id": i, "item_count": len(indices), "item_indices": list(indices)}
            for i, indices in enumerate(clusters)
        ]

        return {
            "message": "Clustering completed successfully",
            "output_csv_file": str(output_path),
            "original_csv_file": str(csv_path),
            "number_of_clusters": len(clusters),
            "cluster_summary": cluster_summary,
            **stats,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error generating clusters: {str(e)}\nTraceback: {error_trace}"
        )


@router.get("/clustering/download/{file_path:path}")
async def download_clustered_csv(file_path: str):
    """Download a clustered CSV file."""
    file = Path(file_path)
    if not file.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file, filename=file.name, media_type="text/csv")
