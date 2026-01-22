"""
Router for labeling endpoints.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from pathlib import Path
import pandas as pd
import json

from app.utils.file_utils import (
    save_uploaded_file, read_csv, save_csv,
    extract_text_column, create_labeled_csv, extract_clusters_from_csv
)
from app.utils.execution_stats import run_with_stats
from app.utils.service_utils import generate_group_labels, generate_single_item_labels
from enums.llm_type import LLMType
from src.llm.utils.token_usage_handler import TokenUsageCallbackHandler

router = APIRouter()


@router.post("/labeling/generate-group-labels")
async def generate_labels_for_groups(
    clustered_csv_file: UploadFile = File(..., description="CSV file containing clustered data (with cluster_group column)"),
    original_csv_file: UploadFile = File(..., description="Original CSV file with all data"),
    id_column: str = Form(..., description="Name of the ID column in both CSV files"),
    cluster_column: str = Form(default="cluster_group", description="Name of the cluster column in clustered CSV"),
    text_column: str = Form(None, description="Name of the text column (auto-detected if not provided)"),
    llm_model: str = Form(default="gpt-5.2-2025-12-11", description="LLM model name"),
    llm_type: str = Form(default="open_ai", description="LLM type: open_ai, groq_ai, ollama, or anthropic"),
    api_key: str = Form(None, description="Optional API key for LLM service")
):
    """
    Endpoint 3: Accept clustered CSV and original CSV to generate labels for each group using LLM.
    Automatically extracts cluster information from the clustered CSV file.
    Assigns labels to each individual item based on their cluster group(s).
    Items can belong to multiple clusters.
    """
    try:
        # Save and read both CSV files
        clustered_content = await clustered_csv_file.read()
        clustered_path = save_uploaded_file(clustered_content, clustered_csv_file.filename)
        clustered_df = read_csv(clustered_path)
        
        original_content = await original_csv_file.read()
        original_path = save_uploaded_file(original_content, original_csv_file.filename)
        original_df = read_csv(original_path)
        
        # Validate ID column exists in both files
        if id_column not in clustered_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"ID column '{id_column}' not found in clustered CSV. Available columns: {list(clustered_df.columns)}"
            )
        
        if id_column not in original_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"ID column '{id_column}' not found in original CSV. Available columns: {list(original_df.columns)}"
            )
        
        # Validate cluster column exists in clustered CSV
        if cluster_column not in clustered_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Cluster column '{cluster_column}' not found in clustered CSV. Available columns: {list(clustered_df.columns)}"
            )
        
        actual_text_col = (text_column or "text").strip() or "text"
        if actual_text_col not in original_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Text column '{actual_text_col}' not found in original CSV. Available columns: {list(original_df.columns)}"
            )
        
        # Validate text column can be extracted
        _ = extract_text_column(original_df, actual_text_col)
        
        # Extract clusters from clustered CSV
        clusters = extract_clusters_from_csv(
            clustered_df=clustered_df,
            original_df=original_df,
            cluster_column=cluster_column,
            id_column=id_column
        )
        
        if not clusters:
            raise HTTPException(
                status_code=400,
                detail="No clusters found in clustered CSV. Please ensure the clustered CSV contains valid cluster assignments."
            )
        
        try:
            llm_type_enum = LLMType.from_string(llm_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        token_handler = TokenUsageCallbackHandler()
        run_config = {"callbacks": [token_handler]}

        def _run():
            item_labels = generate_group_labels(
                clusters=clusters,
                df=original_df,
                id_column=id_column,
                text_column=actual_text_col,
                llm_model=llm_model,
                llm_type=llm_type_enum,
                api_key=api_key,
                run_config=run_config,
            )
            labeled_df = create_labeled_csv(original_df, item_labels, id_column, label_column="labels")
            output_filename = f"labeled_{Path(original_csv_file.filename).stem}.csv"
            output_path = save_csv(labeled_df, output_filename)
            return item_labels, output_path

        (item_labels, output_path), stats = run_with_stats(_run, token_handler=token_handler)
        labeled_count = sum(1 for labels in item_labels.values() if labels)

        return {
            "message": "Labels generated successfully",
            "output_csv_file": str(output_path),
            "original_csv_file": str(original_path),
            "clustered_csv_file": str(clustered_path),
            "labeled_items_count": labeled_count,
            "total_items": len(original_df),
            "number_of_clusters": len(clusters),
            "labels_generated": len(item_labels),
            **stats,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating labels: {str(e)}")


@router.post("/labeling/generate-item-labels")
async def generate_labels_for_items(
    csv_file: UploadFile = File(..., description="CSV file containing items to label"),
    id_column: str = Form(..., description="Name of the ID column in CSV"),
    text_column: str = Form(default="text", description="Name of the text column"),
    llm_model: str = Form(default="gpt-5.2-2025-12-11", description="LLM model name"),
    llm_type: str = Form(default="open_ai", description="LLM type: open_ai, groq_ai, ollama, or anthropic"),
    api_key: str = Form(None, description="Optional API key for LLM service")
):
    """
    Endpoint 6: Accept a CSV file and generate labels for each item.
    Returns a CSV file with items and their labels.
    """
    try:
        # Save and read CSV file
        csv_content = await csv_file.read()
        csv_path = save_uploaded_file(csv_content, csv_file.filename)
        df = read_csv(csv_path)
        
        # Validate ID column exists
        if id_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"ID column '{id_column}' not found in CSV. Available columns: {list(df.columns)}"
            )
        
        actual_text_col = (text_column or "text").strip() or "text"
        if actual_text_col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Text column '{actual_text_col}' not found in CSV. Available columns: {list(df.columns)}"
            )
        
        _ = extract_text_column(df, actual_text_col)
        
        try:
            llm_type_enum = LLMType.from_string(llm_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        token_handler = TokenUsageCallbackHandler()
        run_config = {"callbacks": [token_handler]}

        def _run():
            item_labels = generate_single_item_labels(
                df=df,
                id_column=id_column,
                text_column=actual_text_col,
                llm_model=llm_model,
                llm_type=llm_type_enum,
                api_key=api_key,
                run_config=run_config,
            )
            labeled_df = create_labeled_csv(df, item_labels, id_column, label_column="labels")
            output_filename = f"labeled_{Path(csv_file.filename).stem}.csv"
            output_path = save_csv(labeled_df, output_filename)
            return item_labels, output_path

        (item_labels, output_path), stats = run_with_stats(_run, token_handler=token_handler)
        labeled_count = sum(1 for labels in item_labels.values() if labels)

        return {
            "message": "Labels generated successfully",
            "output_csv_file": str(output_path),
            "original_csv_file": str(csv_path),
            "labeled_items_count": labeled_count,
            "total_items": len(df),
            **stats,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating labels: {str(e)}")


@router.get("/labeling/download/{file_path:path}")
async def download_labeled_csv(file_path: str):
    """Download a labeled CSV file."""
    file = Path(file_path)
    if not file.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file, filename=file.name, media_type="text/csv")
