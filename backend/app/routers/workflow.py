"""
Router for automated workflow endpoints.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pathlib import Path
import pandas as pd

from app.utils.file_utils import save_uploaded_file, read_csv, extract_text_column, create_labeled_csv
from app.utils.execution_stats import run_with_stats
from app.utils.service_utils import (
    generate_embeddings_from_csv,
    generate_clusters,
    generate_group_labels,
    train_knn_model,
)
from enums.embed_type import EmbedType
from enums.llm_type import LLMType
from src.llm.utils.token_usage_handler import TokenUsageCallbackHandler

router = APIRouter()


@router.post("/workflow/full-pipeline")
async def execute_full_pipeline(
    csv_file: UploadFile = File(..., description="CSV file with text data"),
    # Embedding configuration
    embedding_model: str = Form(..., description="Embedding model name"),
    embed_type: str = Form(default="open_ai", description="Embedding type: open_ai, ollama, or huggingface"),
    text_column: str = Form(default="text", description="Name of the text column"),
    batch_size: int = Form(default=32, description="Batch size for embedding generation"),
    embedding_api_key: str = Form(None, description="Optional API key for embedding service"),
    # Clustering configuration
    k: int = Form(None, description="Number of clusters (auto-calculated as number of items - 1 if not provided)"),
    metric: str = Form(default="cosine", description="Distance metric for clustering"),
    # Labeling configuration
    id_column: str = Form(..., description="Name of the ID column in CSV"),
    llm_model: str = Form(default="gpt-5.2-2025-12-11", description="LLM model name"),
    llm_type: str = Form(default="open_ai", description="LLM type: open_ai, groq_ai, ollama, or anthropic"),
    llm_api_key: str = Form(None, description="Optional API key for LLM service"),
    # KNN configuration
    label_column: str = Form(default="labels", description="Name of the label column for KNN training")
):
    """
    Execute the full pipeline: Embeddings → Clustering → Labeling → KNN Training.
    
    Returns both the trained KNN model file path and the labeled CSV file path.
    """
    try:
        # Save and read CSV file
        csv_content = await csv_file.read()
        csv_path = save_uploaded_file(csv_content, csv_file.filename)
        df = read_csv(csv_path)
        
        actual_text_col = (text_column or "text").strip() or "text"
        if actual_text_col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Text column '{actual_text_col}' not found in CSV. Available columns: {list(df.columns)}"
            )
        
        # Validate ID column
        if id_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"ID column '{id_column}' not found in CSV. Available columns: {list(df.columns)}"
            )
        
        try:
            embed_type_enum = EmbedType.from_string(embed_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid embed_type: {str(e)}")
        try:
            llm_type_enum = LLMType.from_string(llm_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid llm_type: {str(e)}")

        if k is None or k == -1:
            k = max(1, len(df) - 1)

        token_handler = TokenUsageCallbackHandler()
        run_config = {"callbacks": [token_handler]}

        def _run():
            from app.utils.file_utils import save_csv, save_model

            embeddings, embedding_tokens = generate_embeddings_from_csv(
                df=df,
                text_column=actual_text_col,
                embedding_model=embedding_model,
                embed_type=embed_type_enum,
                batch_size=batch_size,
                api_key=embedding_api_key,
            )
            clusters = generate_clusters(embeddings=embeddings, k=k, metric=metric)
            item_labels = generate_group_labels(
                clusters=clusters,
                df=df,
                id_column=id_column,
                text_column=actual_text_col,
                llm_model=llm_model,
                llm_type=llm_type_enum,
                api_key=llm_api_key,
                run_config=run_config,
            )
            labeled_df = create_labeled_csv(df, item_labels, id_column, label_column=label_column)
            knn_model, label_data, training_embeddings = train_knn_model(
                embeddings=embeddings,
                labeled_df=labeled_df,
                id_column=id_column,
                label_column=label_column,
            )
            labeled_csv_path = save_csv(labeled_df, f"labeled_{Path(csv_file.filename).stem}.csv")
            model_data = {
                "knn_model": knn_model,
                "label_data": label_data,
                "training_embeddings": training_embeddings,
            }
            model_path = save_model(model_data, f"knn_model_{Path(csv_file.filename).stem}")
            return item_labels, labeled_csv_path, model_path, clusters, label_data, embeddings, embedding_tokens

        (item_labels, labeled_csv_path, model_path, clusters, label_data, embeddings, embedding_tokens), stats = run_with_stats(
            _run, token_handler=token_handler
        )
        labeled_count = sum(1 for labels in item_labels.values() if labels)
        if embedding_tokens > 0:
            stats["tokens_consumed"] = stats.get("tokens_consumed", 0) + embedding_tokens

        return {
            "message": "Full pipeline executed successfully",
            "original_csv_file": str(csv_path),
            "labeled_csv_file": str(labeled_csv_path),
            "knn_model_file": str(model_path),
            "statistics": {
                "total_items": len(df),
                "labeled_items": labeled_count,
                "number_of_clusters": len(clusters),
                "training_samples": len(label_data["labels"]),
                "embedding_shape": list(embeddings.shape),
                **stats,
            },
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing full pipeline: {str(e)}")
