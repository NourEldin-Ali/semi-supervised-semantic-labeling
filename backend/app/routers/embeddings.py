"""
Router for embedding generation endpoints.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from pathlib import Path
import pandas as pd

from app.utils.file_utils import save_uploaded_file, read_csv, save_embeddings, extract_text_column
from app.utils.execution_stats import run_with_stats
from app.utils.service_utils import generate_embeddings_from_csv
from enums.embed_type import EmbedType

router = APIRouter()


@router.post("/embeddings/generate")
async def generate_embeddings(
    file: UploadFile = File(..., description="CSV file to process"),
    embedding_model: str = Form(..., description="Embedding model name (e.g., 'text-embedding-3-large')"),
    embed_type: str = Form(default="open_ai", description="Embedding type: open_ai, ollama, or huggingface"),
    text_column: str = Form(default="text", description="Name of the text column"),
    batch_size: int = Form(default=32, description="Batch size for embedding generation"),
    api_key: str = Form(None, description="Optional API key for embedding service")
):
    """
    Endpoint 1: Accept a CSV file and generate embedding files based on selected embedding model.
    
    Returns:
    - Path to the saved embeddings file (.npy)
    - Path to the original CSV file
    """
    try:
        # Save uploaded file
        file_content = await file.read()
        csv_path = save_uploaded_file(file_content, file.filename)
        
        # Read CSV
        df = read_csv(csv_path)
        
        actual_text_col = (text_column or "text").strip() or "text"
        if actual_text_col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Text column '{actual_text_col}' not found in CSV. Available columns: {list(df.columns)}"
            )
        
        # Extract text values for validation
        _ = extract_text_column(df, actual_text_col)
        
        # Parse embed type
        try:
            embed_type_enum = EmbedType.from_string(embed_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        def _run():
            embeddings, embedding_tokens = generate_embeddings_from_csv(
                df=df,
                text_column=actual_text_col,
                embedding_model=embedding_model,
                embed_type=embed_type_enum,
                batch_size=batch_size,
                api_key=api_key,
            )
            embedding_filename = f"embeddings_{Path(file.filename).stem}"
            embedding_path = save_embeddings(embeddings, embedding_filename)
            return embeddings, embedding_path, embedding_tokens

        (embeddings, embedding_path, embedding_tokens), stats = run_with_stats(_run)
        if embedding_tokens > 0:
            stats["tokens_consumed"] = embedding_tokens

        return {
            "message": "Embeddings generated successfully",
            "embeddings_file": str(embedding_path),
            "csv_file": str(csv_path),
            "embedding_shape": list(embeddings.shape),
            "text_column": actual_text_col,
            **stats,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")


@router.get("/embeddings/download/{file_path:path}")
async def download_embeddings(file_path: str):
    """Download an embeddings file."""
    file = Path(file_path)
    if not file.exists():
        raise HTTPException(status_code=404, detail="Embeddings file not found")
    return FileResponse(file, filename=file.name)
