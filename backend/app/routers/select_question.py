"""
Router for question selection endpoints (BM25 + embeddings).
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app.utils.file_utils import save_uploaded_file, read_csv
from app.utils.execution_stats import run_with_stats
from app.utils.service_utils import (
    select_questions_bm25_service,
    select_questions_embedding_service,
    select_questions_label_embedding_service,
)

router = APIRouter()


@router.post("/select-question/bm25")
async def select_questions_bm25_endpoint(
    file: UploadFile = File(..., description="CSV file containing questions"),
    user_need: str = Form(..., description="User requirement / need text"),
    text_column: str = Form(default="text", description="Name of the question column"),
    id_column: str = Form(default="id", description="Name of the ID column"),
    label_column: str | None = Form(default=None, description="Optional label column"),
    top_k: int = Form(default=5, description="Number of top questions to return"),
):
    """
    Select top questions using BM25.
    """
    try:
        file_content = await file.read()
        csv_path = save_uploaded_file(file_content, file.filename)
        df = read_csv(csv_path)

        actual_text_col = (text_column or "text").strip() or "text"
        actual_id_col = (id_column or "id").strip() or "id"
        actual_label_col = (label_column or "").strip() or None

        def _run():
            return select_questions_bm25_service(
                df=df,
                user_need=user_need,
                text_column=actual_text_col,
                id_column=actual_id_col,
                label_column=actual_label_col,
                top_k=top_k,
            )

        results, stats = run_with_stats(_run)
        return {
            "message": "BM25 selection completed successfully",
            "method": "bm25",
            "results": results,
            "total_results": len(results),
            **stats,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error selecting questions: {str(e)}")


@router.post("/select-question/embedding")
async def select_questions_embedding_endpoint(
    file: UploadFile = File(..., description="CSV file containing questions"),
    user_need: str = Form(..., description="User requirement / need text"),
    embedding_model: str = Form(..., description="Embedding model name"),
    embed_type: str = Form(default="open_ai", description="Embedding type: open_ai, ollama, or huggingface"),
    text_column: str = Form(default="text", description="Name of the question column"),
    id_column: str = Form(default="id", description="Name of the ID column"),
    label_column: str | None = Form(default=None, description="Optional label column"),
    batch_size: int = Form(default=32, description="Batch size for embedding generation"),
    top_k: int = Form(default=5, description="Number of top questions to return"),
    api_key: str = Form(default=None, description="Optional API key for embedding service"),
    endpoint: str = Form(default=None, description="Optional embedding endpoint (e.g., Ollama base URL)"),
):
    """
    Select top questions using embedding similarity.
    """
    try:
        file_content = await file.read()
        csv_path = save_uploaded_file(file_content, file.filename)
        df = read_csv(csv_path)

        actual_text_col = (text_column or "text").strip() or "text"
        actual_id_col = (id_column or "id").strip() or "id"
        actual_label_col = (label_column or "").strip() or None

        def _run():
            return select_questions_embedding_service(
                df=df,
                user_need=user_need,
                embedding_model=embedding_model,
                embed_type=embed_type,
                api_key=api_key,
                endpoint=endpoint,
                text_column=actual_text_col,
                id_column=actual_id_col,
                label_column=actual_label_col,
                batch_size=batch_size,
                top_k=top_k,
            )

        results, stats = run_with_stats(_run)
        return {
            "message": "Embedding-based selection completed successfully",
            "method": "embedding",
            "results": results,
            "total_results": len(results),
            **stats,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error selecting questions: {str(e)}")


@router.post("/select-question/label-embedding")
async def select_questions_label_embedding_endpoint(
    file: UploadFile = File(..., description="CSV file containing questions and labels"),
    user_need: str = Form(..., description="User requirement / need text"),
    embedding_model: str = Form(..., description="Embedding model name"),
    embed_type: str = Form(default="open_ai", description="Embedding type: open_ai, ollama, or huggingface"),
    text_column: str = Form(default="text", description="Name of the question column"),
    id_column: str = Form(default="id", description="Name of the ID column"),
    label_column: str = Form(default="labels", description="Name of the labels column"),
    batch_size: int = Form(default=32, description="Batch size for embedding generation"),
    top_k_labels: int = Form(default=5, description="Number of top labels to keep"),
    top_k_questions: int = Form(default=5, description="Number of top questions to return"),
    api_key: str = Form(default=None, description="Optional API key for embedding service"),
    endpoint: str = Form(default=None, description="Optional embedding endpoint (e.g., Ollama base URL)"),
):
    """
    Select top questions by embedding labels first, then matching questions.
    """
    try:
        file_content = await file.read()
        csv_path = save_uploaded_file(file_content, file.filename)
        df = read_csv(csv_path)

        actual_text_col = (text_column or "text").strip() or "text"
        actual_id_col = (id_column or "id").strip() or "id"
        actual_label_col = (label_column or "labels").strip() or "labels"

        def _run():
            return select_questions_label_embedding_service(
                df=df,
                user_need=user_need,
                embedding_model=embedding_model,
                embed_type=embed_type,
                api_key=api_key,
                endpoint=endpoint,
                text_column=actual_text_col,
                id_column=actual_id_col,
                label_column=actual_label_col,
                batch_size=batch_size,
                top_k_labels=top_k_labels,
                top_k_questions=top_k_questions,
            )

        results, stats = run_with_stats(_run)
        return {
            "message": "Label-embedding selection completed successfully",
            "method": "label_embedding",
            "results": results,
            "total_results": len(results),
            **stats,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error selecting questions: {str(e)}")
