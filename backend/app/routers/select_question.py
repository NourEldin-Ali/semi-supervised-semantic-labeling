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
    score_question_with_llm,
    compare_questions_with_llm,
    score_question_set_with_llm,
    compare_question_sets_with_llm,
)
from enums.llm_type import LLMType

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


@router.post("/select-question/score")
async def score_question_endpoint(
    user_need: str = Form(..., description="User requirement / need text"),
    question: str = Form(..., description="Generated question to evaluate"),
    llm_model: str = Form(default="gpt-5.2-2025-12-11", description="LLM model name for scoring"),
    llm_type: str = Form(default="open_ai", description="LLM type: open_ai, groq_ai, ollama, or anthropic"),
    api_key: str = Form(None, description="Optional API key for LLM service"),
):
    """
    Score a single question against a user need using LLM (/100).
    """
    try:
        try:
            llm_type_enum = LLMType.from_string(llm_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        result = score_question_with_llm(
            user_need=user_need,
            question=question,
            llm_model=llm_model,
            llm_type=llm_type_enum,
            api_key=api_key,
        )
        return {
            "message": "Question scored successfully",
            "user_need": user_need,
            "question": question,
            **result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring question: {str(e)}")


@router.post("/select-question/compare")
async def compare_questions_endpoint(
    user_need: str = Form(..., description="User requirement / need text"),
    question_a: str = Form(..., description="Question set A"),
    question_b: str = Form(..., description="Question set B"),
    llm_model: str = Form(default="gpt-5.2-2025-12-11", description="LLM model name for comparison"),
    llm_type: str = Form(default="open_ai", description="LLM type: open_ai, groq_ai, ollama, or anthropic"),
    api_key: str = Form(None, description="Optional API key for LLM service"),
):
    """
    Compare two questions against a user need and decide which is better.
    """
    try:
        try:
            llm_type_enum = LLMType.from_string(llm_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        result = compare_questions_with_llm(
            user_need=user_need,
            question_a=question_a,
            question_b=question_b,
            llm_model=llm_model,
            llm_type=llm_type_enum,
            api_key=api_key,
        )
        return {
            "message": "Question comparison completed successfully",
            "user_need": user_need,
            "question_a": question_a,
            "question_b": question_b,
            **result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing questions: {str(e)}")


def _split_questions(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _extract_questions_from_df(df, text_column: str) -> list[str]:
    if text_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Text column '{text_column}' not found. Available columns: {list(df.columns)}",
        )
    questions = []
    for val in df[text_column].tolist():
        if val is None:
            continue
        s = str(val).strip()
        if not s or s.lower() in ("nan", "none", "null"):
            continue
        questions.append(s)
    if not questions:
        raise HTTPException(status_code=400, detail="No valid questions found in the file.")
    return questions


@router.post("/select-question/score-set")
async def score_question_set_endpoint(
    user_need: str = Form(..., description="User requirement / need text"),
    questions: str = Form(..., description="Question set (one per line)"),
    llm_model: str = Form(default="gpt-5.2-2025-12-11", description="LLM model name for scoring"),
    llm_type: str = Form(default="open_ai", description="LLM type: open_ai, groq_ai, ollama, or anthropic"),
    api_key: str = Form(None, description="Optional API key for LLM service"),
):
    """
    Score a set of questions against a user need using LLM (/100).
    """
    try:
        try:
            llm_type_enum = LLMType.from_string(llm_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        question_list = _split_questions(questions)
        if not question_list:
            raise HTTPException(status_code=400, detail="No questions provided.")

        result = score_question_set_with_llm(
            user_need=user_need,
            questions=question_list,
            llm_model=llm_model,
            llm_type=llm_type_enum,
            api_key=api_key,
        )
        return {
            "message": "Question set scored successfully",
            "user_need": user_need,
            "questions": question_list,
            **result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring question set: {str(e)}")


@router.post("/select-question/compare-sets")
async def compare_question_sets_endpoint(
    user_need: str = Form(..., description="User requirement / need text"),
    questions_a: str = Form(..., description="Question set A (one per line)"),
    questions_b: str = Form(..., description="Question set B (one per line)"),
    llm_model: str = Form(default="gpt-5.2-2025-12-11", description="LLM model name for comparison"),
    llm_type: str = Form(default="open_ai", description="LLM type: open_ai, groq_ai, ollama, or anthropic"),
    api_key: str = Form(None, description="Optional API key for LLM service"),
):
    """
    Compare two question sets against a user need and decide which is better.
    """
    try:
        try:
            llm_type_enum = LLMType.from_string(llm_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        list_a = _split_questions(questions_a)
        list_b = _split_questions(questions_b)
        if not list_a or not list_b:
            raise HTTPException(status_code=400, detail="Both question sets must be provided.")

        result = compare_question_sets_with_llm(
            user_need=user_need,
            questions_a=list_a,
            questions_b=list_b,
            llm_model=llm_model,
            llm_type=llm_type_enum,
            api_key=api_key,
        )
        return {
            "message": "Question set comparison completed successfully",
            "user_need": user_need,
            "questions_a": list_a,
            "questions_b": list_b,
            **result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing question sets: {str(e)}")


@router.post("/select-question/score-set-file")
async def score_question_set_file_endpoint(
    file: UploadFile = File(..., description="CSV file containing generated questions"),
    user_need: str = Form(..., description="User requirement / need text"),
    text_column: str = Form(default="question", description="Name of the question column"),
    llm_model: str = Form(default="gpt-5.2-2025-12-11", description="LLM model name for scoring"),
    llm_type: str = Form(default="open_ai", description="LLM type: open_ai, groq_ai, ollama, or anthropic"),
    api_key: str = Form(None, description="Optional API key for LLM service"),
):
    """
    Score a set of questions from a CSV file against a user need using LLM (/100).
    """
    try:
        try:
            llm_type_enum = LLMType.from_string(llm_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        file_content = await file.read()
        csv_path = save_uploaded_file(file_content, file.filename)
        df = read_csv(csv_path)

        actual_text_col = (text_column or "question").strip() or "question"
        questions = _extract_questions_from_df(df, actual_text_col)

        result = score_question_set_with_llm(
            user_need=user_need,
            questions=questions,
            llm_model=llm_model,
            llm_type=llm_type_enum,
            api_key=api_key,
        )
        return {
            "message": "Question set scored successfully",
            "file": file.filename,
            "text_column": actual_text_col,
            "user_need": user_need,
            "questions": questions,
            **result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring question set: {str(e)}")


@router.post("/select-question/compare-sets-file")
async def compare_question_sets_file_endpoint(
    file_a: UploadFile = File(..., description="CSV file for question set A"),
    file_b: UploadFile = File(..., description="CSV file for question set B"),
    user_need: str = Form(..., description="User requirement / need text"),
    text_column: str = Form(default="question", description="Name of the question column"),
    llm_model: str = Form(default="gpt-5.2-2025-12-11", description="LLM model name for comparison"),
    llm_type: str = Form(default="open_ai", description="LLM type: open_ai, groq_ai, ollama, or anthropic"),
    api_key: str = Form(None, description="Optional API key for LLM service"),
):
    """
    Compare two CSV question sets against a user need and decide which is better.
    """
    try:
        try:
            llm_type_enum = LLMType.from_string(llm_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        file_content_a = await file_a.read()
        file_content_b = await file_b.read()
        csv_path_a = save_uploaded_file(file_content_a, file_a.filename)
        csv_path_b = save_uploaded_file(file_content_b, file_b.filename)
        df_a = read_csv(csv_path_a)
        df_b = read_csv(csv_path_b)

        actual_text_col = (text_column or "question").strip() or "question"
        questions_a = _extract_questions_from_df(df_a, actual_text_col)
        questions_b = _extract_questions_from_df(df_b, actual_text_col)

        result = compare_question_sets_with_llm(
            user_need=user_need,
            questions_a=questions_a,
            questions_b=questions_b,
            llm_model=llm_model,
            llm_type=llm_type_enum,
            api_key=api_key,
        )
        return {
            "message": "Question set comparison completed successfully",
            "file_a": file_a.filename,
            "file_b": file_b.filename,
            "text_column": actual_text_col,
            "user_need": user_need,
            "questions_a": questions_a,
            "questions_b": questions_b,
            **result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing question sets: {str(e)}")
