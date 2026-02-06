"""
Router for evaluation endpoints.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pathlib import Path
import pandas as pd
import logging

from app.utils.file_utils import save_uploaded_file, read_csv
from app.utils.service_utils import evaluate_with_llm_judge
from enums.llm_type import LLMType

router = APIRouter()
logger = logging.getLogger("uvicorn.error")


@router.post("/evaluation/compare")
async def compare_two_csv_files(
    csv_file1: UploadFile = File(..., description="First CSV file for comparison"),
    csv_file2: UploadFile = File(..., description="Second CSV file for comparison"),
    id_column: str = Form(..., description="Name of the ID column in both CSV files"),
    text_column: str = Form(default="text", description="Name of the text/question column"),
    label_column: str = Form(default="labels", description="Name of the label column in both CSV files"),
    method1_name: str = Form(default="Method 1", description="Name of the first labeling method"),
    method2_name: str = Form(default="Method 2", description="Name of the second labeling method"),
    llm_model: str = Form(default="gpt-5.2-2025-12-11", description="LLM model name for evaluation"),
    llm_type: str = Form(default="open_ai", description="LLM type: open_ai, groq_ai, ollama, or anthropic"),
    api_key: str = Form(None, description="Optional API key for LLM service")
):
    """
    Endpoint 7: Accept two CSV files and generate an evaluation comparison using LLM as judge.
    
    Returns detailed evaluation metrics for each question including:
    - Correctness scores (1-5)
    - Completeness scores (1-5)
    - Generalization scores (1-5)
    - Consistency scores (1-5)
    - Reasoning for each evaluation
    - Average metrics across all questions
    - Pairwise comparison counts (method1 wins, method2 wins, ties)
    
    Questions without labels in either method are automatically ignored.
    """
    try:
        logger.info("Received files: %s, %s", csv_file1.filename, csv_file2.filename)
        # Save and read both CSV files
        csv_content1 = await csv_file1.read()
        csv_path1 = save_uploaded_file(csv_content1, csv_file1.filename)
        df1 = read_csv(csv_path1)
        logger.info("First CSV columns: %s", list(df1.columns))
        logger.info("First CSV sample data:\n%s", df1.head())
        csv_content2 = await csv_file2.read()
        csv_path2 = save_uploaded_file(csv_content2, csv_file2.filename)
        df2 = read_csv(csv_path2)
        logger.info("Second CSV columns: %s", list(df2.columns))
        logger.info("Second CSV sample data:\n%s", df2.head())
        # Validate columns exist in both DataFrames
        if id_column not in df1.columns:
            raise HTTPException(
                status_code=400,
                detail=f"ID column '{id_column}' not found in first CSV. Available columns: {list(df1.columns)}"
            )
        
        if id_column not in df2.columns:
            raise HTTPException(
                status_code=400,
                detail=f"ID column '{id_column}' not found in second CSV. Available columns: {list(df2.columns)}"
            )
        
        if label_column not in df1.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Label column '{label_column}' not found in first CSV. Available columns: {list(df1.columns)}"
            )
        
        if label_column not in df2.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Label column '{label_column}' not found in second CSV. Available columns: {list(df2.columns)}"
            )
        
        # Parse LLM type
        try:
            llm_type_enum = LLMType.from_string(llm_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Generate LLM-based evaluation
        actual_text_col = (text_column or "text").strip() or "text"
        if actual_text_col not in df1.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Text column '{actual_text_col}' not found in first CSV. Available: {list(df1.columns)}"
            )
        if actual_text_col not in df2.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Text column '{actual_text_col}' not found in second CSV. Available: {list(df2.columns)}"
            )
        evaluation_result = evaluate_with_llm_judge(
            df1=df1,
            df2=df2,
            id_column=id_column,
            text_column=actual_text_col,
            label_column=label_column,
            method1_name=method1_name,
            method2_name=method2_name,
            llm_model=llm_model,
            llm_type=llm_type_enum,
            api_key=api_key
        )
        
        return {
            "message": "Evaluation completed successfully",
            "file1": csv_file1.filename,
            "file2": csv_file2.filename,
            "evaluation_results": evaluation_result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating CSV files: {str(e)}")
