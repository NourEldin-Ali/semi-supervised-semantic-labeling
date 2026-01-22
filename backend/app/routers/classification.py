"""
Router for classification endpoints.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from pathlib import Path
import pandas as pd
import tempfile
import numpy as np

from app.utils.file_utils import (
    save_uploaded_file, read_csv, save_csv, save_model, load_model,
    load_embeddings
)
from app.utils.execution_stats import run_with_stats
from app.utils.service_utils import train_knn_model, predict_with_knn

router = APIRouter()


@router.post("/classification/train-knn")
async def train_knn_model_endpoint(
    csv_file: UploadFile = File(..., description="Labeled CSV file"),
    embeddings_file: UploadFile = File(..., description="Embeddings file (.npy)"),
    id_column: str = Form(..., description="Name of the ID column in CSV"),
    label_column: str = Form(default="labels", description="Name of the label column in CSV")
):
    """
    Endpoint 4: Accept labeled CSV file and embeddings to train and prepare a KNN model.
    """
    try:
        # Save and read CSV file
        csv_content = await csv_file.read()
        csv_path = save_uploaded_file(csv_content, csv_file.filename)
        df = read_csv(csv_path)
        
        # Validate columns exist
        if id_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"ID column '{id_column}' not found in CSV. Available columns: {list(df.columns)}"
            )
        
        if label_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Label column '{label_column}' not found in CSV. Available columns: {list(df.columns)}"
            )
        
        # Save and load embeddings file
        embeddings_content = await embeddings_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_file:
            tmp_file.write(embeddings_content)
            tmp_path = Path(tmp_file.name)
        
        embeddings = load_embeddings(tmp_path)
        tmp_path.unlink()  # Clean up temp file
        
        # Validate embeddings shape matches DataFrame
        if len(embeddings) != len(df):
            raise HTTPException(
                status_code=400,
                detail=f"Embeddings shape {len(embeddings)} does not match CSV rows {len(df)}"
            )
        
        def _run():
            knn_model, label_data, training_embeddings = train_knn_model(
                embeddings=embeddings,
                labeled_df=df,
                id_column=id_column,
                label_column=label_column,
            )
            model_filename = f"knn_model_{Path(csv_file.filename).stem}"
            model_path = save_model({
                "knn_model": knn_model,
                "label_data": label_data,
                "training_embeddings": training_embeddings,
            }, model_filename)
            return label_data, model_path

        (label_data, model_path), stats = run_with_stats(_run)
        training_count = len(label_data["labels"])

        return {
            "message": "KNN model trained successfully",
            "model_file": str(model_path),
            "csv_file": str(csv_path),
            "embeddings_file": str(embeddings_file.filename),
            "training_samples": training_count,
            "total_samples": len(df),
            **stats,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training KNN model: {str(e)}")


@router.post("/classification/predict")
async def predict_labels_with_knn(
    csv_file: UploadFile = File(..., description="New CSV file to predict labels for"),
    embeddings_file: UploadFile = File(..., description="New embeddings file (.npy)"),
    model_file: UploadFile = File(..., description="Trained KNN model file (.joblib)"),
    id_column: str = Form(..., description="Name of the ID column in CSV"),
    k: int = Form(default=3, description="Number of nearest neighbors for prediction")
):
    """
    Endpoint 5: Use trained KNN model with new embeddings and CSV file to generate predicted labels.
    Returns a CSV file containing items and their predicted labels.
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
        
        # Save and load embeddings file
        embeddings_content = await embeddings_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_file:
            tmp_file.write(embeddings_content)
            tmp_path = Path(tmp_file.name)
        
        new_embeddings = load_embeddings(tmp_path)
        tmp_path.unlink()  # Clean up temp file
        
        # Validate embeddings shape matches DataFrame
        if len(new_embeddings) != len(df):
            raise HTTPException(
                status_code=400,
                detail=f"Embeddings shape {len(new_embeddings)} does not match CSV rows {len(df)}"
            )
        
        # Save and load model file
        model_content = await model_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp_file:
            tmp_file.write(model_content)
            tmp_path = Path(tmp_file.name)
        
        model_data = load_model(tmp_path)
        tmp_path.unlink()  # Clean up temp file
        
        knn_model = model_data['knn_model']
        label_data = model_data['label_data']
        training_embeddings = model_data['training_embeddings']
        
        def _run():
            predicted_labels = predict_with_knn(
                knn_model=knn_model,
                label_data=label_data,
                training_embeddings=training_embeddings,
                new_embeddings=new_embeddings,
                k=k,
            )
            result_df = df.copy()
            result_df["labels"] = predicted_labels
            output_filename = f"predicted_{Path(csv_file.filename).stem}.csv"
            output_path = save_csv(result_df, output_filename)
            return predicted_labels, output_path

        (predicted_labels, output_path), stats = run_with_stats(_run)
        predicted_count = sum(1 for label in predicted_labels if label)

        return {
            "message": "Predictions generated successfully",
            "output_csv_file": str(output_path),
            "original_csv_file": str(csv_path),
            "predicted_items_count": predicted_count,
            "total_items": len(df),
            **stats,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating predictions: {str(e)}")


@router.get("/classification/download/{file_path:path}")
async def download_result_csv(file_path: str):
    """Download a result CSV file or model file."""
    file = Path(file_path)
    if not file.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type based on file extension
    if file.suffix == '.joblib':
        media_type = 'application/octet-stream'
    elif file.suffix == '.csv':
        media_type = 'text/csv'
    else:
        media_type = 'application/octet-stream'
    
    return FileResponse(file, filename=file.name, media_type=media_type)
