# Semi-Supervised Labeling Framework API Documentation

## Overview

This FastAPI backend provides endpoints for:
1. Generating embeddings from CSV files
2. Clustering data based on embeddings
3. Generating labels using LLMs (both group-based and individual item labeling)
4. Training KNN classification models
5. Predicting labels using trained KNN models
6. Evaluating and comparing two labeling methods

## Base URL

```
http://localhost:8000/api/v1
```

## Endpoints

### 1. Generate Embeddings

**POST** `/embeddings/generate`

Upload a CSV file and generate embedding files based on a selected embedding model.

**Request:**
- `file` (multipart/form-data): CSV file to process
- `embedding_model` (form): Embedding model name (e.g., 'text-embedding-ada-002', 'all-MiniLM-L6-v2')
- `embed_type` (form, optional): Embedding type - `open_ai`, `ollama`, or `huggingface` (default: `open_ai`)
- `text_column` (form, optional): Name of the text column (auto-detected if not provided)
- `batch_size` (form, optional): Batch size for embedding generation (default: 32)
- `api_key` (form, optional): Optional API key for embedding service

**Response:**
```json
{
  "message": "Embeddings generated successfully",
  "embeddings_file": "path/to/embeddings.npy",
  "csv_file": "path/to/original.csv",
  "embedding_shape": [100, 384],
  "text_column": "question"
}
```

---

### 2. Generate Clustering Groups

**POST** `/clustering/generate`

Accept embeddings and original CSV file to generate clustering groups. Produces a new CSV file containing all cluster groups and associated items.

**Request:**
- `csv_file` (multipart/form-data): Original CSV file
- `embeddings_file` (multipart/form-data): Embeddings file (.npy)
- `text_column` (form, optional): Name of the text column (auto-detected if not provided)
- `k` (form, optional): Number of clusters (default: 3)
- `metric` (form, optional): Distance metric for clustering (default: `cosine`)
- `m` (form, optional): Fuzziness parameter for possibilistic clustering (default: 2.0)

**Response:**
```json
{
  "message": "Clustering completed successfully",
  "output_csv_file": "path/to/clustered_output.csv",
  "original_csv_file": "path/to/original.csv",
  "number_of_clusters": 5,
  "cluster_summary": [
    {
      "cluster_id": 0,
      "item_count": 10,
      "item_indices": [0, 1, 2, ...]
    }
  ]
}
```

---

### 3. Generate Labels for Clustering Groups

**POST** `/labeling/generate-group-labels`

Accept clustering groups and CSV file to generate labels for each group using LLM. Assigns labels to each individual item based on their cluster group.

**Request:**
- `csv_file` (multipart/form-data): CSV file containing clustered data
- `clusters_json` (form): JSON string of cluster groups. Format: `{"clusters": [[0,1,2], [3,4,5], ...]}` or `[[0,1,2], [3,4,5], ...]`
- `id_column` (form): Name of the ID column in CSV
- `text_column` (form, optional): Name of the text column (auto-detected if not provided)
- `llm_model` (form, optional): LLM model name (default: `llama-3.3-70b-versatile`)
- `llm_type` (form, optional): LLM type - `open_ai`, `groq_ai`, `ollama`, or `anthropic` (default: `groq_ai`)
- `api_key` (form, optional): Optional API key for LLM service

**Response:**
```json
{
  "message": "Labels generated successfully",
  "output_csv_file": "path/to/labeled_output.csv",
  "original_csv_file": "path/to/original.csv",
  "labeled_items_count": 100,
  "total_items": 100,
  "labels_generated": 100
}
```

---

### 4. Train KNN Model

**POST** `/classification/train-knn`

Accept labeled CSV file and embeddings to train and prepare a KNN model.

**Request:**
- `csv_file` (multipart/form-data): Labeled CSV file
- `embeddings_file` (multipart/form-data): Embeddings file (.npy)
- `id_column` (form): Name of the ID column in CSV
- `label_column` (form, optional): Name of the label column in CSV (default: `labels`)

**Response:**
```json
{
  "message": "KNN model trained successfully",
  "model_file": "path/to/trained_model.joblib",
  "csv_file": "path/to/labeled.csv",
  "embeddings_file": "embeddings.npy",
  "training_samples": 80,
  "total_samples": 100
}
```

---

### 5. Predict Labels with KNN

**POST** `/classification/predict`

Use trained KNN model with new embeddings and CSV file to generate predicted labels. Returns a CSV file containing items and their predicted labels.

**Request:**
- `csv_file` (multipart/form-data): New CSV file to predict labels for
- `embeddings_file` (multipart/form-data): New embeddings file (.npy)
- `model_file` (multipart/form-data): Trained KNN model file (.joblib)
- `id_column` (form): Name of the ID column in CSV
- `k` (form, optional): Number of nearest neighbors for prediction (default: 3)

**Response:**
```json
{
  "message": "Predictions generated successfully",
  "output_csv_file": "path/to/predicted_output.csv",
  "original_csv_file": "path/to/new_data.csv",
  "predicted_items_count": 50,
  "total_items": 50
}
```

---

### 6. Generate Labels for Individual Items

**POST** `/labeling/generate-item-labels`

Accept a CSV file and generate labels for each item. Returns a CSV file with items and their labels.

**Request:**
- `csv_file` (multipart/form-data): CSV file containing items to label
- `id_column` (form): Name of the ID column in CSV
- `text_column` (form, optional): Name of the text column (auto-detected if not provided)
- `llm_model` (form, optional): LLM model name (default: `llama-3.3-70b-versatile`)
- `llm_type` (form, optional): LLM type - `open_ai`, `groq_ai`, `ollama`, or `anthropic` (default: `groq_ai`)
- `api_key` (form, optional): Optional API key for LLM service

**Response:**
```json
{
  "message": "Labels generated successfully",
  "output_csv_file": "path/to/labeled_output.csv",
  "original_csv_file": "path/to/original.csv",
  "labeled_items_count": 100,
  "total_items": 100
}
```

---

### 7. Evaluate and Compare Two CSV Files

**POST** `/evaluation/compare`

Accept two CSV files and generate an evaluation comparison between two different labeling methods. The response includes run statistics: **execution time** (seconds), **tokens consumed**, and **energy consumed** (kWh, via [CodeCarbon](https://codecarbon.io/)), plus optional CO₂ emissions (kg CO₂eq).

**Request:**
- `csv_file1` (multipart/form-data): First CSV file for comparison
- `csv_file2` (multipart/form-data): Second CSV file for comparison
- `id_column` (form): Name of the ID column in both CSV files
- `label_column` (form, optional): Name of the label column in both CSV files (default: `labels`)

**Response:**
```json
{
  "message": "Comparison completed successfully",
  "file1": "method1_results.csv",
  "file2": "method2_results.csv",
  "comparison_results": {
    "total_items_method1": 100,
    "total_items_method2": 100,
    "common_items": 100,
    "exact_matches": 75,
    "partial_matches": 20,
    "no_matches": 5,
    "exact_match_rate": 0.75,
    "partial_match_rate": 0.20,
    "average_jaccard_similarity": 0.85,
    "comparison_details": [
      {
        "item_id": "1",
        "method1_labels": ["label1", "label2"],
        "method2_labels": ["label1"],
        "common_labels": ["label1"],
        "match_type": "partial",
        "jaccard_similarity": 0.5
      }
    ]
  }
}
```

---

## File Downloads

### Download Embeddings

**GET** `/embeddings/download/{file_path}`

Download an embeddings file (.npy).

### Download CSV Files

**GET** `/clustering/download/{file_path}`  
**GET** `/labeling/download/{file_path}`  
**GET** `/classification/download/{file_path}`

Download generated CSV files.

---

## Health Check

**GET** `/health`

Returns API health status.

**GET** `/`

Returns API information.

---

## Environment Variables

Create a `.env` file in the backend directory with:

```env
# For LLM services
API_KEY=your_api_key_here
LLM_TEMPERATURE=0.0

# For embedding services
EMBEDDING_API_KEY=your_embedding_api_key_here

# For Ollama (if using)
ENDPOINT=http://localhost:11434
```

---

## Running the Server

```bash
# Activate virtual environment
conda activate .venv  # or: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
python run_server.py

# Or using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`

Interactive API documentation (Swagger UI) will be available at: `http://localhost:8000/docs`
