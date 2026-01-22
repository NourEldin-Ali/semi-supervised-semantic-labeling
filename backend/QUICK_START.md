# Quick Start Guide

## Setup Options

### Option 1: Docker (Recommended)

1. **Create `.env` file** in the `backend` directory:
   ```env
   API_KEY=your_groq_or_openai_key
   EMBEDDING_API_KEY=your_openai_embedding_key
   ```

2. **Build and run with Docker Compose**:
   ```bash
   cd backend
   docker-compose up --build
   ```

   The API will be available at `http://localhost:8000`
   Interactive docs at `http://localhost:8000/docs`

   See `DOCKER.md` for more Docker options and details.

### Option 2: Local Development

1. **Create and Activate Virtual Environment**
   ```bash
   cd backend
   
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   install_deps.sh # or install_deps.bat for windows
   ```

3. **Set Environment Variables**
   Create a `.env` file in the `backend` directory:
   ```env
   API_KEY=your_groq_or_openai_key
   EMBEDDING_API_KEY=your_openai_embedding_key
   ```

4. **Run the Server**
   ```bash
   python run_server.py
   ```

   The API will be available at `http://localhost:8000`
   Interactive docs at `http://localhost:8000/docs`

5. **Deactivate Virtual Environment** (when done)
   ```bash
   deactivate
   ```

## Typical Workflow

### Workflow 1: Full Pipeline (Clustering + Group Labeling + KNN Training)

1. **Generate Embeddings**
   - POST `/api/v1/embeddings/generate`
   - Upload CSV with text data
   - Get embeddings file path

2. **Generate Clusters**
   - POST `/api/v1/clustering/generate`
   - Upload original CSV + embeddings file
   - Get clustered CSV with cluster assignments

3. **Generate Group Labels**
   - POST `/api/v1/labeling/generate-group-labels`
   - Upload clustered CSV + cluster groups JSON
   - Get labeled CSV

4. **Train KNN Model**
   - POST `/api/v1/classification/train-knn`
   - Upload labeled CSV + embeddings file
   - Get trained model file

5. **Predict with KNN**
   - POST `/api/v1/classification/predict`
   - Upload new CSV + new embeddings + trained model
   - Get predictions CSV

### Workflow 2: Direct Item Labeling

1. **Generate Labels for Items**
   - POST `/api/v1/labeling/generate-item-labels`
   - Upload CSV with items
   - Get labeled CSV

### Workflow 3: Compare Two Methods

1. **Compare Results**
   - POST `/api/v1/evaluation/compare`
   - Upload two labeled CSV files
   - Get comparison metrics

## Example CSV Format

Your CSV should have at minimum:
- An ID column (unique identifier for each row)
- A text column (the content to process)

Example:
```csv
id,question
1,"What is machine learning?"
2,"How does neural network work?"
3,"Explain deep learning concepts"
```

## Notes

- All generated files (embeddings, models, outputs) are saved in respective directories
- Files are automatically assigned unique names to avoid conflicts
- The API handles large files with batch processing where applicable
- Check `API_DOCUMENTATION.md` for detailed endpoint documentation
