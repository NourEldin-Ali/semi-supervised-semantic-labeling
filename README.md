# Semi-Supervised Labeling Framework

A complete full-stack application for semi-supervised labeling, clustering, and classification using embeddings, LLMs, and KNN models.

## Features

- **Embedding Generation**: Generate embeddings from CSV files using various models (OpenAI, Ollama, HuggingFace)
- **Clustering**: Group similar items using possibilistic clustering
- **Labeling**: Generate labels using LLMs (group-based or individual item labeling)
- **KNN Classification**: Train and use KNN models for label prediction
- **Evaluation**: Compare and evaluate different labeling methods

## Architecture

- **Backend**: FastAPI (Python)
- **Frontend**: React + TypeScript + Vite.js
- **Database**: File-based storage (CSV, embeddings, models)

## Quick Start

### Using Docker Compose (Recommended)

1. **Set up environment variables**:

   Create `backend/.env`:
   ```env
   API_KEY=your_groq_or_openai_key
   EMBEDDING_API_KEY=your_openai_embedding_key
   ```

2. **Run everything**:
   ```bash
   docker-compose up --build
   ```

   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Manual Setup

#### Backend

```bash
cd backend
pip install -r requirements.txt
python run_server.py
```

See `backend/README.md` for details.

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

See `frontend/README.md` for details.

## Project Structure

```
.
├── backend/          # FastAPI backend
│   ├── app/         # FastAPI application
│   ├── src/         # Core modules (embedding, clustering, labeling, classification)
│   ├── config/      # Configuration
│   └── Dockerfile   # Backend Docker image
├── frontend/        # React frontend
│   ├── src/         # Source code
│   └── Dockerfile   # Frontend Docker image
└── docker-compose.yml  # Docker Compose configuration
```

## Documentation

- **Backend API**: See `backend/API_DOCUMENTATION.md`
- **Backend Docker**: See `backend/DOCKER.md`
- **Frontend**: See `frontend/README.md`
- **Frontend Docker**: See `frontend/DOCKER.md`

## Workflows

### Main Workflow (4 Steps)

1. **Generate Embeddings**: Upload CSV → Get embeddings file
2. **Generate Clusters**: Upload CSV + embeddings → Get clustered CSV
3. **Generate Labels**: Upload clustered CSV + cluster groups → Get labeled CSV
4. **Train KNN**: Upload labeled CSV + embeddings → Get trained model

### KNN Prediction

- Upload new CSV + embeddings + trained model → Get predictions

### LLM Labeling

- Upload CSV → Get labeled CSV (direct LLM labeling)

### Evaluation

- Upload two labeled CSV files → Get comparison metrics

## License

MIT
