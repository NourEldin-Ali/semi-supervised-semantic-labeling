"""
Main FastAPI application entry point.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import embeddings, clustering, labeling, classification, evaluation, workflow, select_question

# Load environment variables from .env file
# In Docker, environment variables are injected via docker-compose env_file
# This will load .env if it exists locally, but won't fail if it doesn't (Docker scenario)
try:
    backend_dir = Path(__file__).parent.parent
    env_path = backend_dir / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        # Fallback to current directory
        load_dotenv(override=False)
except Exception:
    # If .env loading fails, that's okay - Docker provides env vars via docker-compose
    pass

app = FastAPI(
    title="Semi-Supervised Labeling Framework API",
    description="Backend API for semi-supervised labeling, clustering, and classification",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(embeddings.router, prefix="/api/v1", tags=["embeddings"])
app.include_router(clustering.router, prefix="/api/v1", tags=["clustering"])
app.include_router(labeling.router, prefix="/api/v1", tags=["labeling"])
app.include_router(classification.router, prefix="/api/v1", tags=["classification"])
app.include_router(evaluation.router, prefix="/api/v1", tags=["evaluation"])
app.include_router(workflow.router, prefix="/api/v1", tags=["workflow"])
app.include_router(select_question.router, prefix="/api/v1", tags=["select-question"])


@app.get("/")
async def root():
    """Root endpoint for API health check."""
    return {"message": "Semi-Supervised Labeling Framework API", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
