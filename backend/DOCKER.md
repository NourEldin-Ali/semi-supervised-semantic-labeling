# Docker Setup Guide

This guide explains how to build and run the backend API using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier management)

## Quick Start

### Using Docker Compose (Recommended)

1. **Create a `.env` file** in the `backend` directory:
   ```env
   API_KEY=your_groq_or_openai_key
   EMBEDDING_API_KEY=your_openai_embedding_key
   LLM_TEMPERATURE=0.0
   ENDPOINT=http://host.docker.internal:11434  # Optional, for Ollama
   ```

2. **Build and run the container**:
   ```bash
   cd backend
   docker-compose up --build
   ```

   To run in detached mode (background):
   ```bash
   docker-compose up -d --build
   ```

3. **View logs**:
   ```bash
   docker-compose logs -f
   ```

4. **Stop the container**:
   ```bash
   docker-compose down
   ```

### Using Docker directly

1. **Build the image**:
   ```bash
   cd backend
   docker build -t semi-supervised-labeling-api .
   ```

2. **Run the container**:
   ```bash
   docker run -d \
     --name semi-supervised-labeling-api \
     -p 8000:8000 \
     -e API_KEY=your_api_key \
     -e EMBEDDING_API_KEY=your_embedding_api_key \
     -v $(pwd)/uploads:/app/uploads \
     -v $(pwd)/embeddings:/app/embeddings \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/outputs:/app/outputs \
     semi-supervised-labeling-api
   ```

   Or with a `.env` file:
   ```bash
   docker run -d \
     --name semi-supervised-labeling-api \
     -p 8000:8000 \
     --env-file .env \
     -v $(pwd)/uploads:/app/uploads \
     -v $(pwd)/embeddings:/app/embeddings \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/outputs:/app/outputs \
     semi-supervised-labeling-api
   ```

3. **View logs**:
   ```bash
   docker logs -f semi-supervised-labeling-api
   ```

4. **Stop the container**:
   ```bash
   docker stop semi-supervised-labeling-api
   docker rm semi-supervised-labeling-api
   ```

## Accessing the API

Once the container is running, the API will be available at:
- API: `http://localhost:8000`
- Interactive Docs (Swagger UI): `http://localhost:8000/docs`
- Alternative Docs (ReDoc): `http://localhost:8000/redoc`
- Health Check: `http://localhost:8000/health`

## Volume Mounts

The Docker setup mounts local directories for persistent storage:
- `./uploads` - Uploaded CSV files
- `./embeddings` - Generated embedding files (.npy)
- `./models` - Trained KNN models (.joblib)
- `./outputs` - Generated output CSV files

This ensures that files persist even when the container is stopped or removed.

## Environment Variables

**Important**: Docker Compose automatically loads environment variables from the `.env` file in the `backend` directory via the `env_file` configuration. You don't need to manually pass them.

Create a `.env` file in the `backend` directory:

| Variable | Description | Required |
|----------|-------------|----------|
| `API_KEY` | API key for LLM services (Groq, OpenAI, Anthropic) | Yes (for LLM features) |
| `EMBEDDING_API_KEY` | API key for embedding services (OpenAI). Also accepts `OPENAI_API_KEY` | Yes (for OpenAI embeddings) |
| `LLM_TEMPERATURE` | Temperature for LLM generation | No (default: 0.0) |
| `ENDPOINT` | Endpoint URL for Ollama | No (if using Ollama) |

**How it works in Docker:**
- Docker Compose reads `./backend/.env` and injects all variables as environment variables
- The application reads from these environment variables automatically
- You can still override via form parameters in API requests if needed

## Troubleshooting

### Container won't start

1. **Check logs**:
   ```bash
   docker-compose logs backend
   # or
   docker logs semi-supervised-labeling-api
   ```

2. **Check port availability**:
   Make sure port 8000 is not already in use:
   ```bash
   # Linux/Mac
   lsof -i :8000
   
   # Windows
   netstat -ano | findstr :8000
   ```

3. **Rebuild the image**:
   ```bash
   docker-compose build --no-cache
   ```

### Permission issues with volumes

If you encounter permission issues with mounted volumes:

```bash
# Fix permissions (Linux/Mac)
sudo chown -R $USER:$USER uploads embeddings models outputs

# Or run container with different user
docker run --user $(id -u):$(id -g) ...
```

### Environment variables not loading

- Ensure `.env` file is in the same directory as `docker-compose.yml`
- Check that variable names match exactly (case-sensitive)
- Verify no spaces around `=` in `.env` file

## Production Considerations

For production deployment, consider:

1. **Use specific image tags** instead of `latest`
2. **Set proper resource limits** in docker-compose.yml:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
   ```

3. **Use secrets management** instead of .env file
4. **Set up reverse proxy** (nginx/traefik) for HTTPS
5. **Enable logging** to external service
6. **Use health checks** for container orchestration
7. **Set up backups** for persistent volumes

## Building for different architectures

To build for different CPU architectures (e.g., ARM for Apple Silicon):

```bash
# Build for ARM64
docker buildx build --platform linux/arm64 -t semi-supervised-labeling-api .

# Build for AMD64
docker buildx build --platform linux/amd64 -t semi-supervised-labeling-api .
```
