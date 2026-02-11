# Use full Debian-based Python image that already includes build tools + git
FROM python:3.11-bookworm

# Set working directory
WORKDIR /app

# System build tools + git are already present in the base image

# Copy requirements file
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-build-isolation "git+https://github.com/bm424/scikit-cmeans.git@master"

# Copy application code
COPY . .

# Create directories for file uploads, embeddings, models, and outputs
RUN mkdir -p uploads embeddings models outputs

# Expose port 8000
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
