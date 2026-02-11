# Use Python 3.11 slim image as base (pin to stable Debian)
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies (use HTTPS + retries to avoid transient mirror timeouts)
RUN set -eux; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list; \
    fi; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources; \
    fi; \
    apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update; \
    apt-get install -y --no-install-recommends \
        git \
        build-essential; \
    rm -rf /var/lib/apt/lists/*

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
