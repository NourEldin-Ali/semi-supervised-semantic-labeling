# Use Python image that already includes build tools + git
FROM python:3.11-slim

# Set working directory
WORKDIR /app


# System deps commonly needed for numpy/scipy/sklearn builds + git URL installs
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     git \
#     build-essential \
#     python3-dev \
#     gfortran \
#     pkg-config \
#     libopenblas-dev \
#     liblapack-dev \
#  && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*
 
# WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Avoid --no-build-isolation unless you have a specific reason
RUN pip install --no-cache-dir -vvv "git+https://github.com/bm424/scikit-cmeans.git@master"

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
