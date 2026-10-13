# AquaFlex AI — Backend API container image
# Serves the FastAPI application (uvicorn backend.main:app).
FROM python:3.11-slim

# System dependencies required by rasterio / GDAL and general geospatial stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Fail fast on Python-level issues and keep logs unbuffered.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application source.
COPY backend ./backend
COPY frontend ./frontend

EXPOSE 8000

# Basic container healthcheck hitting the FastAPI health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Launch the API. --reload is omitted for production stability.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
