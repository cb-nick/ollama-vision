FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/logs /app/services /app/ui /app/utils

# Copy application code
COPY main.py config.py ./
COPY services/ ./services/
COPY ui/ ./ui/
COPY utils/ ./utils/

# Copy static files
COPY logo.png ./
COPY *.pkl ./

# Create __init__.py files for proper Python package structure
RUN touch services/__init__.py ui/__init__.py utils/__init__.py

# Create logs volume mount point
VOLUME /app/logs

# Expose port
EXPOSE 8501

# Set environment variables
ENV PYTHONBUFFERED=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.headless=true", "--server.address=0.0.0.0"]
