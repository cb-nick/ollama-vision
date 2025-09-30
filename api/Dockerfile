# Dockerfile for FastAPI-based API service
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI default port
EXPOSE 8505

# Run the API with uvicorn, matching the options in image_recognition_api.py
CMD ["uvicorn", "image_recognition_api:app", "--host", "0.0.0.0", "--port", "8505", "--reload", "--ssl-keyfile", "./certs/api-selfsigned.key", "--ssl-certfile", "./certs/api-selfsigned.crt"]
