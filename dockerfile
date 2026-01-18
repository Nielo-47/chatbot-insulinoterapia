# Use lightweight Python image (no CUDA needed - Ollama runs on host)
FROM python:3.12-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (remove vLLM, add ollama client)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Setup entrypoint
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Create data directory
RUN mkdir -p /app/data/processed

# Expose port
EXPOSE 8080

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]