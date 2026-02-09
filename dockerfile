# Use lightweight Python image
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
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    libgl1 \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data/processed

# Make entrypoint executable and default to starting the Health Chatbot API (Streamlit).
# KB builder is a separate task and will only run if RUN_KB_ON_STARTUP=true.
RUN chmod +x /app/entrypoint.sh

CMD ["bash", "/app/entrypoint.sh"]