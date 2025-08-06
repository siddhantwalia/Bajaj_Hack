FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies including Tesseract
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgomp1 \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Tesseract environment variables
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
ENV TESSERACT_CMD=/usr/bin/tesseract

# Upgrade pip first
RUN pip install --upgrade pip

# Copy and verify requirements file
COPY requirements.txt .
RUN ls -la requirements.txt && head -10 requirements.txt

# Install Python packages with verbose output for debugging
# RUN pip install --no-cache-dir --verbose -r requirements.txt

# Install additional packages
# RUN pip install --no-cache-dir nomic python-dotenv

# Copy application files
COPY . .

# Copy and set permissions for the entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
ENV PORT=8000

# Set the entrypoint to the script
ENTRYPOINT ["/entrypoint.sh"]