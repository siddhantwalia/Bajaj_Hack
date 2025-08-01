FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: install nomic CLI and python-dotenv
RUN pip install --no-cache-dir nomic python-dotenv

# Copy application files (excluding entrypoint.sh for now)
COPY . .

# Copy and set permissions for the entrypoint (do this AFTER copying everything else)
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
ENV PORT=8000

# Set the entrypoint to the script
ENTRYPOINT ["/entrypoint.sh"]