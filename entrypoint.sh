#!/bin/bash

echo "Starting the server..."
# Start FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000
