#!/bin/bash

echo "Starting the server..."
# Start FastAPI server
uvicorn main2:app --host 0.0.0.0 --port 8000
