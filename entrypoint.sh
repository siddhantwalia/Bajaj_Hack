#!/bin/bash

echo "Logging into Nomic..."
echo "Using token: ${EMBEDDING_API_KEY:0:4}********"  # optional debug, hides most of the key

# Login using token (non-interactive mode)
nomic login $EMBEDDING_API_KEY

# Start FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000
