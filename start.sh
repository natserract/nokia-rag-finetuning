#!/bin/bash

# ---------------------------------------------------------------------------- #
#                          Function Definitions                                #
# ---------------------------------------------------------------------------- #

start_ollama() {
    echo "Starting Ollama service..."
    nohup ollama serve > /ollama.log 2>&1 &

    # Wait for Ollama to start up
    echo "Waiting for Ollama to initialize..."
    until curl -s http://localhost:11434/api/version >/dev/null; do
        sleep 1
    done

    # Load the model
    echo "Loading deepseek-r1:8b model..."
    ollama run deepseek-r1:8b > /deepseek-r1:8b.log 2>&1 &

    # Verify model is ready
    sleep 3
    echo "Available models:"
    ollama list
}

# ---------------------------------------------------------------------------- #
#                               Main Program                                   #
# ---------------------------------------------------------------------------- #

# Start required services
start_ollama

# Run handler
echo "Starting app..."
uv run main.py

# Keep container running (remove if handler runs continuously)
sleep infinity
