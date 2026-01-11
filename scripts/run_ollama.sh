#!/bin/bash

# Load environment variables
source ./.env

# Run ollama with the model specified in GEN_AI_MODEL in the background
ollama run "$GEN_AI_MODEL" &
echo "Ollama is running with model: $GEN_AI_MODEL"