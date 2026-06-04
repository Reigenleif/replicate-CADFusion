#!/bin/bash

# Load environment variables
source ./.env

# Run ollama with the model specified in GEN_AI_MODEL in the background
ollama run "$DPO_VLM_MODEL_ID" &
echo "Ollama is running with model: $DPO_VLM_MODEL_ID"