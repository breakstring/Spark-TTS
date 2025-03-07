#!/bin/bash

# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Get the absolute path of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the project root directory
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_LAUNCH_BLOCKING=1

# Default parameters
HOST="0.0.0.0"
PORT=8000
DEBUG=false
RELOAD=false

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --debug)
      DEBUG=true
      RELOAD=true
      shift
      ;;
    --reload)
      RELOAD=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if environment file exists
ENV_FILE="$SCRIPT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
  echo "Environment file does not exist, creating from example: $ENV_FILE"
  cp "$SCRIPT_DIR/.env.example" "$ENV_FILE"
fi

# Load environment variables
if [ -f "$ENV_FILE" ]; then
  echo "Loading environment variables: $ENV_FILE"
  # Extract SPARK_TTS_OUTPUT_DIR value
  OUTPUT_DIR=$(grep "SPARK_TTS_OUTPUT_DIR" "$ENV_FILE" | cut -d '=' -f2)
  if [ -z "$OUTPUT_DIR" ]; then
    # If not set, use default value
    OUTPUT_DIR="$SCRIPT_DIR/outputs"
  else
    # Convert to absolute path
    if [[ "$OUTPUT_DIR" = /* ]]; then
      # Already an absolute path
      echo "Using absolute output path: $OUTPUT_DIR"
    else
      # Relative path, convert to absolute path
      OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
      echo "Output directory (relative to absolute): $OUTPUT_DIR"
    fi
  fi
else
  # If no environment file, use default value
  OUTPUT_DIR="$SCRIPT_DIR/outputs"
fi

# Create output directory
echo "Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Check default prompt audio
DEFAULT_PROMPT_AUDIO="$PROJECT_ROOT/example/prompt_audio.wav"
if [ ! -f "$DEFAULT_PROMPT_AUDIO" ]; then
  echo "Warning: Default prompt audio does not exist: $DEFAULT_PROMPT_AUDIO"
fi

# Check model directory
MODEL_DIR="$PROJECT_ROOT/pretrained_models/Spark-TTS-0.5B"
if [ ! -d "$MODEL_DIR" ]; then
  echo "Warning: Model directory does not exist: $MODEL_DIR"
  echo "Please ensure you have downloaded the model files and placed them in the correct location"
else
  CONFIG_FILE="$MODEL_DIR/config.yaml"
  if [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Model configuration file does not exist: $CONFIG_FILE"
  else
    echo "Model directory check passed: $MODEL_DIR"
  fi
fi

# Start API service
cd "$PROJECT_ROOT"
echo "Starting Spark-TTS API service..."
echo "Project root directory: $PROJECT_ROOT"
echo "Current directory: $(pwd)"

# Set RELOAD parameter
if [ "$RELOAD" = true ]; then
  RELOAD_ARG="--reload"
else
  RELOAD_ARG=""
fi

# Start API service
python -m uvicorn api.main:app --host "$HOST" --port "$PORT" $RELOAD_ARG

# Check exit status
if [ $? -ne 0 ]; then
  echo "Failed to start API service!"
  exit 1
fi

echo "API service has stopped" 