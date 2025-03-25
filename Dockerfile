# Usage Instructions
# 1. Recommended way to build all images at once:
#    ./docker_builder.sh 
#    This creates: spark-tts:latest-lite, spark-tts:latest (alias of latest-lite), and spark-tts:latest-full
#
# 2. Manual build without models: 
#    docker build -t spark-tts:latest-lite .
#    docker tag spark-tts:latest-lite spark-tts:latest
#
# 3. Manual build with models: 
#    docker build --build-arg INCLUDE_MODELS=true -t spark-tts:latest-full .
#
# 4. Run container without models (needs to mount models): 
#    docker run -p 7860:7860 --gpus all -v /local/path/pretrained_models:/app/pretrained_models spark-tts:latest-lite
#
# 5. Run container with models: 
#    docker run -p 7860:7860 --gpus all spark-tts:latest-full
# 
# 6. Run with API (default):
#    docker run -p 7860:7860 --gpus all -e SERVICE_TYPE=api spark-tts:latest-full
#
# 7. Run with WebUI:
#    docker run -p 7860:7860 --gpus all -e SERVICE_TYPE=webui spark-tts:latest-full
#
# 8. Use docker-compose for more advanced configurations:
#    docker-compose up api    # Run API service
#    docker-compose up webui  # Run WebUI service
#
# Note:
# - NVIDIA Container Toolkit must be installed on the host to support GPU
# - If using an image without models, you can provide models in the following ways:
#   a) Mount the model directory from the host: docker run -p 7860:7860 --gpus all -v /local/path/pretrained_models:/app/pretrained_models spark-tts:latest-lite
#   b) Download models inside the container: python -c "from huggingface_hub import snapshot_download; snapshot_download('SparkAudio/Spark-TTS-0.5B', local_dir='pretrained_models/Spark-TTS-0.5B')" 

# Base stage: common setup without models
FROM python:3.12-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    ffmpeg \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model directory (empty by default)
RUN mkdir -p pretrained_models

# Copy project files (layered for caching)
COPY cli/ ./cli/
COPY sparktts/ ./sparktts/
COPY src/ ./src/
COPY example/ ./example/
COPY api/ ./api/
COPY webui.py .
COPY LICENSE README.md ./

# Create outputs directory for API
RUN mkdir -p /app/api/outputs && chmod 777 /app/api/outputs

# Make run_api.sh executable
RUN chmod +x /app/api/run_api.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV SERVICE_TYPE=api

# Expose port
EXPOSE 7860

# Builder stage: intermediate stage to handle models
FROM base AS builder
ARG INCLUDE_MODELS=false
COPY . /tmp/context/
RUN if [ "${INCLUDE_MODELS}" = "true" ] && [ -d "/tmp/context/pretrained_models" ]; then \
        echo "Including models in the image"; \
        cp -r /tmp/context/pretrained_models/* /app/pretrained_models/ || echo "No model files to copy"; \
    else \
        echo "Models will need to be mounted at runtime"; \
    fi

# Full image: includes models if INCLUDE_MODELS=true
FROM builder AS full
CMD if [ "$SERVICE_TYPE" = "webui" ]; then \
        python webui.py --device 0; \
    else \
        ./api/run_api.sh; \
    fi

# Lite image: no models, directly from base
FROM base AS lite
CMD if [ "$SERVICE_TYPE" = "webui" ]; then \
        python webui.py --device 0; \
    else \
        ./api/run_api.sh; \
    fi
