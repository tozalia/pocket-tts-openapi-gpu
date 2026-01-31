#!/bin/bash
# Pocket TTS GPU - Start Script
# Starts the TTS server with GPU optimizations

cd "$(dirname "$0")"

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# GPU Configuration
export POCKET_TTS_DEVICE="auto"       # auto, cuda, cpu
export POCKET_TTS_WORKERS="4"          # Concurrent requests
export POCKET_TTS_COMPILE="true"       # torch.compile() optimization
export POCKET_TTS_TF32="true"          # TF32 for Ampere+ GPUs
export POCKET_TTS_CUDNN_BENCH="true"   # cuDNN benchmark mode
export POCKET_TTS_MODEL="Verylicious/pocket-tts-ungated"  # Ungated (no login)

echo "🚀 Starting Pocket TTS with GPU optimizations..."
echo "   Device: $POCKET_TTS_DEVICE"
echo "   Workers: $POCKET_TTS_WORKERS"
echo "   torch.compile: $POCKET_TTS_COMPILE"

python pocketapi.py
