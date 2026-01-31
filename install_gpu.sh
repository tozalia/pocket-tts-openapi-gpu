#!/bin/bash
# Pocket TTS GPU - Complete Install Script with Whisper
# Sets up Python environment with GPU support and Whisper alignment

set -e
cd "$(dirname "$0")"

echo "🔧 Setting up Pocket TTS GPU environment with Whisper..."
echo ""

# Check for uv (faster) or fall back to pip
if command -v uv &> /dev/null; then
    USE_UV=1
    echo "✓ Using uv for fast installation"
else
    USE_UV=0
    echo "ℹ Using pip (install uv for faster setup: curl -LsSf https://astral.sh/uv/install.sh | sh)"
fi

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ "$USE_UV" = "1" ]; then
    uv venv .venv
    source .venv/bin/activate
else
    python3 -m venv .venv
    source .venv/bin/activate
fi

# Install PyTorch with CUDA support
echo ""
echo "📦 Installing PyTorch with CUDA..."
if [ "$USE_UV" = "1" ]; then
    uv pip install torch --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch --index-url https://download.pytorch.org/whl/cu121
fi

# Install core dependencies
echo ""
echo "📦 Installing Pocket TTS and dependencies..."
if [ "$USE_UV" = "1" ]; then
    uv pip install pocket-tts pydantic soundfile fastapi uvicorn anyio numpy
else
    pip install pocket-tts pydantic soundfile fastapi uvicorn anyio numpy
fi

# Install faster-whisper for accurate alignment
echo ""
echo "📦 Installing faster-whisper (accurate word-level timestamps)..."
if [ "$USE_UV" = "1" ]; then
    uv pip install faster-whisper
else
    pip install faster-whisper
fi

# Preload Whisper model for faster first request
echo ""
echo "📥 Preloading Whisper base.en model..."
python3 -c "
from faster_whisper import WhisperModel
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
compute = 'float16' if device == 'cuda' else 'int8'
print(f'Loading Whisper base.en on {device}...')
model = WhisperModel('base.en', device=device, compute_type=compute)
print('✓ Whisper model cached')
" 2>/dev/null || echo "⚠ Whisper preload skipped (will download on first use)"

# Create voices directory
mkdir -p voices

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ Installation complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📁 Project structure:"
echo "   pocketapi.py       - Main API server"
echo "   whisper_align.py   - Whisper alignment module"
echo "   voices/            - Place voice reference WAVs here"
echo ""
echo "🚀 To start the server:"
echo "   source .venv/bin/activate"
echo "   python pocketapi.py"
echo ""
echo "🎯 Available endpoints:"
echo "   POST /v1/audio/speech                - OpenAI-compatible TTS"
echo "   POST /v1/audio/speech-with-alignment - TTS + proportional timestamps"
echo "   POST /v1/audio/speech-with-whisper   - TTS + Whisper timestamps (most accurate)"
echo ""
echo "📝 Example usage (Whisper alignment):"
echo "   curl -X POST http://localhost:8001/v1/audio/speech-with-whisper \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"input\": \"Hello world\", \"voice\": \"alloy\"}'"
echo ""

