#!/bin/bash
# Pocket TTS GPU - Install Script
# Sets up Python environment with GPU support

cd "$(dirname "$0")"

echo "🔧 Setting up Pocket TTS GPU environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
echo "📦 Installing PyTorch with CUDA..."
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
echo "📦 Installing pocket-tts and dependencies..."
pip install pocket-tts>=1.0.1 pydantic>=2.12.5 soundfile>=0.13.1 fastapi>=0.110.0 uvicorn>=0.29.0 anyio

echo ""
echo "✅ Installation complete!"
echo ""
echo "To start the server:"
echo "  ./start_gpu.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python pocketapi.py"
