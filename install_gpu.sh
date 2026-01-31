#!/bin/bash
# Pocket TTS GPU - Complete Install Script with MFA
# Sets up Python environment with GPU support and Montreal Forced Aligner

set -e
cd "$(dirname "$0")"

echo "🔧 Setting up Pocket TTS GPU environment with MFA..."
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

# Install Montreal Forced Aligner
echo ""
echo "📦 Installing Montreal Forced Aligner..."

# MFA is best installed via conda - check if available
if command -v conda &> /dev/null; then
    echo "✓ Conda found - installing MFA via conda-forge"
    
    # Create MFA environment if it doesn't exist
    if ! conda env list | grep -q "^mfa "; then
        conda create -n mfa -c conda-forge montreal-forced-aligner -y
    else
        echo "✓ MFA conda environment already exists"
    fi
    
    # Download English acoustic model and dictionary
    echo ""
    echo "📥 Downloading MFA English models..."
    conda run -n mfa mfa model download acoustic english_us_arpa || true
    conda run -n mfa mfa model download dictionary english_us_arpa || true
    
    MFA_INSTALLED=1
else
    echo "⚠ Conda not found - MFA requires conda for full functionality"
    echo "  Install conda and run: conda create -n mfa -c conda-forge montreal-forced-aligner"
    echo "  Or use the estimation fallback (proportional word timing)"
    MFA_INSTALLED=0
fi

# Install pyfoal as Python wrapper (optional additional method)
echo ""
echo "📦 Installing pyfoal (Python wrapper for MFA)..."
if [ "$USE_UV" = "1" ]; then
    uv pip install pyfoal 2>/dev/null || echo "⚠ pyfoal install skipped (requires MFA)"
else
    pip install pyfoal 2>/dev/null || echo "⚠ pyfoal install skipped (requires MFA)"
fi

# Install Whisper as fallback alignment method
echo ""
echo "📦 Installing OpenAI Whisper (fallback alignment)..."
if [ "$USE_UV" = "1" ]; then
    uv pip install openai-whisper
else
    pip install openai-whisper
fi

# Create voices directory
mkdir -p voices

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ Installation complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📁 Project structure:"
echo "   pocketapi.py       - Main API server"
echo "   forced_align.py    - Forced alignment utility"
echo "   voices/            - Place voice reference WAVs here"
echo ""
echo "🚀 To start the server:"
echo "   source .venv/bin/activate"
echo "   python pocketapi.py"
echo ""
echo "🎯 Alignment methods available:"
if [ "$MFA_INSTALLED" = "1" ]; then
    echo "   ✓ MFA (Montreal Forced Aligner) - Best accuracy"
fi
echo "   ✓ Whisper transcription + ground-truth mapping"
echo "   ✓ Proportional estimation (fallback)"
echo ""
echo "📝 Example usage:"
echo "   # Generate audio + captions"
echo "   curl -X POST http://localhost:8001/v1/audio/speech-with-alignment \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"input\": \"Hello world\", \"voice\": \"alloy\"}'"
echo ""
echo "   # Force align existing audio"
echo "   python forced_align.py --audio narration.wav --script script.txt -o captions.json"
echo ""
