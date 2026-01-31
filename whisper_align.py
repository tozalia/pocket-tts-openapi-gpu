#!/usr/bin/env python3
"""
Whisper Transcription for Pocket TTS
====================================
Generate word-level timestamps from audio using Whisper (faster-whisper).
Provides accurate forced alignment for TikTok-style captions.

Installation:
    pip install faster-whisper

Usage:
    python whisper_align.py audio.wav -o captions.json
    
Or use from pocketapi.py:
    from whisper_align import transcribe_with_whisper
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import subprocess
import tempfile
import os

logger = logging.getLogger("whisper-align")

# Try to load faster-whisper
WHISPER_AVAILABLE = False
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    logger.warning("faster-whisper not installed. Run: pip install faster-whisper")


# Global model cache
_whisper_model = None
_whisper_model_name = None


def load_whisper_model(model_name: str = "base.en", device: str = "auto"):
    """Load Whisper model with caching."""
    global _whisper_model, _whisper_model_name
    
    if _whisper_model is not None and _whisper_model_name == model_name:
        return _whisper_model
    
    if not WHISPER_AVAILABLE:
        raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")
    
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    compute_type = "float16" if device == "cuda" else "int8"
    
    logger.info(f"Loading Whisper model: {model_name} on {device}")
    _whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
    _whisper_model_name = model_name
    
    return _whisper_model


def transcribe_with_whisper(
    audio_path: str,
    model_name: str = "base.en",
    language: str = "en",
    fps: int = 30,
    words_per_page: int = 6,
) -> Dict:
    """
    Transcribe audio using Whisper with word-level timestamps.
    
    Returns:
        {
            "captions": [{"text": str, "startMs": int, "endMs": int}],
            "pages": [{"text": str, "startMs": int, "durationMs": int, "tokens": [...]}],
            "audioDuration": int
        }
    """
    model = load_whisper_model(model_name)
    
    logger.info(f"Transcribing: {audio_path}")
    
    # Transcribe with word-level timestamps
    segments, info = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )
    
    # Collect word-level captions
    captions = []
    for segment in segments:
        if segment.words:
            for word in segment.words:
                captions.append({
                    "text": word.word,
                    "startMs": int(word.start * 1000),
                    "endMs": int(word.end * 1000),
                    "startFrame": int(word.start * fps),
                    "endFrame": int(word.end * fps),
                    "confidence": word.probability,
                })
    
    # Create TikTok-style pages
    pages = []
    for i in range(0, len(captions), words_per_page):
        page_captions = captions[i:i + words_per_page]
        if not page_captions:
            continue
        
        page_text = " ".join(c["text"].strip() for c in page_captions)
        start_ms = page_captions[0]["startMs"]
        end_ms = page_captions[-1]["endMs"]
        
        tokens = [{
            "text": c["text"] + (" " if j < len(page_captions) - 1 else ""),
            "fromMs": c["startMs"],
            "toMs": c["endMs"],
        } for j, c in enumerate(page_captions)]
        
        pages.append({
            "text": page_text,
            "startMs": start_ms,
            "durationMs": end_ms - start_ms,
            "tokens": tokens,
        })
    
    audio_duration = captions[-1]["endMs"] if captions else 0
    
    return {
        "captions": captions,
        "pages": pages,
        "audioDuration": audio_duration,
    }


def ensure_16khz_wav(input_path: str) -> str:
    """Convert audio to 16kHz WAV for Whisper."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1",
        tmp.name
    ], capture_output=True, check=True)
    
    return tmp.name


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("-o", "--output", default="captions.json", help="Output JSON path")
    parser.add_argument("-m", "--model", default="base.en", help="Whisper model")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--words-per-page", type=int, default=6, help="Words per caption page")
    
    args = parser.parse_args()
    
    # Transcribe
    result = transcribe_with_whisper(
        args.audio,
        model_name=args.model,
        fps=args.fps,
        words_per_page=args.words_per_page,
    )
    
    # Save
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"✅ Saved {len(result['captions'])} words, {len(result['pages'])} pages to {args.output}")
