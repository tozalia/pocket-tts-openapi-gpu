"""
Pocket TTS OpenAPI - GPU Enhanced Edition
==========================================
OpenAI-compatible TTS API with GPU acceleration, multi-threading,
and word-level alignment for Remotion captions.

Based on patterns from groxaxo/Qwen3-TTS-Openai-Fastapi.

Environment Variables:
    POCKET_TTS_DEVICE: "auto", "cuda", or "cpu" (default: auto)
    POCKET_TTS_WORKERS: Max concurrent requests (default: 4)
    POCKET_TTS_COMPILE: Enable torch.compile() (default: false - safer)
    POCKET_TTS_MODEL: HuggingFace model ID (default: Verylicious/pocket-tts-ungated)
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Queue, Full
from typing import Literal, Optional, AsyncIterator, List

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pocket-tts")

DEVICE = os.getenv("POCKET_TTS_DEVICE", "auto")
MAX_WORKERS = int(os.getenv("POCKET_TTS_WORKERS", "4"))
USE_COMPILE = os.getenv("POCKET_TTS_COMPILE", "false").lower() == "true"
MODEL_REPO = os.getenv("POCKET_TTS_MODEL", "Verylicious/pocket-tts-ungated")

QUEUE_SIZE = 256
QUEUE_TIMEOUT = 10.0
EOF_TIMEOUT = 1.0
CHUNK_SIZE = 32 * 1024
DEFAULT_SAMPLE_RATE = 24000
CACHE_DIR = "audio_cache"
VOICES_DIR = "voices"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(VOICES_DIR, exist_ok=True)

# Thread safety
inference_semaphore = threading.Semaphore(MAX_WORKERS)

# ============================================================================
# TORCH SETUP (Optional GPU Optimizations)
# ============================================================================
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    
    # cuDNN Benchmark Mode
    if os.getenv("POCKET_TTS_CUDNN_BENCH", "true").lower() == "true":
        torch.backends.cudnn.benchmark = True
    
    # TF32 for Ampere+ GPUs
    if os.getenv("POCKET_TTS_TF32", "true").lower() == "true":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
except ImportError:
    pass

# ============================================================================
# GLOBAL STATE
# ============================================================================
tts_model = None
device_str = "cpu"
sample_rate = DEFAULT_SAMPLE_RATE
_warmup_complete = False
_request_count = 0

# Voice state cache - caches expensive get_state_for_audio_prompt() results
_voice_state_cache = {}
_voice_state_cache_lock = threading.Lock()

# OpenAI Voice Mapping
VOICE_MAPPING = {
    "alloy": "alba",
    "echo": "jean",
    "fable": "fantine",
    "onyx": "cosette",
    "nova": "eponine",
    "shimmer": "azelma",
}

POCKET_VOICES = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]

FFMPEG_FORMATS = {
    "mp3": ("mp3", "libmp3lame"),
    "opus": ("ogg", "libopus"),
    "aac": ("adts", "aac"),
    "flac": ("flac", "flac"),
}

CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "aac": "audio/aac",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}

# ============================================================================
# PYDANTIC SCHEMAS (OpenAI-compatible)
# ============================================================================
class OpenAISpeechRequest(BaseModel):
    """OpenAI-compatible speech request."""
    model: str = Field(default="tts-1", description="Model to use")
    input: str = Field(..., max_length=4096, description="Text to synthesize")
    voice: str = Field(default="alloy", description="Voice to use")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(default="mp3")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 1737734400
    owned_by: str = "pocket-tts"


class VoiceInfo(BaseModel):
    id: str
    name: str
    description: Optional[str] = None


class SpeechWithAlignmentRequest(BaseModel):
    """Unified TTS + alignment request for Remotion."""
    input: str = Field(..., max_length=4096)
    voice: str = Field(default="alloy")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    fps: int = Field(default=30, ge=1, le=120)
    words_per_page: int = Field(default=6, ge=1, le=20)


class TikTokToken(BaseModel):
    text: str
    fromMs: int
    toMs: int


class TikTokPage(BaseModel):
    text: str
    startMs: int
    durationMs: int
    tokens: List[TikTokToken]


class RemotionCaption(BaseModel):
    text: str
    startMs: int
    endMs: int
    startFrame: Optional[int] = None
    endFrame: Optional[int] = None


# ============================================================================
# MODEL LOADING
# ============================================================================
def detect_device() -> str:
    if DEVICE != "auto":
        return DEVICE
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_tts_model():
    global tts_model, device_str, sample_rate
    
    from pocket_tts import TTSModel
    
    device_str = detect_device()
    logger.info(f"🎯 Device: {device_str} | Model: {MODEL_REPO}")
    
    os.environ["POCKET_TTS_MODEL_ID"] = MODEL_REPO
    
    start = time.time()
    tts_model = TTSModel.load_model()
    sample_rate = getattr(tts_model, "sample_rate", DEFAULT_SAMPLE_RATE)
    logger.info(f"✅ Model loaded in {time.time() - start:.1f}s (sample_rate={sample_rate})")
    
    # GPU placement
    if device_str == "cuda" and hasattr(tts_model, 'model'):
        try:
            tts_model.model = tts_model.model.to(device_str)
            logger.info(f"✅ Model moved to GPU ({torch.cuda.get_device_name()})")
        except Exception as e:
            logger.warning(f"GPU move failed: {e}")
            device_str = "cpu"
    
    # torch.compile (optional)
    if USE_COMPILE and TORCH_AVAILABLE and hasattr(tts_model, 'model'):
        try:
            tts_model.model = torch.compile(tts_model.model, mode="reduce-overhead")
            logger.info("✅ torch.compile() applied")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
    
    load_custom_voices()


def load_custom_voices():
    """Load voices from voices/ directory."""
    for f in Path(VOICES_DIR).glob("*.wav"):
        voice_name = f.stem
        VOICE_MAPPING[voice_name] = str(f.resolve())
        logger.info(f"🎤 Custom voice: {voice_name}")


# ============================================================================
# AUDIO GENERATION
# ============================================================================
class QueueWriter:
    def __init__(self, queue: Queue, timeout: float = QUEUE_TIMEOUT):
        self.queue = queue
        self.timeout = timeout

    def write(self, data: bytes) -> int:
        if not data:
            return 0
        try:
            self.queue.put(data, timeout=self.timeout)
            return len(data)
        except Full:
            return 0

    def flush(self):
        pass

    def close(self):
        try:
            self.queue.put(None, timeout=EOF_TIMEOUT)
        except:
            pass


def get_voice_state(voice_name: str):
    """Get voice state with caching for expensive get_state_for_audio_prompt calls."""
    global _voice_state_cache
    
    # Check cache first
    with _voice_state_cache_lock:
        if voice_name in _voice_state_cache:
            logger.debug(f"Voice cache HIT: {voice_name}")
            return _voice_state_cache[voice_name]
    
    # Generate new state (expensive operation)
    logger.info(f"Voice cache MISS: {voice_name} - generating state...")
    model_state = tts_model.get_state_for_audio_prompt(voice_name)
    
    # Cache it
    with _voice_state_cache_lock:
        _voice_state_cache[voice_name] = model_state
    
    return model_state


def generate_audio_sync(voice_name: str, text: str) -> bytes:
    """Generate audio synchronously with thread safety and voice caching."""
    global _warmup_complete, _request_count
    
    with inference_semaphore:
        start = time.time()
        
        if not _warmup_complete:
            logger.info("🔥 Warmup inference...")
        
        # Use cached voice state
        model_state = get_voice_state(voice_name)
        
        audio_chunks = tts_model.generate_audio_stream(
            model_state=model_state,
            text_to_generate=text
        )
        
        # Collect audio chunks to numpy array
        all_chunks = []
        for chunk in audio_chunks:
            if isinstance(chunk, np.ndarray):
                all_chunks.append(chunk)
            else:
                all_chunks.append(np.array(chunk))
        
        if not all_chunks:
            raise RuntimeError("No audio chunks generated")
        
        audio_array = np.concatenate(all_chunks)
        
        # Encode to WAV using soundfile
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sample_rate, format='WAV')
        audio_data = buffer.getvalue()
        
        elapsed = time.time() - start
        _request_count += 1
        
        if not _warmup_complete:
            _warmup_complete = True
            logger.info(f"✅ Warmup complete ({elapsed:.1f}s)")
        else:
            logger.info(f"Generated {len(audio_data)} bytes in {elapsed:.1f}s")
        
        return audio_data


def encode_audio(wav_data: bytes, fmt: str) -> bytes:
    """Encode WAV to target format using FFmpeg."""
    if fmt in ("wav", "pcm"):
        return wav_data
    
    if fmt not in FFMPEG_FORMATS:
        return wav_data
    
    out_fmt, codec = FFMPEG_FORMATS[fmt]
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "wav", "-i", "pipe:0",
        "-f", out_fmt, "-codec:a", codec, "pipe:1"
    ]
    
    try:
        proc = subprocess.run(cmd, input=wav_data, capture_output=True, timeout=30)
        if proc.returncode == 0:
            return proc.stdout
        logger.warning(f"FFmpeg failed: {proc.stderr.decode()[:200]}")
    except Exception as e:
        logger.warning(f"FFmpeg error: {e}")
    
    return wav_data


# ============================================================================
# WORD TIMESTAMP ESTIMATION
# ============================================================================
def estimate_word_timestamps(text: str, duration_ms: int, fps: int = 30) -> List[RemotionCaption]:
    """Estimate word-level timestamps based on character distribution."""
    words = text.split()
    if not words:
        return []
    
    total_chars = sum(len(w) for w in words)
    if total_chars == 0:
        return []
    
    captions = []
    current_ms = 0
    
    for word in words:
        word_weight = len(word) / total_chars
        word_duration = max(50, int(duration_ms * word_weight))
        end_ms = min(current_ms + word_duration, duration_ms)
        
        captions.append(RemotionCaption(
            text=word,
            startMs=current_ms,
            endMs=end_ms,
            startFrame=int((current_ms / 1000) * fps),
            endFrame=int((end_ms / 1000) * fps),
        ))
        current_ms = end_ms
    
    return captions


def captions_to_pages(captions: List[RemotionCaption], words_per_page: int) -> List[TikTokPage]:
    """Convert captions to TikTok-style pages."""
    pages = []
    
    for i in range(0, len(captions), words_per_page):
        page_captions = captions[i:i + words_per_page]
        if not page_captions:
            continue
        
        page_text = " ".join(c.text for c in page_captions)
        start_ms = page_captions[0].startMs
        end_ms = page_captions[-1].endMs
        
        tokens = [
            TikTokToken(
                text=c.text + (" " if j < len(page_captions) - 1 else ""),
                fromMs=c.startMs,
                toMs=c.endMs,
            )
            for j, c in enumerate(page_captions)
        ]
        
        pages.append(TikTokPage(
            text=page_text,
            startMs=start_ms,
            durationMs=end_ms - start_ms,
            tokens=tokens,
        ))
    
    return pages


# ============================================================================
# FASTAPI APP
# ============================================================================
@asynccontextmanager
async def lifespan(app):
    logger.info("🚀 Starting Pocket TTS API...")
    load_tts_model()
    yield

app = FastAPI(
    title="Pocket TTS OpenAPI",
    description="OpenAI-compatible TTS API with GPU support",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def home():
    return f"""<!DOCTYPE html><html><head><title>Pocket TTS</title>
    <style>body{{font-family:system-ui;max-width:800px;margin:40px auto;padding:20px;background:#1a1a2e;color:#fff}}
    a{{color:#00d9ff}}code{{background:#333;padding:2px 6px;border-radius:4px}}</style></head>
    <body><h1>🎤 Pocket TTS API</h1>
    <p><b>Device:</b> {device_str} | <b>Model:</b> {MODEL_REPO}</p>
    <p><b>Endpoints:</b></p>
    <ul>
    <li><code>POST /v1/audio/speech</code> - OpenAI-compatible TTS</li>
    <li><code>POST /v1/audio/speech-with-alignment</code> - TTS + Remotion captions</li>
    <li><code>GET /v1/models</code> - List models</li>
    <li><code>GET /v1/voices</code> - List voices</li>
    <li><a href="/docs">Swagger UI</a></li>
    </ul></body></html>"""


@app.post("/v1/audio/speech")
async def create_speech(request: OpenAISpeechRequest):
    """OpenAI-compatible TTS endpoint."""
    if tts_model is None:
        raise HTTPException(503, "Model not loaded")
    
    voice_name = VOICE_MAPPING.get(request.voice.lower(), request.voice)
    
    try:
        # Generate audio synchronously in thread pool
        wav_data = await asyncio.to_thread(generate_audio_sync, voice_name, request.input)
        
        # Encode to target format
        audio_data = await asyncio.to_thread(encode_audio, wav_data, request.response_format)
        
        return Response(
            content=audio_data,
            media_type=CONTENT_TYPES.get(request.response_format, "audio/mpeg"),
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
            },
        )
    except Exception as e:
        logger.exception(f"Speech generation failed: {e}")
        raise HTTPException(500, f"Generation failed: {str(e)}")


@app.post("/v1/audio/speech-with-alignment")
async def speech_with_alignment(request: SpeechWithAlignmentRequest):
    """Generate speech + word-level timestamps for Remotion (proportional estimation)."""
    if tts_model is None:
        raise HTTPException(503, "Model not loaded")
    
    voice_name = VOICE_MAPPING.get(request.voice.lower(), request.voice)
    
    try:
        # Generate audio
        wav_data = await asyncio.to_thread(generate_audio_sync, voice_name, request.input)
        
        # Get duration from audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_data)
            tmp_path = tmp.name
        
        try:
            info = sf.info(tmp_path)
            duration_ms = int(info.duration * 1000)
        finally:
            os.unlink(tmp_path)
        
        # Generate timestamps
        captions = estimate_word_timestamps(request.input, duration_ms, request.fps)
        pages = captions_to_pages(captions, request.words_per_page)
        
        return {
            "audio_base64": base64.b64encode(wav_data).decode(),
            "audio_duration_ms": duration_ms,
            "sample_rate": sample_rate,
            "captions": [c.model_dump() for c in captions],
            "timeline": [{"text": c.text, "startMs": c.startMs, "endMs": c.endMs} for c in captions],
            "pages": [p.model_dump() for p in pages],
        }
    except Exception as e:
        logger.exception(f"Speech+alignment failed: {e}")
        raise HTTPException(500, f"Generation failed: {str(e)}")


@app.post("/v1/audio/speech-with-whisper")
async def speech_with_whisper(request: SpeechWithAlignmentRequest):
    """Generate speech + Whisper-based word-level timestamps (most accurate)."""
    if tts_model is None:
        raise HTTPException(503, "Model not loaded")
    
    voice_name = VOICE_MAPPING.get(request.voice.lower(), request.voice)
    
    try:
        # Generate audio
        wav_data = await asyncio.to_thread(generate_audio_sync, voice_name, request.input)
        
        # Save to temp file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_data)
            tmp_path = tmp.name
        
        try:
            # Use Whisper for accurate transcription
            from whisper_align import transcribe_with_whisper
            
            whisper_result = await asyncio.to_thread(
                transcribe_with_whisper,
                tmp_path,
                model_name="base.en",
                fps=request.fps,
                words_per_page=request.words_per_page,
            )
            
            info = sf.info(tmp_path)
            duration_ms = int(info.duration * 1000)
        finally:
            os.unlink(tmp_path)
        
        return {
            "audio_base64": base64.b64encode(wav_data).decode(),
            "audio_duration_ms": duration_ms,
            "sample_rate": sample_rate,
            "captions": whisper_result["captions"],
            "timeline": [{"text": c["text"], "startMs": c["startMs"], "endMs": c["endMs"]} for c in whisper_result["captions"]],
            "pages": whisper_result["pages"],
            "alignment_method": "whisper",
        }
    except ImportError:
        raise HTTPException(500, "Whisper not available. Install: pip install faster-whisper")
    except Exception as e:
        logger.exception(f"Speech+whisper failed: {e}")
        raise HTTPException(500, f"Generation failed: {str(e)}")


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            ModelInfo(id="tts-1", owned_by="pocket-tts").model_dump(),
            ModelInfo(id="tts-1-hd", owned_by="pocket-tts").model_dump(),
            ModelInfo(id="pocket-tts", owned_by="pocket-tts").model_dump(),
        ],
    }


@app.get("/v1/voices")
@app.get("/v1/audio/voices")
async def list_voices():
    """List available voices."""
    openai_voices = [
        VoiceInfo(id="alloy", name="Alloy", description="Maps to alba"),
        VoiceInfo(id="echo", name="Echo", description="Maps to jean"),
        VoiceInfo(id="fable", name="Fable", description="Maps to fantine"),
        VoiceInfo(id="onyx", name="Onyx", description="Maps to cosette"),
        VoiceInfo(id="nova", name="Nova", description="Maps to eponine"),
        VoiceInfo(id="shimmer", name="Shimmer", description="Maps to azelma"),
    ]
    
    pocket_voices = [VoiceInfo(id=v, name=v.capitalize()) for v in POCKET_VOICES]
    
    custom_voices = [
        VoiceInfo(id=k, name=k, description="Custom voice")
        for k in VOICE_MAPPING if k not in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        and k not in POCKET_VOICES
    ]
    
    return {
        "voices": [v.model_dump() for v in openai_voices + pocket_voices + custom_voices],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": tts_model is not None,
        "device": device_str,
        "sample_rate": sample_rate,
        "warmup_complete": _warmup_complete,
        "requests_served": _request_count,
        "workers": MAX_WORKERS,
        "model_repo": MODEL_REPO,
    }


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import socket
    
    def find_port(start=8001, retries=20):
        for p in range(start, start + retries):
            try:
                with socket.socket() as s:
                    s.bind(("0.0.0.0", p))
                    return p
            except:
                continue
        return 8001
    
    port = find_port()
    logger.info(f"🚀 http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
