"""
Pocket TTS OpenAPI - GPU Enhanced Edition
==========================================
OpenAI-compatible TTS API with GPU acceleration and multi-threading.

Optimizations adapted from groxaxo/Qwen3-TTS-Openai-Fastapi:
- torch.compile() with reduce-overhead mode
- cuDNN benchmark mode for convolution optimization
- TF32 precision for Ampere+ GPUs (RTX 30xx/40xx)
- Thread-safe concurrent inference with semaphore
- Audio caching for repeated phrases

Environment Variables:
    POCKET_TTS_DEVICE: "auto", "cuda", or "cpu" (default: auto)
    POCKET_TTS_WORKERS: Max concurrent requests (default: 4)
    POCKET_TTS_COMPILE: Enable torch.compile() (default: true)
    POCKET_TTS_MODEL: HuggingFace model ID (default: Verylicious/pocket-tts-ungated)
    POCKET_TTS_TF32: Enable TF32 for Ampere+ GPUs (default: true)
    POCKET_TTS_CUDNN_BENCH: Enable cuDNN benchmark (default: true)
"""

import asyncio
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uvicorn
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field, field_validator
from queue import Queue, Full
from typing import Literal, Optional, AsyncIterator, Dict, Any
import soundfile as sf
import numpy as np
from anyio import open_file
import tempfile
import shutil
from pathlib import Path

# Forced Alignment support (MFA via pyfoal or fallback)
ALIGNMENT_AVAILABLE = False
try:
    import pyfoal
    ALIGNMENT_AVAILABLE = True
    logger = logging.getLogger("pocket-tts")  # Will be defined later, just checking import
except ImportError:
    pass

# ============================================================================
# TORCH SETUP & GPU OPTIMIZATIONS (from Qwen3-TTS patterns)
# ============================================================================
try:
    import torch
    TORCH_AVAILABLE = True
    
    # cuDNN Benchmark Mode - finds fastest convolution algorithms
    if os.getenv("POCKET_TTS_CUDNN_BENCH", "true").lower() == "true":
        torch.backends.cudnn.benchmark = True
    
    # TF32 Precision - 3-5x speedup on Ampere+ GPUs (RTX 30xx/40xx)
    if os.getenv("POCKET_TTS_TF32", "true").lower() == "true":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

except ImportError:
    TORCH_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pocket-tts")

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = os.getenv("POCKET_TTS_DEVICE", "auto")
MAX_WORKERS = int(os.getenv("POCKET_TTS_WORKERS", "4"))
USE_COMPILE = os.getenv("POCKET_TTS_COMPILE", "true").lower() == "true"
MODEL_REPO = os.getenv("POCKET_TTS_MODEL", "Verylicious/pocket-tts-ungated")

# Constants
QUEUE_SIZE = 256
QUEUE_TIMEOUT = 10.0
EOF_TIMEOUT = 1.0
CHUNK_SIZE = 32 * 1024
DEFAULT_SAMPLE_RATE = 24000
AUDIO_CACHE_DIR = "audio_cache"
VOICES_DIR = "voices"
CACHE_LIMIT = 50  # Increased for production

os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
os.makedirs(VOICES_DIR, exist_ok=True)

# Thread safety
inference_semaphore = threading.Semaphore(MAX_WORKERS)
inference_lock = threading.RLock()

# Performance tracking
_warmup_complete = False
_request_count = 0
_total_inference_time = 0.0

# ============================================================================
# TERMINAL COLORS
# ============================================================================
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# ============================================================================
# VOICE MAPPINGS
# ============================================================================
VOICE_MAPPING = {
    # OpenAI -> Pocket TTS mappings
    "alloy": "alba",
    "echo": "jean",
    "fable": "fantine",
    "onyx": "cosette",
    "nova": "eponine",
    "shimmer": "azelma",
}

DEFAULT_VOICES = {
    "openai_aliases": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    "pocket_tts": ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
}

FFMPEG_FORMATS = {
    "mp3": ("mp3", "mp3_mf" if sys.platform == "win32" else "libmp3lame"),
    "opus": ("ogg", "libopus"),
    "aac": ("adts", "aac"),
    "flac": ("flac", "flac"),
}

MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "aac": "audio/aac",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}

# ============================================================================
# GLOBAL STATE
# ============================================================================
tts_model: Optional[object] = None
device_str: Optional[str] = None
sample_rate: Optional[int] = None
optimization_info: Dict[str, Any] = {}


# ============================================================================
# GPU DETECTION & MODEL LOADING
# ============================================================================
def detect_device() -> str:
    """Detect best available device with capability check."""
    if DEVICE != "auto":
        return DEVICE
    
    if not TORCH_AVAILABLE:
        return "cpu"
    
    if torch.cuda.is_available():
        # Check GPU compute capability for advanced features
        capability = torch.cuda.get_device_capability()
        gpu_name = torch.cuda.get_device_name()
        logger.info(f"{Colors.CYAN}🎮 GPU: {gpu_name} (Compute {capability[0]}.{capability[1]}){Colors.RESET}")
        
        # Flash Attention 2 needs compute >= 7.5
        if capability[0] >= 7 or (capability[0] == 7 and capability[1] >= 5):
            optimization_info["flash_attention_eligible"] = True
            
        # TF32 needs Ampere+ (compute >= 8.0)
        if capability[0] >= 8:
            optimization_info["tf32_eligible"] = True
            logger.info(f"{Colors.GREEN}✓ TF32 enabled (Ampere+ GPU){Colors.RESET}")
        
        return "cuda"
    
    return "cpu"


def load_tts_model() -> None:
    """Load TTS model with GPU optimizations."""
    global tts_model, device_str, sample_rate, _warmup_complete
    
    from pocket_tts import TTSModel
    
    device_str = detect_device()
    
    logger.info(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
    logger.info(f"{Colors.CYAN}{Colors.BOLD}🚀 Pocket TTS Initialization{Colors.RESET}")
    logger.info(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
    logger.info(f"   Device: {Colors.GREEN}{device_str}{Colors.RESET}")
    logger.info(f"   Model: {Colors.GREEN}{MODEL_REPO}{Colors.RESET}")
    logger.info(f"   Workers: {Colors.GREEN}{MAX_WORKERS}{Colors.RESET}")
    logger.info(f"   torch.compile: {Colors.GREEN if USE_COMPILE else Colors.YELLOW}{USE_COMPILE}{Colors.RESET}")
    
    # Set environment for ungated model
    os.environ["POCKET_TTS_MODEL_ID"] = MODEL_REPO
    
    # Load model
    start_time = time.time()
    tts_model = TTSModel.load_model()
    load_time = time.time() - start_time
    sample_rate = getattr(tts_model, "sample_rate", DEFAULT_SAMPLE_RATE)
    
    logger.info(f"   Model loaded in: {Colors.GREEN}{load_time:.2f}s{Colors.RESET}")
    
    # GPU placement
    if device_str == "cuda" and hasattr(tts_model, 'model'):
        try:
            tts_model.model = tts_model.model.to(device_str)
            vram_used = torch.cuda.memory_allocated() / 1024**2
            logger.info(f"   VRAM used: {Colors.GREEN}{vram_used:.0f} MB{Colors.RESET}")
            optimization_info["gpu_memory_mb"] = vram_used
        except Exception as e:
            logger.warning(f"{Colors.YELLOW}⚠️ GPU move failed: {e}{Colors.RESET}")
            device_str = "cpu"
    
    # torch.compile() optimization (from Qwen3-TTS patterns)
    if USE_COMPILE and TORCH_AVAILABLE and hasattr(tts_model, 'model'):
        try:
            logger.info(f"{Colors.CYAN}🔧 Applying torch.compile(mode='reduce-overhead')...{Colors.RESET}")
            logger.info(f"{Colors.YELLOW}   ⚠️ First 2-3 requests will be slower (warmup){Colors.RESET}")
            
            # Use reduce-overhead mode for inference speed (from Qwen3-TTS)
            tts_model.model = torch.compile(
                tts_model.model, 
                mode="reduce-overhead",
                fullgraph=False  # Allow partial graph compilation
            )
            optimization_info["torch_compile"] = True
            logger.info(f"{Colors.GREEN}   ✓ torch.compile() applied{Colors.RESET}")
        except Exception as e:
            logger.warning(f"{Colors.YELLOW}   ⚠️ torch.compile() failed: {e}{Colors.RESET}")
            optimization_info["torch_compile"] = False
    
    # Store optimization state
    optimization_info.update({
        "device": device_str,
        "model_repo": MODEL_REPO,
        "sample_rate": sample_rate,
        "cudnn_benchmark": torch.backends.cudnn.benchmark if TORCH_AVAILABLE else False,
        "tf32_enabled": torch.backends.cuda.matmul.allow_tf32 if TORCH_AVAILABLE else False,
    })
    
    logger.info(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
    load_custom_voices()
    
    # Warmup will happen on first request
    _warmup_complete = False


def load_custom_voices():
    """Scan voices directory and update mapping."""
    custom_voices = []
    if os.path.exists(VOICES_DIR):
        for f in os.listdir(VOICES_DIR):
            if f.lower().endswith(".wav"):
                voice_name = os.path.splitext(f)[0]
                file_path = os.path.join(VOICES_DIR, f)
                
                try:
                    file_info = sf.info(file_path)
                    if file_info.subtype != 'PCM_16':
                        logger.info(f"Converting {f} to PCM_16...")
                        audio_data, sr = sf.read(file_path)
                        if audio_data.dtype in [np.float32, np.float64]:
                            audio_data = np.clip(audio_data, -1.0, 1.0)
                        audio_data = (audio_data * 32767).astype(np.int16)
                        sf.write(file_path, audio_data, sr, subtype='PCM_16')
                    
                    full_path = os.path.abspath(file_path).replace("\\", "/")
                    VOICE_MAPPING[voice_name] = full_path
                    custom_voices.append(voice_name)
                except Exception as e:
                    logger.warning(f"Failed to load voice '{voice_name}': {e}")
    
    logger.info(f"{Colors.CYAN}🔊 Voices: {', '.join(DEFAULT_VOICES['pocket_tts'])}{Colors.RESET}")
    if custom_voices:
        logger.info(f"{Colors.GREEN}🎤 Custom: {', '.join(custom_voices)}{Colors.RESET}")


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================
class SpeechRequest(BaseModel):
    model: Literal["tts-1", "tts-1-hd", "pocket-tts"] = Field("tts-1")
    input: str = Field(..., min_length=1, max_length=4096)
    voice: str = Field("alloy")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field("mp3")
    speed: Optional[float] = Field(1.0, ge=0.25, le=4.0)

    @field_validator("model", mode="before")
    @classmethod
    def validate_model(cls, v): return v or "tts-1"

    @field_validator("voice", mode="before")
    @classmethod
    def validate_voice(cls, v): return v.strip() if v else "alloy"

    @field_validator("response_format", mode="before")
    @classmethod
    def validate_format(cls, v): return v or "mp3"


# ============================================================================
# AUDIO STREAMING (Thread-Safe)
# ============================================================================
class FileLikeQueueWriter:
    def __init__(self, queue: Queue, timeout: float = QUEUE_TIMEOUT):
        self.queue = queue
        self.timeout = timeout

    def write(self, data: bytes) -> int:
        if not data: return 0
        try:
            self.queue.put(data, timeout=self.timeout)
            return len(data)
        except Full:
            return 0

    def flush(self): pass
    
    def close(self):
        try: self.queue.put(None, timeout=EOF_TIMEOUT)
        except: pass

    def __enter__(self): return self
    def __exit__(self, *args): self.close(); return False


def _start_audio_producer(queue: Queue, voice_name: str, text: str) -> threading.Thread:
    """Thread-safe audio producer with semaphore for concurrent inference."""
    from pocket_tts.data.audio import stream_audio_chunks
    global _warmup_complete, _request_count, _total_inference_time

    def producer():
        global _warmup_complete, _request_count, _total_inference_time
        
        # Acquire semaphore for thread-safe concurrent inference
        with inference_semaphore:
            start_time = time.time()
            try:
                if not _warmup_complete:
                    logger.info(f"{Colors.YELLOW}🔥 Warmup inference (may be slow)...{Colors.RESET}")
                
                model_state = tts_model.get_state_for_audio_prompt(voice_name)
                audio_chunks = tts_model.generate_audio_stream(
                    model_state=model_state, text_to_generate=text
                )
                
                with FileLikeQueueWriter(queue) as writer:
                    stream_audio_chunks(writer, audio_chunks, sample_rate or DEFAULT_SAMPLE_RATE)
                
                elapsed = time.time() - start_time
                _request_count += 1
                _total_inference_time += elapsed
                
                if not _warmup_complete:
                    _warmup_complete = True
                    logger.info(f"{Colors.GREEN}✓ Warmup complete ({elapsed:.2f}s){Colors.RESET}")
                else:
                    logger.info(f"[Worker] Generated in {elapsed:.2f}s (avg: {_total_inference_time/_request_count:.2f}s)")
                    
            except Exception as e:
                logger.exception(f"Audio generation failed: {e}")
            finally:
                try: queue.put(None, timeout=EOF_TIMEOUT)
                except: pass

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    return thread


async def _stream_queue_chunks(queue: Queue) -> AsyncIterator[bytes]:
    while True:
        chunk = await asyncio.to_thread(queue.get)
        if chunk is None: break
        yield chunk


def _start_ffmpeg_process(fmt: str) -> tuple:
    out_fmt, codec = FFMPEG_FORMATS.get(fmt, ("mp3", "libmp3lame"))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "wav", "-i", "pipe:0"]
    if fmt == "mp3": cmd.extend(["-ar", "44100"])
    cmd.extend(["-f", out_fmt, "-codec:a", codec, "pipe:1"])
    
    r_fd, w_fd = os.pipe()
    proc = subprocess.Popen(cmd, stdin=os.fdopen(r_fd, "rb"), stdout=subprocess.PIPE)
    return proc, w_fd


def _start_pipe_writer(queue: Queue, write_fd: int) -> threading.Thread:
    def writer():
        try:
            with os.fdopen(write_fd, "wb") as pipe:
                while (data := queue.get()) is not None:
                    try: pipe.write(data)
                    except: break
        except: pass
    
    t = threading.Thread(target=writer, daemon=True)
    t.start()
    return t


# ============================================================================
# AUDIO GENERATION WITH CACHING
# ============================================================================
async def _generate_audio_core(text: str, voice_name: str, speed: float, fmt: str) -> AsyncIterator[bytes]:
    queue = Queue(maxsize=QUEUE_SIZE)
    producer = _start_audio_producer(queue, voice_name, text)

    try:
        if fmt in ("wav", "pcm"):
            async for chunk in _stream_queue_chunks(queue): yield chunk
            producer.join()
            return

        if fmt in FFMPEG_FORMATS:
            proc, w_fd = _start_ffmpeg_process(fmt)
            writer = _start_pipe_writer(queue, w_fd)
            
            try:
                while (chunk := await asyncio.to_thread(proc.stdout.read, CHUNK_SIZE)):
                    yield chunk
            finally:
                proc.wait()
                producer.join()
                writer.join()
            return

        async for chunk in _stream_queue_chunks(queue): yield chunk
        producer.join()
    except Exception:
        logger.exception("Streaming error")
        raise


async def generate_audio(text: str, voice: str = "alloy", speed: float = 1.0, fmt: str = "mp3") -> AsyncIterator[bytes]:
    if tts_model is None:
        raise HTTPException(503, "TTS model not loaded")

    voice_name = VOICE_MAPPING.get(voice, voice)
    
    # Cache key
    cache_hash = hashlib.md5(f"{text}|{voice_name}|{fmt}|{speed}".encode()).hexdigest()
    cache_path = os.path.join(AUDIO_CACHE_DIR, f"{cache_hash}.{fmt}")
    
    # Check cache
    if os.path.exists(cache_path):
        logger.info(f"Cache HIT: {cache_hash[:8]}")
        async with await open_file(cache_path, "rb") as f:
            while (chunk := await f.read(CHUNK_SIZE)): yield chunk
        return

    # Generate and cache
    logger.info(f"Cache MISS: {cache_hash[:8]} - Generating...")
    temp_path = f"{cache_path}.tmp"
    
    try:
        async with await open_file(temp_path, "wb") as cache_file:
            async for chunk in _generate_audio_core(text, voice_name, speed, fmt):
                await cache_file.write(chunk)
                yield chunk
        
        if os.path.exists(temp_path):
            os.replace(temp_path, cache_path)
            asyncio.create_task(cleanup_cache())
    except:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise


async def cleanup_cache():
    def _cleanup():
        try:
            files = [(os.path.join(AUDIO_CACHE_DIR, f), os.path.getmtime(os.path.join(AUDIO_CACHE_DIR, f)))
                     for f in os.listdir(AUDIO_CACHE_DIR) if f.endswith(tuple(MEDIA_TYPES.keys()))]
            if len(files) <= CACHE_LIMIT: return
            for path, _ in sorted(files, key=lambda x: x[1])[:len(files) - CACHE_LIMIT]:
                try: os.remove(path)
                except: pass
        except: pass
    await asyncio.to_thread(_cleanup)


# ============================================================================
# FASTAPI APP
# ============================================================================
@asynccontextmanager
async def lifespan(app):
    logger.info(f"{Colors.BOLD}🚀 Starting Pocket TTS API (GPU Enhanced){Colors.RESET}")
    load_tts_model()
    yield

app = FastAPI(
    title="Pocket TTS OpenAPI",
    description="GPU-accelerated OpenAI-compatible TTS API",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def home():
    return f"""
    <html><head><title>Pocket TTS API</title>
    <style>body{{font-family:system-ui;max-width:800px;margin:40px auto;padding:20px;background:#1a1a2e;color:#eee}}
    a{{color:#00d9ff}}h1{{color:#00d9ff}}code{{background:#333;padding:2px 6px;border-radius:4px}}</style></head>
    <body><h1>🎤 Pocket TTS OpenAPI</h1>
    <p><b>Device:</b> <code>{device_str}</code> | <b>Model:</b> <code>{MODEL_REPO}</code></p>
    <p><b>Endpoints:</b></p>
    <ul>
    <li><code>POST /v1/audio/speech</code> - Generate speech</li>
    <li><code>GET /v1/voices</code> - List voices</li>
    <li><code>GET /v1/models</code> - List models</li>
    <li><code>GET /health</code> - Health check</li>
    <li><a href="/docs">Swagger UI</a> | <a href="/redoc">ReDoc</a></li>
    </ul></body></html>
    """


@app.post("/v1/audio/speech")
async def text_to_speech(request: SpeechRequest) -> StreamingResponse:
    logger.info(f"Request: voice={request.voice}, format={request.response_format}, len={len(request.input)}")
    return StreamingResponse(
        generate_audio(request.input, request.voice, request.speed, request.response_format),
        media_type=MEDIA_TYPES.get(request.response_format, "audio/mpeg"),
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": tts_model is not None,
        "warmup_complete": _warmup_complete,
        "requests_served": _request_count,
        "avg_inference_time": round(_total_inference_time / max(1, _request_count), 3),
        "alignment_available": ALIGNMENT_AVAILABLE,
        "alignment_method": "mfa" if ALIGNMENT_AVAILABLE else "estimation",
        **optimization_info,
    }


@app.get("/v1/voices")
async def list_voices():
    custom = [k for k in VOICE_MAPPING if k not in DEFAULT_VOICES['openai_aliases'] and k not in DEFAULT_VOICES['pocket_tts']]
    return {"openai": DEFAULT_VOICES['openai_aliases'], "pocket_tts": DEFAULT_VOICES['pocket_tts'], "custom": custom}


@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": "tts-1", "object": "model"}, {"id": "pocket-tts", "object": "model"}]}


# ============================================================================
# SPEECH WITH ALIGNMENT (Unified Endpoint for Remotion)
# Returns audio + word-level timestamps in single response
# ============================================================================

class SpeechWithAlignmentRequest(BaseModel):
    """Request for unified TTS + alignment endpoint."""
    model: Literal["tts-1", "tts-1-hd", "pocket-tts"] = Field("tts-1")
    input: str = Field(..., min_length=1, max_length=4096, description="Text to synthesize")
    voice: str = Field("alloy", description="Voice ID or path to WAV for cloning")
    speed: Optional[float] = Field(1.0, ge=0.25, le=4.0)
    # Alignment options
    fps: int = Field(30, ge=1, le=120, description="Frames per second for frame calculations")
    words_per_page: int = Field(6, ge=1, le=20, description="Words per TikTok caption page")


class RemotionCaption(BaseModel):
    """Single caption in Remotion format."""
    text: str
    startMs: int  # milliseconds
    endMs: int    # milliseconds
    startFrame: Optional[int] = None
    endFrame: Optional[int] = None


class TikTokToken(BaseModel):
    """Token for TikTok-style word highlighting."""
    text: str
    fromMs: int
    toMs: int


class TikTokPage(BaseModel):
    """Page of tokens for TikTok-style captions."""
    text: str
    startMs: int
    durationMs: int
    tokens: list[TikTokToken]


class SpeechWithAlignmentResponse(BaseModel):
    """Response with audio + Remotion-compatible captions."""
    audio_base64: str  # Base64 encoded WAV audio
    audio_duration_ms: int
    sample_rate: int
    # Standard Remotion format
    captions: list[RemotionCaption]
    timeline: list[dict]
    # TikTok-style format for word highlighting
    pages: list[TikTokPage]


def estimate_word_timestamps(text: str, audio_duration_ms: int, fps: int = 30) -> list[RemotionCaption]:
    """
    Estimate word-level timestamps based on character distribution.
    This is a fallback when MFA is not available.
    Uses proportional timing based on word length.
    """
    words = text.split()
    if not words:
        return []
    
    # Calculate total "weight" (character count)
    total_chars = sum(len(w) for w in words)
    if total_chars == 0:
        return []
    
    captions = []
    current_ms = 0
    
    for word in words:
        # Proportional duration based on word length
        word_weight = len(word) / total_chars
        word_duration_ms = int(audio_duration_ms * word_weight)
        
        # Minimum 50ms per word
        word_duration_ms = max(50, word_duration_ms)
        
        end_ms = min(current_ms + word_duration_ms, audio_duration_ms)
        
        captions.append(RemotionCaption(
            text=word,
            startMs=current_ms,
            endMs=end_ms,
            startFrame=int((current_ms / 1000) * fps),
            endFrame=int((end_ms / 1000) * fps),
        ))
        
        current_ms = end_ms
    
    return captions


def run_forced_alignment(audio_path: str, text: str, fps: int = 30) -> list[RemotionCaption]:
    """
    Run Montreal Forced Aligner (via pyfoal) for precise word timestamps.
    Falls back to estimation if MFA not available.
    """
    if not ALIGNMENT_AVAILABLE:
        # Get audio duration for fallback
        try:
            info = sf.info(audio_path)
            duration_ms = int(info.duration * 1000)
            return estimate_word_timestamps(text, duration_ms, fps)
        except:
            return []
    
    try:
        # pyfoal returns alignment with word boundaries
        alignment = pyfoal.from_file(text, audio_path)
        
        captions = []
        for word_info in alignment.words():
            start_ms = int(word_info.start * 1000)
            end_ms = int(word_info.end * 1000)
            
            captions.append(RemotionCaption(
                text=word_info.word,
                startMs=start_ms,
                endMs=end_ms,
                startFrame=int((start_ms / 1000) * fps),
                endFrame=int((end_ms / 1000) * fps),
            ))
        
        return captions
    except Exception as e:
        logger.warning(f"MFA alignment failed, using estimation: {e}")
        try:
            info = sf.info(audio_path)
            duration_ms = int(info.duration * 1000)
            return estimate_word_timestamps(text, duration_ms, fps)
        except:
            return []


@app.post("/v1/audio/speech-with-alignment", response_model=SpeechWithAlignmentResponse)
async def speech_with_alignment(request: SpeechWithAlignmentRequest):
    """
    Generate speech and return audio + word-level timestamps.
    
    Returns a JSON response with:
    - audio_base64: WAV audio as base64 string
    - captions: Word-level timestamps in Remotion Caption format
    - timeline: Simplified timeline for scene syncing
    
    Compatible with Remotion's Caption component.
    """
    import base64
    
    if tts_model is None:
        raise HTTPException(503, "TTS model not loaded")
    
    voice_name = VOICE_MAPPING.get(request.voice, request.voice)
    
    # Generate audio to a temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Collect all audio chunks
        audio_chunks = []
        async for chunk in _generate_audio_core(request.input, voice_name, request.speed, "wav"):
            audio_chunks.append(chunk)
        
        # Write to temp file
        with open(tmp_path, "wb") as f:
            for chunk in audio_chunks:
                f.write(chunk)
        
        # Get audio info
        info = sf.info(tmp_path)
        duration_ms = int(info.duration * 1000)
        
        # Run forced alignment
        captions = await asyncio.to_thread(
            run_forced_alignment, tmp_path, request.input, request.fps
        )
        
        # Read audio as base64
        with open(tmp_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Create simplified timeline
        timeline = [{"text": c.text, "startMs": c.startMs, "endMs": c.endMs} for c in captions]
        
        # Generate TikTok-style pages from captions
        pages = []
        words_per_page = request.words_per_page
        
        for i in range(0, len(captions), words_per_page):
            page_captions = captions[i:i + words_per_page]
            if not page_captions:
                continue
            
            page_text = " ".join(c.text for c in page_captions)
            page_start_ms = page_captions[0].startMs
            page_end_ms = page_captions[-1].endMs
            
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
                startMs=page_start_ms,
                durationMs=page_end_ms - page_start_ms,
                tokens=tokens,
            ))
        
        return SpeechWithAlignmentResponse(
            audio_base64=audio_base64,
            audio_duration_ms=duration_ms,
            sample_rate=info.samplerate,
            captions=captions,
            timeline=timeline,
            pages=pages,
        )
    
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/v1/audio/align")
async def align_audio(
    text: str = Field(..., description="Transcript text"),
    audio_path: str = Field(..., description="Path to audio file"),
    fps: int = Field(30, description="Frames per second"),
):
    """
    Align existing audio with transcript.
    Returns word-level timestamps in Remotion format.
    """
    if not os.path.exists(audio_path):
        raise HTTPException(404, f"Audio file not found: {audio_path}")
    
    captions = await asyncio.to_thread(run_forced_alignment, audio_path, text, fps)
    
    info = sf.info(audio_path)
    
    return {
        "captions": [c.dict() for c in captions],
        "timeline": [{"text": c.text, "startMs": c.startMs, "endMs": c.endMs} for c in captions],
        "audio_duration_ms": int(info.duration * 1000),
        "alignment_method": "mfa" if ALIGNMENT_AVAILABLE else "estimation",
    }


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import socket
    
    def find_port(start=8001, retries=20):
        for p in range(start, start + retries):
            try:
                with socket.socket() as s: s.bind(("0.0.0.0", p)); return p
            except: continue
        raise RuntimeError("No free port")

    port = find_port()
    logger.info(f"{Colors.GREEN}✅ http://0.0.0.0:{port}{Colors.RESET}")
    uvicorn.run(app, host="0.0.0.0", port=port)
