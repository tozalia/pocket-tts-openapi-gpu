# Pocket TTS OpenAPI - GPU Enhanced Edition

GPU-accelerated text-to-speech API with OpenAI compatibility, voice cloning, and word-level timestamps for Remotion video captions.

## Features

- 🚀 **GPU Acceleration**: CUDA support with ~1.7x faster than realtime
- 🎤 **Voice Cloning**: Clone any voice from a 10-15 second reference audio  
- 📝 **Word-Level Timestamps**: TikTok-style captions for Remotion videos
- 🔌 **OpenAI Compatible**: Drop-in replacement for OpenAI TTS API
- 🎯 **Multiple Alignment Methods**: MFA, Whisper, or proportional estimation
- ⚡ **Voice State Caching**: Fast subsequent requests for same voice

## Quick Start

```bash
# Clone the repository
git clone https://github.com/siva-sub/pocket-tts-openapi-gpu
cd pocket-tts-openapi-gpu

# Install with MFA support
chmod +x install_gpu.sh
./install_gpu.sh

# Start the server
source .venv/bin/activate
python pocketapi.py
```

Server runs at `http://localhost:8001`

## API Endpoints

### OpenAI-Compatible TTS
```bash
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "alloy"}' \
  --output speech.mp3
```

### TTS + Word-Level Alignment (Recommended)
```bash
curl -X POST http://localhost:8001/v1/audio/speech-with-alignment \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Your narration text here",
    "voice": "siva",
    "fps": 30,
    "words_per_page": 6
  }' | jq
```

Returns:
```json
{
  "audio_base64": "UklGR...",
  "audio_duration_ms": 5000,
  "sample_rate": 24000,
  "captions": [{"text": "Your", "startMs": 0, "endMs": 200, ...}],
  "pages": [{"text": "Your narration text", "startMs": 0, "tokens": [...]}]
}
```

## Voice Cloning

1. Place your reference audio (10-15 seconds, clear speech) in `voices/`:
   ```bash
   ffmpeg -i reference.mp3 -ar 24000 -ac 1 voices/myvoice.wav
   ```

2. Use the filename as the voice parameter:
   ```json
   {"input": "Hello", "voice": "myvoice"}
   ```

## Forced Alignment with MFA

For highest accuracy word-level timestamps:

```bash
# Align existing audio with script
python forced_align.py \
  --audio narration.wav \
  --script script.txt \
  --output captions.json \
  --method mfa
```

Alignment methods:
- `mfa` - Montreal Forced Aligner (best accuracy, requires conda)
- `whisper` - OpenAI Whisper + ground-truth mapping
- `estimate` - Proportional estimation (fastest, no dependencies)

## Remotion Integration

The `pages` format is directly compatible with TikTokCaptions:

```tsx
import captionsData from '../../../public/captions.json';

interface CaptionPage {
  text: string;
  startMs: number;
  durationMs: number;
  tokens: { text: string; fromMs: number; toMs: number; }[];
}

const pages = captionsData.pages as CaptionPage[];
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `POCKET_TTS_DEVICE` | `auto` | Device: `auto`, `cuda`, `cpu` |
| `POCKET_TTS_WORKERS` | `4` | Max concurrent requests |
| `POCKET_TTS_COMPILE` | `false` | Enable torch.compile() |
| `POCKET_TTS_MODEL` | `Verylicious/pocket-tts-ungated` | HuggingFace model |

## Performance

- First request: ~5-10s (voice encoding + warmup)
- Subsequent requests: Audio generated at 1.5-2x realtime on RTX GPUs
- Voice state caching eliminates repeat encoding overhead

## License

MIT - Based on [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts)
