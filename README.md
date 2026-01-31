# Pocket TTS OpenAPI - GPU Enhanced Edition

**A free, fully local alternative to ElevenLabs** with natural voice cloning, GPU acceleration, and Remotion TikTok-style captions.

> 🎤 **Drop-in replacement for OpenAI TTS API** with voice cloning. Runs entirely on your device. Costs nothing.

## Why Pocket TTS?

| Feature | ElevenLabs | Pocket TTS GPU |
|---------|------------|----------------|
| **Cost** | $5-330/month | **Free** |
| **Privacy** | Cloud-based | **100% Local** |
| **Voice Cloning** | ✅ | ✅ (10-15s sample) |
| **Natural English** | ✅ | ✅ |
| **API Compatible** | Custom API | **OpenAI Compatible** |
| **Remotion Integration** | Manual | **Built-in TikTok Captions** |

## Features

- 🚀 **GPU Acceleration**: CUDA support with ~1.7x faster than realtime
- 🎤 **Voice Cloning**: Clone any voice from 10-15 second reference audio
- 📝 **TikTok-Style Captions**: Word-level timestamps for Remotion videos
- 🔌 **OpenAI Compatible**: Drop-in replacement for `/v1/audio/speech`
- 🔒 **100% Local**: No cloud, no API keys, no usage fees
- ⚡ **Voice State Caching**: Fast subsequent requests for same voice

## Quick Start

```bash
# Clone and install
git clone https://github.com/siva-sub/pocket-tts-openapi-gpu
cd pocket-tts-openapi-gpu
chmod +x install_gpu.sh && ./install_gpu.sh

# Start server
source .venv/bin/activate
python pocketapi.py
# Runs at http://localhost:8001
```

## Remotion Integration

The API returns `pages` in [Remotion's TikTokPage format](https://www.remotion.dev/docs/captions/create-tiktok-style-captions):

```python
import requests, base64, json

response = requests.post("http://localhost:8001/v1/audio/speech-with-alignment", json={
    "input": "Your script text here",
    "voice": "myvoice",
    "fps": 30,
    "words_per_page": 6,
})

data = response.json()

# Save for Remotion
with open("public/narration.wav", "wb") as f:
    f.write(base64.b64decode(data["audio_base64"]))
with open("public/captions.json", "w") as f:
    json.dump({"pages": data["pages"]}, f)
```

### Response Format

```json
{
  "audio_base64": "UklGR...",
  "audio_duration_ms": 119840,
  "pages": [{
    "text": "wire five hundred dollars to a",
    "startMs": 1776,
    "durationMs": 1644,
    "tokens": [
      {"text": "wire ", "fromMs": 1776, "toMs": 2039},
      {"text": "five ", "fromMs": 2039, "toMs": 2302}
    ]
  }]
}
```

## Voice Cloning

```bash
# Convert reference audio (10-15 seconds, clear speech)
ffmpeg -i reference.mp3 -ar 24000 -ac 1 voices/myvoice.wav

# Use voice name in API
{"voice": "myvoice"}
```

## OpenAI API Compatibility

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed")
response = client.audio.speech.create(model="tts-1", voice="alloy", input="Hello world")
response.stream_to_file("output.mp3")
```

## Performance

| Metric | Value |
|--------|-------|
| Generation speed | 1.5-2x realtime (RTX GPUs) |
| First request | ~5-10s (voice encoding) |
| Cached voice | Instant |
| Sample rate | 24kHz |

## Credits & Acknowledgments

### Core Model

**[Kyutai Labs](https://github.com/kyutai-labs)** - Creators of [Pocket TTS](https://github.com/kyutai-labs/pocket-tts), a 100M parameter lightweight TTS model with in-context learning for voice cloning. This project wraps their model with a FastAPI server and Remotion integration.

### Reference Implementations

- **[Qwen3-TTS-OpenAI-FastAPI](https://github.com/Verylicious/Qwen3-TTS-Openai-Fastapi)** - Inspiration for OpenAI-compatible API structure
- **[pocket-tts-ungated](https://huggingface.co/Verylicious/pocket-tts-ungated)** - Ungated HuggingFace model used for inference
- **[Montreal Forced Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner)** - Word-level alignment for precise captions

### Remotion Integration

- **[Remotion](https://www.remotion.dev/)** - Programmatic video framework
- **[@remotion/captions](https://www.remotion.dev/docs/captions)** - TikTok-style caption format specification

## License

MIT - Built on top of [Kyutai Labs' Pocket TTS](https://github.com/kyutai-labs/pocket-tts)
