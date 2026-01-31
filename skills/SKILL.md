---
name: pocket-tts-gpu
description: Generate TTS audio with Remotion-compatible TikTok-style captions using Pocket TTS GPU server. Supports voice cloning, Whisper-based word-level timestamps, and OpenAI API compatibility. Free local alternative to ElevenLabs.
triggers:
  - pocket tts
  - tts with captions
  - voice cloning
  - tiktok captions
  - remotion narration
  - generate narration
  - text to speech local
  - whisper captions
---

# Pocket TTS GPU Skill

Generate high-quality text-to-speech with **Whisper-based word-level timestamps** for Remotion TikTok captions. A **free, local alternative to ElevenLabs**.

## Server Setup

```bash
git clone https://github.com/siva-sub/pocket-tts-openapi-gpu
cd pocket-tts-openapi-gpu
chmod +x install_gpu.sh && ./install_gpu.sh
source .venv/bin/activate
python pocketapi.py  # Runs at http://localhost:8001
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/audio/speech` | OpenAI-compatible TTS |
| `POST /v1/audio/speech-with-whisper` | TTS + Whisper timestamps (**recommended**) |
| `POST /v1/audio/speech-with-alignment` | TTS + proportional timestamps (fast) |

## Generate Narration + TikTok Captions (Whisper)

```python
import requests, base64, json

response = requests.post("http://localhost:8001/v1/audio/speech-with-whisper", json={
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
    json.dump({"pages": data["pages"], "captions": data["captions"]}, f)
```

## Response Format

```json
{
  "audio_base64": "UklGR...",
  "audio_duration_ms": 119840,
  "alignment_method": "whisper",
  "pages": [{
    "text": "wire five hundred dollars to a",
    "startMs": 1830,
    "durationMs": 1644,
    "tokens": [
      {"text": "wire ", "fromMs": 1830, "toMs": 2100},
      {"text": "five ", "fromMs": 2100, "toMs": 2360}
    ]
  }],
  "captions": [
    {"text": "wire", "startMs": 1830, "endMs": 2100, "confidence": 0.99}
  ]
}
```

## Voice Cloning

```bash
ffmpeg -i reference.mp3 -ar 24000 -ac 1 voices/myvoice.wav
# Use: {"voice": "myvoice"}
```

## OpenAI API Compatibility

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed")
response = client.audio.speech.create(model="tts-1", voice="alloy", input="Hello")
response.stream_to_file("output.mp3")
```

## Standalone Whisper Alignment

```bash
python whisper_align.py audio.wav -o captions.json
```

## Features

- 🚀 **GPU Acceleration**: ~1.7x faster than realtime
- 🎤 **Voice Cloning**: 10-15s reference audio
- 📝 **Whisper Captions**: Accurate word-level timestamps
- 🔌 **OpenAI Compatible**: Drop-in replacement
- 🔒 **100% Local**: No cloud, no API keys, free

## Credits

Built on [Kyutai Labs' Pocket TTS](https://github.com/kyutai-labs/pocket-tts) with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## GitHub

https://github.com/siva-sub/pocket-tts-openapi-gpu
