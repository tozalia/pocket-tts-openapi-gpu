---
name: pocket-tts-gpu
description: Generate TTS audio with Remotion-compatible TikTok-style captions using Pocket TTS GPU server. Supports voice cloning, word-level timestamps, and OpenAI API compatibility. Free local alternative to ElevenLabs.
triggers:
  - pocket tts
  - tts with captions
  - voice cloning
  - tiktok captions
  - remotion narration
  - generate narration
  - text to speech local
---

# Pocket TTS GPU Skill

Generate high-quality text-to-speech with word-level timestamps for Remotion TikTok captions. A **free, local alternative to ElevenLabs**.

## Server Setup

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

## Voice Cloning

```bash
# Convert reference audio (10-15 seconds, clear speech)
ffmpeg -i reference.mp3 -ar 24000 -ac 1 voices/myvoice.wav
# Use filename as voice parameter: {"voice": "myvoice"}
```

## Generate Narration + TikTok Captions

```python
import requests, base64, json

response = requests.post("http://localhost:8001/v1/audio/speech-with-alignment", json={
    "input": "Your script text here",
    "voice": "myvoice",  # Custom cloned voice or "alloy"
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

## Response Format (Remotion TikTokPage)

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
  }],
  "captions": [
    {"text": "wire", "startMs": 1776, "endMs": 2039}
  ]
}
```

## Remotion Usage

```tsx
import captions from '../../../public/captions.json';

const page = captions.pages.find(p => 
  currentMs >= p.startMs && currentMs < p.startMs + p.durationMs
);

{page?.tokens.map(t => (
  <span style={{ color: currentMs >= t.fromMs ? '#fff' : '#666' }}>
    {t.text}
  </span>
))}
```

## OpenAI API Compatibility

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed")
response = client.audio.speech.create(model="tts-1", voice="alloy", input="Hello")
response.stream_to_file("output.mp3")
```

## MFA for Higher Accuracy

```bash
python forced_align.py --audio narration.wav --script script.txt -o captions.json
```

## Features

- 🚀 **GPU Acceleration**: ~1.7x faster than realtime
- 🎤 **Voice Cloning**: 10-15s reference audio
- 📝 **TikTok Captions**: Remotion `createTikTokStyleCaptions()` compatible
- 🔌 **OpenAI Compatible**: Drop-in replacement
- 🔒 **100% Local**: No cloud, no API keys, free
- ⚡ **Voice Caching**: Fast subsequent requests

## Credits

Built on [Kyutai Labs' Pocket TTS](https://github.com/kyutai-labs/pocket-tts) with [Montreal Forced Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) integration.

## GitHub

https://github.com/siva-sub/pocket-tts-openapi-gpu
