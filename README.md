# Pocket TTS OpenAPI - GPU Enhanced Edition

GPU-accelerated text-to-speech API with OpenAI compatibility, voice cloning, and **Remotion TikTok-style captions**.

## Features

- 🚀 **GPU Acceleration**: CUDA support with ~1.7x faster than realtime
- 🎤 **Voice Cloning**: Clone any voice from 10-15 second reference audio (ICL)
- 📝 **TikTok-Style Captions**: Word-level timestamps for Remotion videos
- 🔌 **OpenAI Compatible**: Drop-in replacement for OpenAI TTS API
- 🎯 **MFA Integration**: Montreal Forced Aligner for precise alignment
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

### TikTok-Style Captions

The API returns `pages` in [Remotion's TikTokPage format](https://www.remotion.dev/docs/captions/create-tiktok-style-captions):

```typescript
interface TikTokPage {
  text: string;      // "wire five hundred dollars to a"
  startMs: number;   // 1776
  durationMs: number; // 1644
  tokens: {
    text: string;    // "five "
    fromMs: number;  // 2039
    toMs: number;    // 2302
  }[];
}
```

### Complete Workflow

```python
import requests
import base64
import json

# Generate narration + captions
response = requests.post("http://localhost:8001/v1/audio/speech-with-alignment", json={
    "input": "Your full script here...",
    "voice": "siva",
    "fps": 30,
    "words_per_page": 6,
})

data = response.json()

# Save audio for Remotion
with open("public/narration.wav", "wb") as f:
    f.write(base64.b64decode(data["audio_base64"]))

# Save captions for TikTokCaptions component
with open("public/captions.json", "w") as f:
    json.dump({"pages": data["pages"]}, f)
```

### Remotion Component Usage

```tsx
import captionData from '../../../public/captions.json';

interface TikTokPage {
  text: string;
  startMs: number;
  durationMs: number;
  tokens: { text: string; fromMs: number; toMs: number }[];
}

const pages = captionData.pages as TikTokPage[];

export const TikTokCaptions: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const currentMs = (frame / fps) * 1000;

  const activePage = pages.find(
    p => currentMs >= p.startMs && currentMs < p.startMs + p.durationMs
  );

  if (!activePage) return null;

  return (
    <div style={{ textAlign: 'center', fontSize: 48 }}>
      {activePage.tokens.map((token, i) => {
        const isActive = currentMs >= token.fromMs && currentMs < token.toMs;
        return (
          <span key={i} style={{ 
            color: isActive ? '#fff' : 'rgba(255,255,255,0.4)',
            textShadow: isActive ? '0 0 20px #C19A8A' : 'none'
          }}>
            {token.text}
          </span>
        );
      })}
    </div>
  );
};
```

## Voice Cloning

```bash
# Convert reference audio (10-15 seconds, clear speech)
ffmpeg -i reference.mp3 -ar 24000 -ac 1 voices/myvoice.wav

# Use voice name in API
curl -X POST http://localhost:8001/v1/audio/speech-with-alignment \
  -d '{"input": "Hello world", "voice": "myvoice"}'
```

## API Endpoints

### `/v1/audio/speech-with-alignment` (Recommended)

Returns audio + TikTok-style caption pages in single response.

**Request:**
```json
{
  "input": "Your script text...",
  "voice": "siva",
  "fps": 30,
  "words_per_page": 6
}
```

**Response:**
```json
{
  "audio_base64": "UklGR...",
  "audio_duration_ms": 119840,
  "sample_rate": 24000,
  "pages": [...]
}
```

### `/v1/audio/speech` (OpenAI Compatible)

Standard OpenAI TTS API replacement.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed")
response = client.audio.speech.create(model="tts-1", voice="alloy", input="Hello")
response.stream_to_file("output.mp3")
```

## MFA Forced Alignment

For existing audio + script:

```bash
python forced_align.py --audio narration.wav --script script.txt -o captions.json
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POCKET_TTS_DEVICE` | `auto` | Device: `auto`, `cuda`, `cpu` |
| `POCKET_TTS_WORKERS` | `4` | Max concurrent requests |
| `POCKET_TTS_MODEL` | `Verylicious/pocket-tts-ungated` | HuggingFace model |

## Performance

| Metric | Value |
|--------|-------|
| First request | ~5-10s (voice encoding + warmup) |
| Generation speed | 1.5-2x realtime on RTX GPUs |
| Voice cache | Eliminates repeat encoding overhead |
| Sample rate | 24kHz |

## License

MIT - Based on [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts)
