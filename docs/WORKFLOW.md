# Pocket TTS GPU - Remotion Workflow

Complete workflow for generating narration with TikTok-style captions for Remotion videos.

## Prerequisites

```bash
cd pocket-tts-openapi-gpu
source .venv/bin/activate && python pocketapi.py
# Server runs at http://localhost:8001
```

## Step 1: Voice Setup (Optional)

For voice cloning, convert your reference audio:

```bash
ffmpeg -i reference.mp3 -ar 24000 -ac 1 voices/myvoice.wav
```

## Step 2: Generate Narration + Captions

```python
import requests, base64, json

SCRIPT = """Your full narration script here..."""

response = requests.post("http://localhost:8001/v1/audio/speech-with-alignment", json={
    "input": SCRIPT,
    "voice": "myvoice",  # or "alloy" for default
    "fps": 30,
    "words_per_page": 6,
}, timeout=300)

data = response.json()
print(f"Duration: {data['audio_duration_ms']/1000:.1f}s, Pages: {len(data['pages'])}")

# Save for Remotion
with open("remotion/public/narration.wav", "wb") as f:
    f.write(base64.b64decode(data["audio_base64"]))

with open("remotion/public/captions.json", "w") as f:
    json.dump({"pages": data["pages"]}, f, indent=2)
```

## Step 3: Use in Remotion Component

```tsx
import captions from '../../../public/captions.json';

interface TikTokPage {
  text: string;
  startMs: number;
  durationMs: number;
  tokens: { text: string; fromMs: number; toMs: number }[];
}

const pages = captions.pages as TikTokPage[];

export const TikTokCaptions: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const currentMs = (frame / fps) * 1000;

  const page = pages.find(p => 
    currentMs >= p.startMs && currentMs < p.startMs + p.durationMs
  );

  if (!page) return null;

  return (
    <div style={{ textAlign: 'center', fontSize: 48 }}>
      {page.tokens.map((token, i) => {
        const isActive = currentMs >= token.fromMs && currentMs < token.toMs;
        return (
          <span key={i} style={{ 
            color: isActive ? '#fff' : 'rgba(255,255,255,0.4)',
            fontWeight: 900,
          }}>
            {token.text}
          </span>
        );
      })}
    </div>
  );
};
```

## Step 4: Render Video

```bash
cd remotion && npx remotion render YourComposition --output=out/final.mp4
```

## Optional: MFA for Higher Accuracy

For existing audio files, use Montreal Forced Aligner:

```bash
python forced_align.py --audio narration.wav --script script.txt -o captions.json
```

## Response Format Reference

```json
{
  "audio_base64": "UklGR...",
  "audio_duration_ms": 119840,
  "sample_rate": 24000,
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
    {"text": "wire", "startMs": 1776, "endMs": 2039},
    {"text": "five", "startMs": 2039, "endMs": 2302}
  ]
}
```
