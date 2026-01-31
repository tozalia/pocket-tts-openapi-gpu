#!/usr/bin/env python3
"""
Montreal Forced Aligner Integration for Pocket TTS
===================================================
Uses MFA (Montreal Forced Aligner) CLI to generate word-level timestamps
from audio + transcript for TikTok-style captions in Remotion videos.

Prerequisites:
    conda create -n mfa -c conda-forge montreal-forced-aligner
    conda activate mfa
    mfa model download acoustic english_us_arpa
    mfa model download dictionary english_us_arpa

Usage:
    python forced_align.py --audio narration.wav --script script.txt --output captions.json
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional


def clean_text(text: str) -> str:
    """Remove punctuation for fuzzy matching."""
    return re.sub(r'[^a-zA-Z0-9]', '', text).lower()


def run_mfa_align(audio_path: str, script_text: str, fps: int = 30) -> List[Dict]:
    """
    Run Montreal Forced Aligner CLI to get word-level alignments.
    
    MFA outputs TextGrid files which we parse for word timings.
    """
    audio_path = Path(audio_path).resolve()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_dir = Path(tmpdir) / "corpus"
        output_dir = Path(tmpdir) / "output"
        corpus_dir.mkdir()
        output_dir.mkdir()
        
        # MFA expects: corpus/basename.wav and corpus/basename.txt
        basename = audio_path.stem
        
        # Copy audio to corpus
        shutil.copy(audio_path, corpus_dir / f"{basename}.wav")
        
        # Write transcript (MFA expects plain text, one utterance)
        (corpus_dir / f"{basename}.txt").write_text(script_text.strip())
        
        # Run MFA align
        cmd = [
            "mfa", "align",
            str(corpus_dir),
            "english_us_arpa",  # Dictionary
            "english_us_arpa",  # Acoustic model
            str(output_dir),
            "--clean",
            "--single_speaker"
        ]
        
        # Try running via conda environment
        env = os.environ.copy()
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            # Try running via conda run
            cmd = ["conda", "run", "-n", "mfa"] + cmd[0:]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"MFA failed: {result.stderr}")
        
        # Parse TextGrid output
        textgrid_path = output_dir / f"{basename}.TextGrid"
        if not textgrid_path.exists():
            # Check subdirectories
            for f in output_dir.rglob("*.TextGrid"):
                textgrid_path = f
                break
        
        if not textgrid_path.exists():
            raise RuntimeError(f"MFA didn't produce TextGrid. Output: {result.stdout}")
        
        return parse_textgrid(textgrid_path, fps)


def parse_textgrid(textgrid_path: Path, fps: int = 30) -> List[Dict]:
    """Parse Praat TextGrid file to extract word timings."""
    try:
        import textgrid
        tg = textgrid.TextGrid.fromFile(str(textgrid_path))
        
        captions = []
        for tier in tg.tiers:
            if tier.name.lower() == "words":
                for interval in tier:
                    if interval.mark and interval.mark.strip():
                        start_ms = int(interval.minTime * 1000)
                        end_ms = int(interval.maxTime * 1000)
                        captions.append({
                            "text": interval.mark,
                            "startMs": start_ms,
                            "endMs": end_ms,
                            "startFrame": int((start_ms / 1000) * fps),
                            "endFrame": int((end_ms / 1000) * fps),
                        })
        
        return captions
        
    except ImportError:
        # Manual TextGrid parsing
        return parse_textgrid_manual(textgrid_path, fps)


def parse_textgrid_manual(textgrid_path: Path, fps: int) -> List[Dict]:
    """Manual TextGrid parsing without textgrid package."""
    content = textgrid_path.read_text()
    captions = []
    
    # Simple regex-based parsing for word tier
    # TextGrid format has intervals with xmin, xmax, text
    in_words = False
    lines = content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if '"words"' in line.lower() or 'name = "words"' in line.lower():
            in_words = True
        
        if in_words and 'xmin' in line:
            xmin = float(re.search(r'[\d.]+', line).group())
            i += 1
            xmax = float(re.search(r'[\d.]+', lines[i]).group())
            i += 1
            text_match = re.search(r'"([^"]*)"', lines[i])
            text = text_match.group(1) if text_match else ""
            
            if text and text.strip():
                start_ms = int(xmin * 1000)
                end_ms = int(xmax * 1000)
                captions.append({
                    "text": text,
                    "startMs": start_ms,
                    "endMs": end_ms,
                    "startFrame": int((start_ms / 1000) * fps),
                    "endFrame": int((end_ms / 1000) * fps),
                })
        
        i += 1
    
    return captions


def estimate_timestamps(script_text: str, duration_ms: int, fps: int = 30) -> List[Dict]:
    """Fallback: proportional estimation based on character count."""
    words = script_text.strip().split()
    if not words:
        return []
    
    total_chars = sum(len(w) for w in words)
    captions = []
    current_ms = 0
    
    for word in words:
        word_weight = len(word) / total_chars
        word_duration = max(50, int(duration_ms * word_weight))
        end_ms = min(current_ms + word_duration, duration_ms)
        
        captions.append({
            "text": word,
            "startMs": current_ms,
            "endMs": end_ms,
            "startFrame": int((current_ms / 1000) * fps),
            "endFrame": int((end_ms / 1000) * fps),
        })
        current_ms = end_ms
    
    return captions


def captions_to_pages(captions: List[Dict], words_per_page: int = 6) -> List[Dict]:
    """Convert word captions to TikTok-style pages."""
    pages = []
    
    for i in range(0, len(captions), words_per_page):
        page_chunk = captions[i:i + words_per_page]
        if not page_chunk:
            continue
        
        text = " ".join(c['text'] for c in page_chunk)
        pages.append({
            "text": text,
            "startMs": page_chunk[0]['startMs'],
            "durationMs": page_chunk[-1]['endMs'] - page_chunk[0]['startMs'],
            "tokens": [
                {"text": c['text'] + " ", "fromMs": c['startMs'], "toMs": c['endMs']}
                for c in page_chunk
            ]
        })
        # Remove trailing space from last token
        if pages[-1]['tokens']:
            pages[-1]['tokens'][-1]['text'] = pages[-1]['tokens'][-1]['text'].rstrip()
    
    return pages


def get_audio_duration(audio_path: str) -> int:
    """Get audio duration in milliseconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return int(float(result.stdout.strip()) * 1000)
    except:
        return 0


def check_mfa_available() -> bool:
    """Check if MFA is available."""
    try:
        result = subprocess.run(["mfa", "version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except:
        pass
    
    try:
        result = subprocess.run(["conda", "run", "-n", "mfa", "mfa", "version"], 
                              capture_output=True, timeout=10)
        return result.returncode == 0
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Forced alignment for Remotion TikTok captions using MFA"
    )
    parser.add_argument("--audio", "-a", required=True, help="Input audio file (WAV)")
    parser.add_argument("--script", "-s", required=True, help="Script text file")
    parser.add_argument("--output", "-o", required=True, help="Output captions JSON")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default: 30)")
    parser.add_argument("--words-per-page", type=int, default=6, help="Words per TikTok page")
    parser.add_argument("--method", choices=["mfa", "estimate"], default="auto",
                        help="Alignment method (auto tries MFA first, falls back to estimate)")
    
    args = parser.parse_args()
    
    # Load script
    with open(args.script) as f:
        script_text = f.read()
    
    # Select method
    method = args.method
    if method == "auto":
        if check_mfa_available():
            method = "mfa"
            print("✓ MFA is available, using forced alignment")
        else:
            method = "estimate"
            print("⚠ MFA not found, using proportional estimation")
    
    print(f"🎯 Alignment method: {method}")
    
    # Align
    try:
        if method == "mfa":
            captions = run_mfa_align(args.audio, script_text, args.fps)
            print(f"✓ MFA aligned {len(captions)} words")
        else:
            duration_ms = get_audio_duration(args.audio)
            if duration_ms == 0:
                print("⚠ Could not detect audio duration, using default 60s")
                duration_ms = 60000
            captions = estimate_timestamps(script_text, duration_ms, args.fps)
            print(f"✓ Estimated {len(captions)} word timestamps")
    except Exception as e:
        print(f"⚠ Alignment failed ({e}), falling back to estimation")
        duration_ms = get_audio_duration(args.audio)
        if duration_ms == 0:
            duration_ms = 60000
        captions = estimate_timestamps(script_text, duration_ms, args.fps)
    
    # Generate pages
    pages = captions_to_pages(captions, args.words_per_page)
    
    # Save
    result = {
        "method": method,
        "fps": args.fps,
        "captions": captions,
        "pages": pages,
    }
    
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✅ Generated {len(pages)} TikTok caption pages")
    print(f"💾 Saved to {args.output}")


if __name__ == "__main__":
    main()
