#!/usr/bin/env python3
"""
Ground-Truth Fuzzy Alignment Utility
=====================================
Maps generated TTS audio captions to ground-truth script text.
Fixes Whisper-style splinters and ensures technical terms are correctly spelled.

This is used when proportional estimation produces word splits that need correction,
or when you want to align high-quality known text with the audio timing.

Example:
    python align_ground_truth.py --audio narration.wav --text script.txt --output captions.json
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional


def clean_text(text: str) -> str:
    """Remove punctuation for fuzzy matching."""
    return re.sub(r'[^a-zA-Z0-9]', '', text).lower()


def align_to_ground_truth(
    estimated_captions: List[Dict],
    ground_truth_text: str,
    words_per_page: int = 6,
) -> Dict:
    """
    Align estimated captions to ground-truth text.
    
    Args:
        estimated_captions: List of {text, startMs, endMs} from TTS estimation
        ground_truth_text: The original script text
        words_per_page: Words per TikTok caption page
    
    Returns:
        Dict with aligned 'captions' and 'pages'
    """
    # Tokenize ground truth
    gt_tokens = ground_truth_text.strip().replace('\n', ' ').split()
    gt_tokens = [t for t in gt_tokens if t.strip()]
    
    if not estimated_captions or not gt_tokens:
        return {"captions": [], "pages": []}
    
    mapped_captions = []
    est_idx = 0
    
    for gt_word in gt_tokens:
        if est_idx >= len(estimated_captions):
            break
        
        gt_clean = clean_text(gt_word)
        
        # Look ahead for matching
        found = False
        for lookahead in range(1, 6):
            if est_idx + lookahead <= len(estimated_captions):
                potential = "".join([
                    clean_text(c['text']) 
                    for c in estimated_captions[est_idx:est_idx + lookahead]
                ])
                
                if gt_clean in potential or potential in gt_clean or gt_clean == potential:
                    start_ms = estimated_captions[est_idx]['startMs']
                    end_ms = estimated_captions[est_idx + lookahead - 1]['endMs']
                    
                    mapped_captions.append({
                        "text": gt_word,
                        "startMs": start_ms,
                        "endMs": end_ms,
                    })
                    est_idx += lookahead
                    found = True
                    break
        
        if not found:
            # Use current estimate timing
            mapped_captions.append({
                "text": gt_word,
                "startMs": estimated_captions[est_idx]['startMs'],
                "endMs": estimated_captions[est_idx]['endMs'],
            })
            est_idx += 1
    
    # Generate TikTok-style pages
    pages = []
    for i in range(0, len(mapped_captions), words_per_page):
        page_chunk = mapped_captions[i:i + words_per_page]
        if not page_chunk:
            continue
        
        text = " ".join(c['text'] for c in page_chunk)
        pages.append({
            "text": text,
            "startMs": page_chunk[0]['startMs'],
            "durationMs": page_chunk[-1]['endMs'] - page_chunk[0]['startMs'],
            "tokens": [
                {"text": c['text'], "fromMs": c['startMs'], "toMs": c['endMs']}
                for c in page_chunk
            ]
        })
    
    return {
        "captions": mapped_captions,
        "pages": pages,
    }


def main():
    parser = argparse.ArgumentParser(description="Align TTS captions to ground-truth text")
    parser.add_argument("--input", "-i", required=True, help="Input captions JSON from Pocket TTS")
    parser.add_argument("--text", "-t", required=True, help="Ground-truth text file")
    parser.add_argument("--output", "-o", required=True, help="Output aligned captions JSON")
    parser.add_argument("--words-per-page", type=int, default=6, help="Words per TikTok page")
    
    args = parser.parse_args()
    
    # Load input captions
    with open(args.input) as f:
        data = json.load(f)
    
    # Get captions (could be 'captions' or extracted from 'pages')
    if 'captions' in data:
        estimated = data['captions']
    else:
        # Extract from pages
        estimated = []
        for page in data.get('pages', []):
            for token in page.get('tokens', []):
                estimated.append({
                    "text": token['text'].strip(),
                    "startMs": token['fromMs'],
                    "endMs": token['toMs'],
                })
    
    # Load ground-truth text
    with open(args.text) as f:
        ground_truth = f.read()
    
    # Align
    result = align_to_ground_truth(estimated, ground_truth, args.words_per_page)
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✅ Aligned {len(result['captions'])} words into {len(result['pages'])} pages")
    print(f"💾 Saved to {args.output}")


if __name__ == "__main__":
    main()
