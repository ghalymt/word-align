"""Labelled speech for the benchmark.

A corpus is a folder of items, each an audio file plus its truth beside it,
matched by name:

* ``name.words.json`` -- ``[{"word", "start", "end"}, ...]``: text and word
  timing (the best kind; e.g. converted from MFA TextGrids or AMI's word
  annotations);
* ``name.TextGrid``   -- an MFA/Praat TextGrid with a ``words`` tier;
* ``name.srt``        -- a corrected subtitle file: text, and cue-level
  timing only (this is what your own reviewed jobs give);
* ``name.txt``        -- text only.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

AUDIO_EXT = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".mp4", ".mkv",
             ".mov", ".webm", ".aac")


@dataclass
class Item:
    id: str
    audio: str
    words: List[Dict] = field(default_factory=list)   # word, start?, end?
    cues: List[Dict] = field(default_factory=list)    # start, end, text (from SRT)
    timed: bool = False                               # word timings known

    @property
    def text(self) -> str:
        return " ".join(w["word"] for w in self.words)


def _read_srt(path: Path) -> List[Dict]:
    import srt
    cues = []
    for sub in srt.parse(path.read_text(encoding="utf-8-sig")):
        text = re.sub(r"\[.*?\]", "", sub.content).strip()
        if text:
            cues.append({"start": sub.start.total_seconds(),
                         "end": sub.end.total_seconds(), "text": text})
    return cues


def _read_textgrid(path: Path) -> List[Dict]:
    """Minimal TextGrid reader (long format): the tier named 'words'."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    words, in_words, cur = [], False, {}
    for line in lines:
        s = line.strip()
        if s.startswith("name ="):
            in_words = '"words"' in s.lower()
            continue
        if not in_words:
            continue
        if s.startswith("xmin ="):
            cur = {"start": float(s.split("=")[1])}
        elif s.startswith("xmax =") and "start" in cur:
            cur["end"] = float(s.split("=")[1])
        elif s.startswith("text =") and "end" in cur:
            text = s.split("=", 1)[1].strip().strip('"').strip()
            if text and text not in ("<eps>", "sil", "sp"):
                words.append({"word": text, **cur})
            cur = {}
    return words


def load_item(audio: Path) -> Optional[Item]:
    stem = audio.with_suffix("")
    for ext, kind in ((".words.json", "json"), (".TextGrid", "tg"),
                      (".srt", "srt"), (".txt", "txt")):
        p = Path(str(stem) + ext)
        if not p.exists():
            continue
        if kind == "json":
            words = json.loads(p.read_text(encoding="utf-8"))
            return Item(stem.name, str(audio), words=words,
                        timed=all("start" in w for w in words))
        if kind == "tg":
            return Item(stem.name, str(audio), words=_read_textgrid(p), timed=True)
        if kind == "srt":
            cues = _read_srt(p)
            words = [{"word": w} for c in cues for w in c["text"].split()]
            return Item(stem.name, str(audio), words=words, cues=cues)
        text = re.sub(r"\[.*?\]", "", p.read_text(encoding="utf-8"))
        return Item(stem.name, str(audio), words=[{"word": w} for w in text.split()])
    return None


def load_corpus(folder: str) -> List[Item]:
    items = []
    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            if name.lower().endswith(AUDIO_EXT):
                item = load_item(Path(root) / name)
                if item and item.words:
                    items.append(item)
    return items
