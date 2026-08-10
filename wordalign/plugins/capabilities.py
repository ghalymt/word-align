"""Capability flags for engine plugins.

A plugin may expose more than one capability. The pipeline queries the
registry by capability, not by engine name.
"""
from __future__ import annotations

from enum import Flag, auto


class Capability(Flag):
    TRANSCRIBE    = auto()   # produces word/text from audio
    WORD_TIMING   = auto()   # produces word-level timestamps
    FORCED_ALIGN  = auto()   # aligns given text to audio
    VOTE          = auto()   # participates in ensemble voting
    DIARIZE       = auto()   # assigns speaker IDs
    AUDIO_EVENTS  = auto()   # detects non-speech events
    VAD           = auto()   # voice activity detection
    TEXT_QA       = auto()   # reviews transcript text
    AUDIO_QA      = auto()   # reviews audio directly
    EXPORT        = auto()   # produces output files

    def labels(self) -> list[str]:
        """Human-readable capability labels."""
        names = []
        for cap in Capability:
            if self & cap:
                names.append(cap.name.lower())
        return names
