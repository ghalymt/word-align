"""Shared helpers: text normalization, timecode math, ffmpeg wrappers."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from .config import NON_TERMINAL_ABBREVIATIONS, TERMINAL_PUNCT_PATTERN

# Trailing closing quotes/brackets to peel off before inspecting the last word.
_TRAILING_CLOSERS_RE = re.compile(r"[\"'”’»）】]+$")

# NOTE: the tag-stripping regex is deliberately written once, here, and
# imported everywhere. An earlier version of this pipeline lost the
# backslashes in three call sites (``[[.*?]]`` instead of ``\[.*?\]``),
# which silently let bracketed tags leak into the alignment word list.
TAG_RE = re.compile(r"\[.*?\]")
TIMECODE_TAG_RE = re.compile(r"\d{2}:\d{2}:\d{2}")


def strip_tags(text: str) -> str:
    """Remove all ``[bracketed]`` annotations from *text*."""
    return TAG_RE.sub("", text)


def extract_tags_from_transcript(text: str) -> set:
    """Return the set of tag labels a human transcriber embedded in *text*."""
    tags = re.findall(r"\[(.*?)\]", text)
    return {
        tag.strip()
        for tag in tags
        if not TIMECODE_TAG_RE.match(tag.strip()) and "END OF AUDIO" not in tag
    }


def normalize_word(word) -> str:
    return re.sub(r"[^\w\s]", "", str(word)).lower()


def count_visible_characters(text: str) -> int:
    return len(text.replace("\n", "").replace("\r", ""))


def ends_with_nonterminal_abbreviation(text: str) -> bool:
    """True if *text*'s final token is an abbreviation like ``St.`` or ``Mr.``.

    Used to stop the segmenter from treating an abbreviation's period as a
    sentence boundary. Only the plain-period case matters -- ``?``/``!``/``…``
    are unambiguous ends and are never abbreviations.
    """
    t = _TRAILING_CLOSERS_RE.sub("", text.strip())
    if not t.endswith("."):
        return False
    tokens = t.split()
    if not tokens:
        return False
    key = tokens[-1].lower().rstrip(".")
    return key in NON_TERMINAL_ABBREVIATIONS


def has_terminal_punctuation(text: str) -> bool:
    stripped = text.strip()
    if stripped.endswith("--"):
        return True
    if not TERMINAL_PUNCT_PATTERN.search(stripped):
        return False
    # A trailing abbreviation ("...at St.") is not a sentence end.
    if ends_with_nonterminal_abbreviation(stripped):
        return False
    return True


def time_to_ms(time_str: str) -> int:
    try:
        hours, minutes, seconds_ms = time_str.split(":")
        seconds, milliseconds = seconds_ms.split(",")
        return (int(hours) * 3_600_000 + int(minutes) * 60_000
                + int(seconds) * 1_000 + int(milliseconds))
    except (ValueError, AttributeError):
        return 0


def ms_to_time(ms: int) -> str:
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    seconds = ms // 1_000
    milliseconds = ms % 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def get_audio_duration(input_path: str) -> float:
    ffprobe = os.environ.get("WORDALIGN_FFPROBE") or shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    cmd = [ffprobe, "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", input_path]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        return float(result.stdout)
    except (ValueError, OSError):
        return 0.0


def create_chunk_wav(input_path: str, start_time: float, duration: float,
                     output_path: str) -> None:
    # Use an absolute path to ffmpeg when possible. The bare command
    # "ffmpeg" would otherwise hit torchaudio's bundled shim (which
    # is broken in a one-file PyInstaller build) before reaching the
    # real binary on the host's PATH.
    import shutil
    ffmpeg = os.environ.get("WORDALIGN_FFMPEG") or shutil.which("ffmpeg")
    if ffmpeg and os.path.isfile(ffmpeg):
        cmd = [ffmpeg, "-ss", str(start_time), "-t", str(duration),
               "-i", input_path,
               "-ar", "16000", "-ac", "1", "-f", "wav",
               "-loglevel", "error", "-y", output_path]
    else:
        cmd = ["ffmpeg", "-ss", str(start_time), "-t", str(duration),
               "-i", input_path,
               "-ar", "16000", "-ac", "1", "-f", "wav",
               "-loglevel", "error", "-y", output_path]
    subprocess.run(cmd, check=True)


def detect_language(text: str, default: str = "en") -> str:
    """Best-effort language detection from a transcript."""
    try:
        from langdetect import detect as detect_lang_text
    except ImportError:
        print("[warn] langdetect not installed; defaulting to "
              f"'{default}'.")
        return default
    clean_text = strip_tags(text).strip()
    if not clean_text:
        return default
    try:
        # langdetect says "zh-cn" / "zh-tw"; every engine map is keyed by
        # the primary subtag.
        lang = detect_lang_text(clean_text).split("-")[0].lower()
        print(f"[ok] Auto-detected language: {lang}")
        return lang
    except Exception as exc:  # langdetect raises its own exception type
        print(f"[warn] Language detection failed ({exc}); defaulting to "
              f"'{default}'.")
        return default
