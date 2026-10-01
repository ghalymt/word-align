"""WebVTT and Advanced SubStation Alpha (ASS) writers.

Both take the same cue list the sentence-level SRT is built from: objects
with ``start`` / ``end`` (``datetime.timedelta``) and ``content`` (text with
``\\n`` line breaks) -- i.e. ``srt.Subtitle`` instances. A cue may also carry
a ``speaker`` attribute (set by diarization); VTT writes it as a voice span
and ASS as the event's Name field.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Iterable, Optional


def _parts(td: timedelta):
    ms = max(0, int(round(td.total_seconds() * 1000)))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    seconds, ms = divmod(ms, 1000)
    return hours, minutes, seconds, ms


def vtt_timestamp(td: timedelta) -> str:
    """``HH:MM:SS.mmm`` (WebVTT always uses a dot and three digits)."""
    h, m, s, ms = _parts(td)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def ass_timestamp(td: timedelta) -> str:
    """``H:MM:SS.cc`` -- ASS uses centiseconds, truncated so that a cue's
    end never rounds past the next cue's start."""
    h, m, s, ms = _parts(td)
    return f"{h:d}:{m:02d}:{s:02d}.{ms // 10:02d}"


def _vtt_escape(text: str) -> str:
    # Cue text is HTML-like: a literal & or < would start an entity or tag.
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def compose_vtt(cues: Iterable) -> str:
    """Render cues as a WebVTT document."""
    blocks = ["WEBVTT", ""]
    for n, cue in enumerate(cues, 1):
        lines = [_vtt_escape(line) for line in cue.content.strip().split("\n")]
        speaker: Optional[str] = getattr(cue, "speaker", None)
        if speaker and lines:
            lines[0] = f"<v {_vtt_escape(speaker)}>{lines[0]}"
        blocks.append(str(n))
        blocks.append(f"{vtt_timestamp(cue.start)} --> {vtt_timestamp(cue.end)}")
        blocks.extend(lines)
        blocks.append("")
    return "\n".join(blocks)


def _ass_escape(text: str) -> str:
    # "{...}" is an override block in ASS; libass/mpv/VLC/ffmpeg render
    # "\{" and "\}" as literal braces. Line breaks become the hard "\N".
    text = text.replace("{", "\\{").replace("}", "\\}")
    return "\\N".join(line.strip() for line in text.strip().split("\n"))


def compose_ass(cues: Iterable, title: str = "WordAlign subtitles",
                play_res=(1920, 1080), font: str = "Arial",
                font_size: Optional[int] = None) -> str:
    """Render cues as an ASS (v4.00+) script with one bottom-centre style.

    WrapStyle 2 disables the renderer's own line wrapping: the cues are
    already laid out to the job's CPL and line count.
    """
    width, height = play_res
    size = font_size or max(16, round(height * 0.055))
    margin_v = round(height * 0.045)
    out = [
        "[Script Info]",
        f"Title: {title}",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font},{size},&H00FFFFFF,&H000000FF,&H00000000,"
        f"&H80000000,0,0,0,0,100,100,0,0,1,3,1,2,60,60,{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text",
    ]
    for cue in cues:
        speaker = (getattr(cue, "speaker", None) or "").replace(",", " ")
        out.append(f"Dialogue: 0,{ass_timestamp(cue.start)},"
                   f"{ass_timestamp(cue.end)},Default,{speaker},0,0,0,,"
                   f"{_ass_escape(cue.content)}")
    return "\n".join(out) + "\n"
