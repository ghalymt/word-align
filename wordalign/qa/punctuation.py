"""Punctuation & capitalization restoration for subtitle cues.

Speech engines (Vosk especially) output lowercase text with no punctuation,
which makes subtitles hard to read. This module restores punctuation and
capitalization either with:

- LlamaCppRestorer   — a local LLM (llama.cpp, optional MTP draft model)
- RuleBasedRestorer  — fast deterministic fallback (always available)

Both operate per-cue so timestamps are preserved 1:1.

Environment variables:
    WORDALIGN_LLM_ENGINE     path to llama.cpp directory (or llama-cli.exe)
    WORDALIGN_LLM_MODEL      path to the main GGUF model
    WORDALIGN_LLM_MTP_MODEL  path to the MTP draft GGUF (optional, enables MTP)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, List, Optional

# ---------------------------------------------------------------------------
# Rule-based restorer — fast deterministic fallback
# ---------------------------------------------------------------------------
_I_RE = re.compile(r"\bi\b", re.IGNORECASE)

# Words that begin a sentence and get capitalized (first word of cue + after
# terminal punctuation). We capitalize conservatively: first character of the
# cue, and any word following . ! ? — the rest stays as transcribed.
_TERMINAL = re.compile(r"([.!?…。！？]+)\s+(\w)")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…。！？])\s+")


def _wrap_text(text: str, max_cpl: int) -> str:
    """Greedy-wrap cue text into lines of at most *max_cpl* chars.

    Breaks at sentence boundaries first, then word boundaries. Single-line
    cues shorter than max_cpl are returned unchanged.
    """
    text = text.strip()
    if not text or max_cpl <= 0:
        return text
    lines = text.split("\n")
    wrapped: List[str] = []
    for line in lines:
        if len(line) <= max_cpl:
            wrapped.append(line)
            continue
        # Break at sentence boundaries first
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(line) if s.strip()]
        current = ""
        for sent in sentences:
            if current and len(current) + 1 + len(sent) <= max_cpl:
                current += " " + sent
            elif not current:
                current = sent
            else:
                wrapped.append(current)
                current = sent
            # A single sentence still over budget → word-wrap it
            if len(current) > max_cpl:
                for piece in _word_wrap(current, max_cpl):
                    wrapped.append(piece)
                current = ""
        if current:
            wrapped.append(current)
    return "\n".join(wrapped)


def _word_wrap(text: str, max_cpl: int) -> List[str]:
    """Wrap text at word boundaries to max_cpl chars per line."""
    words = text.split()
    if not words:
        return []
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= max_cpl:
            current += " " + word
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


class RuleBasedRestorer:
    """Deterministic punctuation/capitalization heuristics."""

    def __init__(self, add_terminal_period: bool = True):
        self.add_terminal_period = add_terminal_period

    def restore(self, text: str) -> str:
        text = text.strip()
        if not text:
            return text

        # 1. Lowercase "i" -> "I" (covers I, I'm, I've, I'll, I'd ...)
        text = _I_RE.sub("I", text)

        # 2. Capitalize after terminal punctuation
        def _cap(match: "re.Match") -> str:
            return match.group(1) + " " + match.group(2).upper()
        text = _TERMINAL.sub(_cap, text)

        # 3. Capitalize the first character of the cue
        text = text[0].upper() + text[1:]

        # 4. Optional terminal period (only if cue looks like a sentence and
        #    has no terminal punctuation yet)
        if self.add_terminal_period and text and not re.search(
                r"[.!?…。！？»”'\"]$", text):
            text += "."

        return text

    def restore_cues(self, cues: List[str]) -> List[str]:
        return [self.restore(c) for c in cues]


# ---------------------------------------------------------------------------
# llama.cpp restorer — local LLM with optional MTP
# ---------------------------------------------------------------------------
_BANNER_RE = re.compile(r"^▄|^██|^Loading model", re.MULTILINE)
_STATS_RE = re.compile(r"^\s*\[ (Prompt|Generation): .* \]\s*$")
_EXIT_RE = re.compile(r"^Exiting\.\.\.\s*$")
_ECHO_RE = re.compile(r"^>\s+")
_NUMBERED_RE = re.compile(r"^\s*(\d+)[.)]\s*(.+?)\s*$")
# Model meta-commentary that must never be treated as a restored cue
_COMMENTARY_RE = re.compile(
    r"^\s*(wait|note|note:|hmm|i (think|can'?t|cannot|should)|there is no|no cue|that's? (odd|weird)|the (prompt|input))",
    re.IGNORECASE)
_THINK_START_RE = re.compile(r"\[Start (thinking|reasoning)\]", re.IGNORECASE)
_THINK_END_RE = re.compile(r"\[End (thinking|reasoning)\]|\[End of (thinking|reasoning)\]", re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove [Start thinking] ... [End thinking] blocks from output."""
    start = _THINK_START_RE.search(text)
    while start:
        end = _THINK_END_RE.search(text, start.end())
        if not end:
            text = text[:start.start()]
            break
        text = text[:start.start()] + text[end.end():]
        start = _THINK_START_RE.search(text)
    return text


class LlamaCppRestorer:
    """Restore punctuation using a local llama.cpp model (optional MTP)."""

    def __init__(self,
                 engine: Optional[str] = None,
                 model: Optional[str] = None,
                 mtp_model: Optional[str] = None,
                 use_mtp: bool = True,
                 max_tokens: int = 2048,
                 temperature: float = 0.1,
                 max_cpl: int = 42,
                 verbose: bool = False):
        self.engine = engine or os.environ.get("WORDALIGN_LLM_ENGINE")
        self.model = model or os.environ.get("WORDALIGN_LLM_MODEL")
        self.mtp_model = mtp_model or os.environ.get("WORDALIGN_LLM_MTP_MODEL")
        self.use_mtp = use_mtp
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_cpl = max_cpl
        self.verbose = verbose
        self._cli_path: Optional[str] = None

    # -- resolution --------------------------------------------------------
    def cli_path(self) -> Optional[str]:
        """Locate llama-cli.exe (engine dir, env, or PATH)."""
        if self._cli_path:
            return self._cli_path
        candidates: List[str] = []
        if self.engine:
            eng = Path(self.engine)
            if eng.is_dir():
                candidates.append(str(eng / "llama-cli.exe"))
                candidates.append(str(eng / "llama-cli"))
            elif eng.is_file():
                candidates.append(str(eng))
        candidates.append(shutil.which("llama-cli") or "")
        for c in candidates:
            if c and Path(c).exists():
                self._cli_path = c
                return c
        return None

    def is_ready(self) -> tuple[bool, str]:
        """Check all components are available."""
        if not self.cli_path():
            return False, f"llama-cli not found (engine={self.engine!r})"
        if not self.model or not Path(self.model).exists():
            return False, f"model not found: {self.model!r}"
        if self.use_mtp and self.mtp_model and not Path(self.mtp_model).exists():
            return False, f"MTP model not found: {self.mtp_model!r}"
        return True, "ready"

    # -- inference ---------------------------------------------------------
    def _build_prompt(self, cues: List[str]) -> str:
        lines = []
        for i, cue in enumerate(cues, 1):
            lines.append(f"{i}. {cue}")
        return (
            "You restore punctuation and capitalization in subtitle text.\n"
            "Rules: do NOT add, remove, or reword any words. Only add "
            "punctuation and capitalize correctly. Keep the exact word order.\n"
            "For each numbered line below, output ONLY the corrected line "
            "with the same number.\n\n"
            + "\n".join(lines)
        )

    def _clean_output(self, raw: str) -> str:
        raw = _strip_thinking(raw)
        out_lines = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if _BANNER_RE.match(line):
                continue
            if _ECHO_RE.match(line) and not _NUMBERED_RE.match(line):
                continue
            if _STATS_RE.match(line) or _EXIT_RE.match(line):
                continue
            out_lines.append(line)
        return "\n".join(out_lines)

    def _parse_cues(self, cleaned: str, count: int) -> Optional[List[str]]:
        """Parse numbered output back into per-cue text."""
        results: dict[int, str] = {}
        for line in cleaned.splitlines():
            m = _NUMBERED_RE.match(line)
            if not m:
                continue
            idx = int(m.group(1))
            text = m.group(2).strip()
            if _COMMENTARY_RE.match(text):
                continue
            if "[" in text or "]" in text:
                continue  # brackets = meta-commentary, never a cue
            if 1 <= idx <= count:
                results[idx] = text
        if len(results) != count:
            return None
        return [results[i + 1] for i in range(count)]

    def _wrap_cues(self, cues: List[str]) -> List[str]:
        """Split cue text into lines that respect max CPL."""
        out = []
        for cue in cues:
            out.append(_wrap_text(cue, self.max_cpl))
        return out

    def restore_cues(self, cues: List[str]) -> List[str]:
        """Restore all cues in one LLM call; fall back to rules on failure."""
        fallback = RuleBasedRestorer()
        ready, msg = self.is_ready()
        if not ready:
            if self.verbose:
                print(f"[punctuation] LLM unavailable ({msg}); using rules.")
            return self._wrap_cues(fallback.restore_cues(cues))
        if not cues:
            return []

        prompt = self._build_prompt(cues)
        cmd = [
            self.cli_path(), "-m", self.model,
            "-st", "--no-display-prompt", "--no-warmup",
            "--reasoning", "off",
            "-p", prompt,
            "-n", str(self.max_tokens),
            "--temp", str(self.temperature),
            "--ctx-size", "8192",
        ]
        if self.use_mtp and self.mtp_model:
            cmd += ["--spec-type", "draft-mtp", "-md", self.mtp_model]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as exc:
            if self.verbose:
                print(f"[punctuation] llama-cli failed ({exc}); using rules.")
            return self._wrap_cues(fallback.restore_cues(cues))

        if proc.returncode != 0:
            if self.verbose:
                tail = "\n".join(proc.stderr.strip().splitlines()[-3:])
                print(f"[punctuation] llama-cli exit {proc.returncode}: {tail}")
            return self._wrap_cues(fallback.restore_cues(cues))

        cleaned = self._clean_output(proc.stdout)
        parsed = self._parse_cues(cleaned, len(cues))
        if parsed is not None:
            if self.verbose:
                print(f"[punctuation] LLM restored {len(parsed)} cues "
                      f"(mtp={'on' if self.use_mtp and self.mtp_model else 'off'})")
            return self._wrap_cues(parsed)

        if self.verbose:
            print("[punctuation] LLM output unparseable; using rules.")
        return self._wrap_cues(fallback.restore_cues(cues))


# ---------------------------------------------------------------------------
# Factory + segment-level helpers
# ---------------------------------------------------------------------------
def create_restorer(cfg) -> Callable[[List[str]], List[str]]:
    """Create a cue restorer from a PipelineConfig.

    Returns a callable taking a list of cue texts and returning restored
    texts. LLM when configured, rules otherwise.
    """
    mp = cfg.model_paths
    use_llm = bool(getattr(cfg, "llm_model", None) or
                   os.environ.get("WORDALIGN_LLM_MODEL"))
    if use_llm:
        restorer = LlamaCppRestorer(
            engine=getattr(cfg, "llm_engine", None) or mp.custom_paths.get("llm"),
            model=getattr(cfg, "llm_model", None),
            mtp_model=getattr(cfg, "llm_mtp_model", None),
            use_mtp=getattr(cfg, "llm_mtp", True),
            max_cpl=getattr(cfg, "max_cpl", 42),
            verbose=True,
        )
        return restorer.restore_cues
    return RuleBasedRestorer().restore_cues


def restore_segments(segments: List[dict],
                     restorer: Callable[[List[str]], List[str]]) -> List[dict]:
    """Restore punctuation/capitalization on segment texts in place.

    Returns the same list of segment dicts with 'text' updated.
    """
    if not segments:
        return segments
    texts = [s.get("text", "") for s in segments]
    restored = restorer(texts)
    for seg, new_text in zip(segments, restored):
        if new_text:
            seg["text"] = new_text
    return segments
