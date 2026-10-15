"""Speaker diarization: who spoke when, mapped onto aligned words.

Uses pyannote.audio. The default model, ``pyannote/speaker-diarization-
community-1`` (pyannote.audio 4), is gated on Hugging Face: accept its terms
once and log in or set ``HF_TOKEN`` (or keep the model in the local HF
cache). ``pyannote/speaker-diarization-3.1`` is the fallback.
``WORDALIGN_DIARIZATION_MODEL`` selects another pipeline.

When pyannote.audio is not importable here (the frozen app, or a main env
without it), diarization runs out-of-process in the WhisperX venv, which
ships pyannote.audio 4 (see ``engines/_diarize_worker.py``).

Besides who-spoke-when, the turns give the **overlap regions** (two or more
speakers at once): about 30% of NOTSOFAR-1 meeting audio but up to 90% of
its ASR errors, so they drive review triage and overlapped-speech ASR.

There is deliberately no fallback here. Treating every pause as a new
speaker (what the plugin adapter's offline heuristic does) puts wrong
names on subtitles, which is worse than no names; without pyannote the
stage reports why and is skipped.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"
FALLBACK_MODEL = "pyannote/speaker-diarization-3.1"
# A word with no overlapping speaker turn takes the nearest turn within this
# many seconds (diarization turns often stop just short of a word's end).
NEAREST_TURN_S = 1.0

Turn = Tuple[float, float, str]


class DiarizationUnavailable(RuntimeError):
    """Diarization cannot run here (dependency, model or audio problem)."""


def pyannote_available() -> bool:
    try:
        import pyannote.audio  # noqa: F401
        return True
    except Exception:
        return False


def _to_wav16k(audio_path: str, workdir: str) -> str:
    """Decode any media file to 16 kHz mono WAV, which pyannote reads."""
    ffmpeg = os.environ.get("WORDALIGN_FFMPEG") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise DiarizationUnavailable("ffmpeg not found (needed to decode audio)")
    out = os.path.join(workdir, "diarize.wav")
    subprocess.run([ffmpeg, "-i", audio_path, "-ar", "16000", "-ac", "1",
                    "-f", "wav", "-loglevel", "error", "-y", out], check=True)
    return out


def _load_pipeline(model: str, cache_dir: Optional[str]):
    from pyannote.audio import Pipeline
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    try:                                    # pyannote.audio >= 4
        pipeline = Pipeline.from_pretrained(model, token=token, **kwargs)
    except TypeError:                       # pyannote.audio 3.x
        pipeline = Pipeline.from_pretrained(model, use_auth_token=token, **kwargs)
    if pipeline is None:
        raise DiarizationUnavailable(
            f"could not load {model}: accept its terms on huggingface.co and "
            "set HF_TOKEN, or put it in the local Hugging Face cache")
    try:
        import torch
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception:
        pass
    return pipeline


def diarize_turns(audio_path: str, num_speakers: Optional[int] = None,
                  model: Optional[str] = None,
                  cache_dir: Optional[str] = None,
                  cancel_event=None) -> List[Turn]:
    """Run pyannote and return ``(start, end, raw_label)`` speaker turns.

    Turns may overlap (two people talking at once); see overlap_regions().
    """
    model = model or os.environ.get("WORDALIGN_DIARIZATION_MODEL", DEFAULT_MODEL)
    workdir = tempfile.mkdtemp(prefix="wordalign_diarize_")
    try:
        wav = _to_wav16k(audio_path, workdir)
        # Prefer the backend venv (pyannote.audio 4 + CUDA torch); an
        # importable pyannote here is only the fallback, since the main
        # environment usually has a CPU torch or an older pyannote.
        from .backends import backend_python
        if backend_python("diarize") or not pyannote_available():
            return _diarize_out_of_process(wav, num_speakers, model, cache_dir,
                                           cancel_event)
        pipeline = None
        for candidate in dict.fromkeys((model, FALLBACK_MODEL)):
            try:
                pipeline = _load_pipeline(candidate, cache_dir)
                break
            except DiarizationUnavailable:
                pipeline = None
        if pipeline is None:
            raise DiarizationUnavailable(
                f"could not load {model}: accept its terms on huggingface.co "
                "and set HF_TOKEN, or put it in the local Hugging Face cache")
        options = {"num_speakers": num_speakers} if num_speakers else {}
        # A waveform dict, not a path: pyannote.audio 4 reads files through
        # torchcodec, which is often missing.
        output = pipeline(_waveform(wav), **options)
        # pyannote 4 wraps the annotation; 3.x returns it directly.
        annotation = getattr(output, "speaker_diarization", output)
        return [(float(turn.start), float(turn.end), str(label))
                for turn, _, label in annotation.itertracks(yield_label=True)]
    except DiarizationUnavailable:
        raise
    except Exception as exc:
        raise DiarizationUnavailable(f"pyannote failed: {exc}") from exc
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _waveform(wav_path: str) -> Dict:
    import wave
    import numpy as np
    import torch
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    x = torch.from_numpy(data.astype(np.float32) / 32767.0)[None, :]
    return {"waveform": x, "sample_rate": sr}


def _diarize_out_of_process(wav: str, num_speakers: Optional[int], model: str,
                            cache_dir: Optional[str], cancel_event=None) -> List[Turn]:
    from pathlib import Path
    from .backends import backend_python
    from .engines._worker import WorkerCancelled, WorkerError, run_worker
    python = backend_python("diarize")
    if not python:
        raise DiarizationUnavailable(
            "pyannote.audio is not installed here and no WhisperX venv was "
            "found (it ships pyannote.audio 4)")
    args = ["--audio", wav, "--model", model]
    if cache_dir:
        args += ["--cache-dir", cache_dir]
    if num_speakers:
        args += ["--num-speakers", str(num_speakers)]
    worker = Path(__file__).resolve().parent / "engines" / "_diarize_worker.py"
    try:
        payload = run_worker(python, worker, args, "__WA_DIARIZE_JSON__",
                             cancel_event=cancel_event, name="diarization worker")
    except WorkerCancelled:
        raise
    except WorkerError as exc:
        raise DiarizationUnavailable(str(exc).splitlines()[-1]) from exc
    return [(float(s), float(e), str(lbl)) for s, e, lbl in payload.get("turns", [])]


def overlap_regions(turns: Sequence[Turn], min_duration: float = 0.1) -> List[Tuple[float, float]]:
    """Merged ``(start, end)`` spans where two or more speakers talk at once."""
    events = []
    for s, e, _ in turns:
        if e > s:
            events.append((s, 1))
            events.append((e, -1))
    events.sort(key=lambda x: (x[0], x[1]))
    regions: List[Tuple[float, float]] = []
    active, start = 0, None
    for t, delta in events:
        before = active
        active += delta
        if before < 2 <= active:
            start = t
        elif before >= 2 > active and start is not None:
            if t - start >= min_duration:
                if regions and start - regions[-1][1] < 0.05:
                    regions[-1] = (regions[-1][0], t)
                else:
                    regions.append((start, t))
            start = None
    return [(round(s, 3), round(e, 3)) for s, e in regions]


def speaker_names(turns: Sequence[Turn]) -> Dict[str, str]:
    """Raw pyannote labels -> "Speaker 1", "Speaker 2"... by first appearance."""
    names: Dict[str, str] = {}
    for _, _, label in sorted(turns, key=lambda t: t[0]):
        names.setdefault(label, f"Speaker {len(names) + 1}")
    return names


def assign_speakers(words: List[Dict], turns: Sequence[Turn]) -> int:
    """Set ``word["speaker"]`` from the turns; returns how many got one.

    Each word takes the speaker whose turn overlaps it most; a word inside
    no turn takes the nearest turn within NEAREST_TURN_S, else none.
    """
    names = speaker_names(turns)
    labelled = 0
    for word in words:
        if word.get("overlap_added") and word.get("speaker"):
            # Transcribed for the *other* speaker in an overlap (SE-DiCoW);
            # the turn overlapping it most is usually the dominant speaker.
            labelled += 1
            continue
        start, end = word.get("start"), word.get("end")
        if start is None or end is None:
            continue
        best, best_overlap, nearest, nearest_gap = None, 0.0, None, None
        for t_start, t_end, label in turns:
            overlap = min(end, t_end) - max(start, t_start)
            if overlap > best_overlap:
                best, best_overlap = label, overlap
            gap = max(t_start - end, start - t_end, 0.0)
            if nearest_gap is None or gap < nearest_gap:
                nearest, nearest_gap = label, gap
        if best is None and nearest_gap is not None and nearest_gap <= NEAREST_TURN_S:
            best = nearest
        if best is not None:
            word["speaker"] = names[best]
            labelled += 1
    return labelled
