"""Exercise every bad-audio backend on a real clip and report what works.

    python scripts/smoke_robust.py path/to/clip.wav [--seconds 60] [--only vad,cohere]

Each step runs the same worker the pipeline uses and prints OK / FAIL /
SKIP with its time and a short result, so a broken venv or model shows up
before a real job. Set WORDALIGN_MODELS_DIR / WORDALIGN_VENVS_DIR when the
models or venvs live outside the project folder.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STEPS = ("vad", "sed", "diarize", "qwen", "qwen_align", "cohere", "parakeet",
         "whisperx", "wav2vec2_align", "separate", "denoise", "sedicow")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--language", default="en")
    a = ap.parse_args()
    only = [s.strip() for s in a.only.split(",") if s.strip()] or list(STEPS)

    from wordalign.audio.io import load_audio, write_wav
    from wordalign.models.paths import _project_models_root
    models = str(_project_models_root())
    hub = os.path.join(models, "huggingface", "hub")
    work = tempfile.mkdtemp(prefix="wa_smoke_")
    audio = load_audio(a.audio, duration=a.seconds)
    clip = os.path.join(work, "clip.wav")
    write_wav(clip, audio)
    dur = len(audio) / 16000
    print(f"clip: {dur:.1f}s  models: {models}  work: {work}")
    state = {}
    results = []

    def step(name, fn):
        if name not in only:
            return
        t0 = time.time()
        try:
            note = fn()
            status = "OK" if note is not False else "SKIP"
        except Exception as exc:  # report and continue
            status, note = "FAIL", f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        secs = time.time() - t0
        results.append((name, status, secs, note))
        print(f"[{status:4}] {name:<15} {secs:6.1f}s  {str(note)[:300]}", flush=True)

    def vad():
        from wordalign.audio.analysis import analyze_audio
        q = analyze_audio(clip, models_dir=models, use_sed=False, audio=audio, workdir=work)
        state["q"] = q
        state["chunks"] = [(c.start, c.end) for c in q.chunks]
        s = q.summary()
        return (f"speech {s['speech_seconds']}s, {s['chunks']} chunks, median SNR "
                f"{s['median_snr_db']} dB, notes={s['notes']}")
    step("vad", vad)
    chunks = state.get("chunks") or [(0.0, min(dur, 28.0))]

    def sed():
        from wordalign.audio.analysis import analyze_audio
        q = analyze_audio(clip, models_dir=models, use_sed=True, audio=audio, workdir=work)
        if not q.has_sed:
            raise RuntimeError("; ".join(q.notes) or "no sound events")
        top = sorted({e["label"] for e in q.sed_events})[:8]
        return f"{len(q.sed_events)} events, groups {sorted(q.sed_groups)}; labels {top}"
    step("sed", sed)

    def diarize():
        from wordalign.diarize import diarize_turns, overlap_regions
        turns = diarize_turns(clip, cache_dir=hub)
        state["turns"] = turns
        return f"{len(turns)} turns, {len({t[2] for t in turns})} speakers, overlaps {overlap_regions(turns)[:3]}"
    step("diarize", diarize)

    def qwen():
        from wordalign.engines.qwen_engine import run_qwen
        p = run_qwen(clip, a.language, models_dir=hub, chunks=chunks, return_payload=True)
        if not p:
            raise RuntimeError("no payload")
        state["qwen_words"] = p.get("words", [])
        text = " ".join(w["word"] for w in p["words"][:25])
        return f"{len(p['words'])} words, failed={p.get('failed')}: {text}"
    step("qwen", qwen)

    def qwen_align():
        from wordalign.engines.qwen_engine import qwen_align
        words = [w["word"] for w in state.get("qwen_words", [])][:60]
        if not words:
            return False
        res = qwen_align(clip, [{"id": 0, "start": 0.0, "end": min(dur, 60.0), "words": words}],
                         a.language, models_dir=hub)
        if not res:
            raise RuntimeError("no result")
        times = res[0].get("times", [])
        placed = sum(1 for t in times if t)
        return f"status={res[0].get('status')} placed {placed}/{len(words)}; first {times[:3]}"
    step("qwen_align", qwen_align)

    def cohere():
        from wordalign.engines.voters import run_cohere
        out = run_cohere(clip, chunks, a.language, hub, log=print)
        if out is None:
            raise RuntimeError("no output (see log above)")
        text = " ".join(w["word"] for i in sorted(out) for w in out[i])
        return f"{len(out)}/{len(chunks)} chunks: {text[:200]}"
    step("cohere", cohere)

    def parakeet():
        from wordalign.engines.voters import run_parakeet
        out = run_parakeet(clip, chunks, hub, log=print)
        if out is None:
            raise RuntimeError("no output (see log above)")
        words = [w for i in sorted(out) for w in out[i]]
        return f"{len(words)} timed words: {' '.join(w['word'] for w in words[:25])}"
    step("parakeet", parakeet)

    def whisperx():
        from wordalign.engines.whisperx_engine import run_whisperx
        words, events = run_whisperx(clip, a.language, "large-v3", None,
                                     os.path.join(models, "whisper"))
        if not words:
            raise RuntimeError("no words")
        return f"{len(words)} words: {' '.join(w['word'] for w in words[:25])}"
    step("whisperx", whisperx)

    def wav2vec2_align():
        from wordalign.engines.whisperx_engine import wav2vec2_align
        words = [w["word"] for w in state.get("qwen_words", [])][:40]
        if not words:
            return False
        res = wav2vec2_align(clip, [{"id": 0, "start": 0.0, "end": min(dur, 30.0), "words": words}],
                             a.language, models_dir=os.path.join(models, "whisper"))
        if not res:
            raise RuntimeError("no result")
        placed = sum(1 for t in res[0]["times"] if t)
        return f"status={res[0].get('status')} placed {placed}/{len(words)}"
    step("wav2vec2_align", wav2vec2_align)

    def cleanup(op):
        def run():
            from wordalign.audio.prep import prepare_asr_audio
            q = state.get("q")
            if q is None or not q.chunks:
                return False
            path, applied = prepare_asr_audio(audio, q, [(0, op)], work, models_dir=models,
                                              log=print)
            if not applied:
                raise RuntimeError("nothing applied (see log above)")
            return f"{applied} -> {path}"
        return run
    step("separate", cleanup("vocals"))
    step("denoise", cleanup("denoise"))

    def sedicow():
        from wordalign.engines.sedicow_engine import run_sedicow
        turns = state.get("turns") or []
        if len({t[2] for t in turns}) < 2:
            # Fake a second speaker so the worker path is exercised.
            mid = min(dur, 20.0) / 2
            turns = [(0.0, mid + 1.0, "A"), (mid - 1.0, min(dur, 20.0), "B")]
        res = run_sedicow(clip, [(0.0, min(dur, 20.0))], turns, a.language, hub, log=print)
        if res is None:
            raise RuntimeError("no result (see log above)")
        segs = [(sp["label"], [s["text"] for s in sp["segments"]][:2])
                for w in res for sp in w.get("speakers", [])]
        return json.dumps(segs)[:300]
    step("sedicow", sedicow)

    failed = [r for r in results if r[1] == "FAIL"]
    print(f"\n{len(results) - len(failed)}/{len(results)} steps OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
