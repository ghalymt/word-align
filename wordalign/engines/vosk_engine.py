"""Parallel chunked Vosk decoding (CPU, lightweight, per-word confidence)."""
from __future__ import annotations

import concurrent.futures
import json
import math
import os
import shutil
import tempfile
import time
import wave
from typing import Dict, List, Tuple

from ..config import MAX_WORKERS, VOSK_CHUNK_SECONDS
from ..utils import create_chunk_wav, get_audio_duration

try:
    from vosk import Model, KaldiRecognizer, SetLogLevel
    VOSK_AVAILABLE = True
    SetLogLevel(-1)
except ImportError:
    VOSK_AVAILABLE = False


# One loaded model per worker process, keyed by model path. ProcessPool
# workers are reused across tasks, so this caps model loads at MAX_WORKERS
# instead of one per chunk -- Vosk models run 1-2 GB, so reloading them for
# every 10-minute chunk dominated wall time and peak RAM on long audio.
_MODEL_CACHE: Dict[str, object] = {}


def _get_model(model_path: str):
    model = _MODEL_CACHE.get(model_path)
    if model is None:
        model = Model(model_path)
        _MODEL_CACHE[model_path] = model
    return model


def _vosk_worker(task_args: Tuple[str, str, float]) -> List[Dict]:
    model_path, wav_path, offset = task_args
    words: List[Dict] = []
    try:
        model = _get_model(model_path)
        wf = wave.open(wav_path, "rb")
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                for w in res.get("result", []):
                    w["start"] += offset
                    w["end"] += offset
                    words.append(w)
        final_res = json.loads(rec.FinalResult())
        for w in final_res.get("result", []):
            w["start"] += offset
            w["end"] += offset
            words.append(w)
        wf.close()
        del rec          # NB: never del the cached model -- it is reused
    except Exception as exc:
        print(f"[warn] Vosk worker failed at offset {offset}: {exc}")
    return words


def run_vosk_parallel(audio_path: str, model_path: str) -> List[Dict]:
    """Decode *audio_path* with Vosk in parallel 10-minute chunks."""
    print("\n" + "=" * 60 + "\nRUNNING VOSK (PARALLEL MODE)")
    start_time_total = time.time()
    total_duration = get_audio_duration(audio_path)
    # System temp dir, not CWD: the caller's working directory may be
    # read-only, shared, or somewhere the user does not want scratch WAVs.
    temp_dir = tempfile.mkdtemp(prefix="wordalign_vosk_")
    tasks = []
    num_chunks = math.ceil(total_duration / VOSK_CHUNK_SECONDS) or 1
    for i in range(num_chunks):
        start = i * VOSK_CHUNK_SECONDS
        duration = min(VOSK_CHUNK_SECONDS, total_duration - start)
        chunk_filename = os.path.join(temp_dir, f"chunk_{i}.wav")
        create_chunk_wav(audio_path, start, duration, chunk_filename)
        tasks.append((model_path, chunk_filename, start))
    all_results: List[Dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for res in ex.map(_vosk_worker, tasks):
            all_results.extend(res)
    shutil.rmtree(temp_dir, ignore_errors=True)
    all_results.sort(key=lambda x: x["start"])
    for w in all_results:
        w.setdefault("conf", 1.0)
    print(f"[ok] Vosk finished in {time.time() - start_time_total:.1f}s "
          f"({len(all_results)} words).")
    return all_results
