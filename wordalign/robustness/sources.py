"""Public audio for the bad-audio benchmark, downloaded on demand.

* **Speech with word timings**: LibriSpeech test-clean with Montreal Forced
  Aligner word alignments, ``gilkeyio/librispeech-alignments`` on Hugging
  Face (CC-BY-4.0; LibriSpeech itself is CC-BY-4.0).
* **Noise**: the free-sound part of MUSAN, ``FluidInference/musan``
  (Creative Commons / public domain).
* **Music**: Free Music Archive tracks cleared for commercial use, 16 kHz,
  ``benjamin-paine/free-music-archive-commercial-16khz-full`` (CC-BY-4.0).

Everything is cached under the sources folder (``<dest>``), as 16 kHz mono
WAV files plus ``<name>.words.json`` truth files, so the suite builder needs
no network after the first run. Needs ``huggingface_hub``, ``pyarrow`` and
``soundfile`` (``pip install wordalign[bench]``).
"""
from __future__ import annotations

import io
import json
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

SR = 16_000
LIBRISPEECH_REPO = "gilkeyio/librispeech-alignments"
MUSAN_REPO = "FluidInference/musan"
FMA_REPO = "benjamin-paine/free-music-archive-commercial-16khz-full"


def _need(module: str):
    try:
        return __import__(module)
    except ImportError as exc:
        raise RuntimeError(
            f"the benchmark needs '{module}' (pip install wordalign[bench])") from exc


def _decode(blob: bytes) -> np.ndarray:
    """Decode an audio file held in memory to 16 kHz mono float32."""
    sf = _need("soundfile")
    data, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != SR:
        librosa = _need("librosa")
        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    return np.asarray(data, dtype=np.float32)


def _write(path: Path, audio: np.ndarray) -> None:
    from ..audio.io import write_wav
    write_wav(str(path), audio)


def fetch_librispeech(dest: str, split: str = "test_clean", speakers: int = 12,
                      per_speaker: int = 12, seed: int = 0,
                      log: Callable[[str], None] = print) -> List[Dict]:
    """Utterances with word timings; returns their metadata.

    Picks *speakers* speakers (balanced by sex) and up to *per_speaker*
    utterances each, so suites can join one speaker's utterances into a
    natural-sounding passage.
    """
    out_dir = Path(dest) / "speech"
    index_path = out_dir / "index.json"
    if index_path.is_file():
        return json.loads(index_path.read_text(encoding="utf-8"))
    hub = _need("huggingface_hub")
    pq = _need("pyarrow.parquet")
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {LIBRISPEECH_REPO} ({split}) ...")
    path = hub.hf_hub_download(LIBRISPEECH_REPO, f"data/{split}-00000-of-00001.parquet",
                               repo_type="dataset", cache_dir=str(Path(dest) / "hf"))
    table = pq.read_table(path, columns=["id", "sex", "words", "audio", "transcript"])
    rows = table.to_pylist()
    by_spk: Dict[str, List[Dict]] = {}
    for r in rows:
        by_spk.setdefault(r["id"].split("-")[0], []).append(r)
    rng = random.Random(seed)
    spk_ids = sorted(by_spk)
    rng.shuffle(spk_ids)
    male = [s for s in spk_ids if by_spk[s][0]["sex"].upper().startswith("M")]
    female = [s for s in spk_ids if s not in male]
    chosen = []
    while len(chosen) < speakers and (male or female):
        for pool in (female, male):
            if pool and len(chosen) < speakers:
                chosen.append(pool.pop(0))
    meta: List[Dict] = []
    for spk in chosen:
        utts = sorted(by_spk[spk], key=lambda r: r["id"])[:per_speaker]
        for r in utts:
            audio = _decode(r["audio"]["bytes"])
            words = [{"word": w["word"], "start": round(float(w["start"]), 3),
                      "end": round(float(w["end"]), 3)}
                     for w in r["words"] if str(w.get("word", "")).strip()]
            if not words:
                continue
            name = r["id"]
            _write(out_dir / f"{name}.wav", audio)
            (out_dir / f"{name}.words.json").write_text(json.dumps(words), encoding="utf-8")
            meta.append({"id": name, "speaker": spk, "sex": r["sex"],
                         "seconds": round(len(audio) / SR, 3),
                         "text": r["transcript"]})
    index_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"  {len(meta)} utterances from {len(chosen)} speakers")
    return meta


def fetch_noise(dest: str, count: int = 30, seed: int = 0,
                log: Callable[[str], None] = print) -> List[str]:
    """MUSAN free-sound noise clips as 16 kHz WAV; returns their paths."""
    out_dir = Path(dest) / "noise"
    existing = sorted(str(p) for p in out_dir.glob("*.wav")) if out_dir.is_dir() else []
    if len(existing) >= count:
        return existing[:count]
    hub = _need("huggingface_hub")
    api = hub.HfApi()
    files = [f for f in api.list_repo_files(MUSAN_REPO, repo_type="dataset")
             if f.startswith("noise/free-sound/") and f.lower().endswith(".wav")]
    rng = random.Random(seed)
    rng.shuffle(files)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {count} noise clips from {MUSAN_REPO} ...")
    paths = []
    for f in files[:count]:
        local = hub.hf_hub_download(MUSAN_REPO, f, repo_type="dataset",
                                    cache_dir=str(Path(dest) / "hf"))
        audio = _decode(Path(local).read_bytes())
        target = out_dir / (Path(f).stem + ".wav")
        _write(target, audio)
        paths.append(str(target))
    return paths


def fetch_music(dest: str, count: int = 24, seed: int = 0, max_seconds: float = 90.0,
                log: Callable[[str], None] = print) -> List[str]:
    """Free Music Archive tracks (16 kHz) as WAV; returns their paths.

    Reads only the first parquet shard's first row group(s), so the
    download stays around the size of one shard.
    """
    out_dir = Path(dest) / "music"
    existing = sorted(str(p) for p in out_dir.glob("*.wav")) if out_dir.is_dir() else []
    if len(existing) >= count:
        return existing[:count]
    hub = _need("huggingface_hub")
    pq = _need("pyarrow.parquet")
    log(f"Downloading music from {FMA_REPO} (one shard) ...")
    shard = hub.hf_hub_download(FMA_REPO, "data/train-00000-of-00013.parquet",
                                repo_type="dataset", cache_dir=str(Path(dest) / "hf"))
    pf = pq.ParquetFile(shard)
    audio_col = next((n for n in pf.schema_arrow.names if n in ("audio", "track", "mp3")),
                     pf.schema_arrow.names[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    paths: List[str] = []
    for g in range(pf.num_row_groups):
        rows = pf.read_row_group(g, columns=[audio_col]).to_pylist()
        rng.shuffle(rows)
        for k, r in enumerate(rows):
            blob = r[audio_col]
            if isinstance(blob, dict):
                blob = blob.get("bytes")
            if not blob:
                continue
            try:
                audio = _decode(blob)
            except Exception:
                continue
            if len(audio) < 20 * SR:
                continue
            audio = audio[:int(max_seconds * SR)]
            target = out_dir / f"fma_{g:02d}_{k:04d}.wav"
            _write(target, audio)
            paths.append(str(target))
            if len(paths) >= count:
                return paths
    return paths


def load_wavs(paths: List[str], limit: Optional[int] = None) -> List[np.ndarray]:
    from ..audio.io import read_wav
    out = []
    for p in paths[:limit] if limit else paths:
        try:
            out.append(read_wav(p))
        except Exception:
            continue
    return out


def ensure_sources(dest: str, log: Callable[[str], None] = print,
                   speakers: int = 12, per_speaker: int = 12) -> Dict[str, object]:
    """Download (once) and return the speech index, noise and music paths."""
    os.makedirs(dest, exist_ok=True)
    return {
        "speech": fetch_librispeech(dest, speakers=speakers, per_speaker=per_speaker, log=log),
        "noise": fetch_noise(dest, log=log),
        "music": fetch_music(dest, log=log),
        "dest": dest,
    }
