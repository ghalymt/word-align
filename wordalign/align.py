"""The alignment waterfall: match human words to engine timestamps.

Stages (cheapest first, each only touching still-unmatched words):
    1. Vosk               -- fast CPU pass, typically covers ~90% of words
    2. Rough SRT          -- optional, linear interpolation inside cues
    3. WhisperX           -- GPU pass with wav2vec2 forced alignment
    4. MFA surgical mode  -- forced alignment run only on the leftover gaps
    5. Interpolation      -- mathematical fill for whatever remains
"""
from __future__ import annotations

import tempfile
from difflib import SequenceMatcher
from typing import Dict, List

from .engines.mfa_engine import MFAWrapper
from .utils import create_chunk_wav, get_audio_duration, normalize_word


def match_timestamps(aligned_words: List[Dict], source_words: List[Dict],
                     source_name: str, human_words_norm: List[str]) -> Dict:
    """Copy timestamps from *source_words* onto still-unmatched entries."""
    if not source_words:
        return {"matched": 0, "total": len(aligned_words)}
    source_norm = [normalize_word(w["word"]) for w in source_words]
    matcher = SequenceMatcher(None, human_words_norm, source_norm,
                              autojunk=False)
    count = 0
    for match in matcher.get_matching_blocks():
        for offset in range(match.size):
            h_idx = match.a + offset
            s_idx = match.b + offset
            if h_idx >= len(aligned_words):
                continue
            if not aligned_words[h_idx].get("matched", False):
                aligned_words[h_idx].update({
                    "start": source_words[s_idx]["start"],
                    "end": source_words[s_idx]["end"],
                    "matched": True,
                    "source": source_name,
                })
                count += 1
    pct = (count / len(aligned_words) * 100) if aligned_words else 0.0
    print(f"[ok] {source_name}: matched {count}/{len(aligned_words)} words "
          f"({pct:.1f}%)")
    return {"matched": count, "total": len(aligned_words), "percentage": pct}


def run_mfa_on_gaps(aligned_words: List[Dict], audio_path: str,
                    mfa_wrapper: MFAWrapper) -> None:
    """Force-align only the contiguous runs of unmatched words."""
    print("\n" + "=" * 60 + "\nRUNNING MFA ON MISSING TIMESTAMPS (SURGICAL MODE)")
    gaps: List[List[int]] = []
    current_gap: List[int] = []
    for i, word in enumerate(aligned_words):
        if not word.get("matched", False):
            current_gap.append(i)
        elif current_gap:
            gaps.append(current_gap)
            current_gap = []
    if current_gap:
        gaps.append(current_gap)
    if not gaps:
        print("[ok] No gaps to align with MFA.")
        return
    print(f"Found {len(gaps)} gaps covering "
          f"{sum(len(g) for g in gaps)} words.")

    chunks_metadata: List[Dict] = []
    padding = 0.25
    audio_duration = get_audio_duration(audio_path)
    for gap_indices in gaps:
        first_idx, last_idx = gap_indices[0], gap_indices[-1]
        start_time = (aligned_words[first_idx - 1].get("end", 0.0)
                      if first_idx > 0 else 0.0)
        end_time = (aligned_words[last_idx + 1].get("start", audio_duration)
                    if last_idx < len(aligned_words) - 1 else audio_duration)
        chunk_start = max(0, start_time - padding)
        chunk_end = end_time + padding
        if chunk_end <= chunk_start:
            continue
        text = " ".join(aligned_words[i]["word"] for i in gap_indices)
        chunks_metadata.append({
            "filename": f"gap_{first_idx}_{last_idx}",
            "text": text,
            "start_offset": chunk_start,
            "end_offset": chunk_end,
            "word_indices": gap_indices,
        })
    if not chunks_metadata:
        return

    print(f"Extracting {len(chunks_metadata)} audio slices for MFA...")
    mfa_wrapper.corpus_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks_metadata:
        wav_path = mfa_wrapper.corpus_dir / f"{chunk['filename']}.wav"
        lab_path = mfa_wrapper.corpus_dir / f"{chunk['filename']}.lab"
        create_chunk_wav(audio_path, chunk["start_offset"],
                         chunk["end_offset"] - chunk["start_offset"],
                         str(wav_path))
        lab_path.write_text(chunk["text"], encoding="utf-8")

    print("Running MFA alignment...")
    try:
        words = mfa_wrapper.detect_oov_words()
        dict_path = mfa_wrapper.generate_custom_dictionary(words)
        mfa_wrapper.run_alignment(str(dict_path))
        import textgrid
        updated_count = 0
        for chunk in chunks_metadata:
            tg_path = mfa_wrapper.output_dir / f"{chunk['filename']}.TextGrid"
            if not tg_path.exists():
                continue
            try:
                tg = textgrid.TextGrid.fromFile(str(tg_path))
                word_tier = tg.getFirst("words")
                mfa_intervals = [i for i in word_tier if i.mark]
                indices = chunk["word_indices"]
                if len(mfa_intervals) == len(indices):
                    for idx, interval in zip(indices, mfa_intervals):
                        aligned_words[idx].update({
                            "start": chunk["start_offset"] + interval.minTime,
                            "end": chunk["start_offset"] + interval.maxTime,
                            "matched": True,
                            "source": "MFA",
                        })
                        updated_count += 1
            except Exception as exc:
                print(f"[warn] Failed to parse TextGrid {tg_path}: {exc}")
        print(f"[ok] MFA force-aligned {updated_count} words.")
    except Exception as exc:
        print(f"[warn] MFA failed: {exc}")
    mfa_wrapper.cleanup()


def interpolate_timestamps(aligned_words: List[Dict]) -> None:
    """Fill any remaining words by linear interpolation between neighbors."""
    print("Interpolating timestamps for any remaining words...")
    interpolated_count = 0
    for i, word_data in enumerate(aligned_words):
        if word_data.get("start") is not None:
            continue
        prev_w = next_w = None
        prev_idx = next_idx = -1
        for j in range(i - 1, -1, -1):
            if aligned_words[j].get("end") is not None:
                prev_w, prev_idx = aligned_words[j], j
                break
        for j in range(i + 1, len(aligned_words)):
            if aligned_words[j].get("start") is not None:
                next_w, next_idx = aligned_words[j], j
                break
        if prev_w and next_w:
            gap = i - prev_idx
            total_gap = next_idx - prev_idx
            time_per_word = ((next_w["start"] - prev_w["end"]) / total_gap
                             if total_gap > 0 else 0)
            word_data["start"] = prev_w["end"] + time_per_word * gap
            word_data["end"] = word_data["start"] + time_per_word
        elif prev_w:
            word_data["start"] = prev_w["end"] + 0.05
            word_data["end"] = word_data["start"] + 0.25
        else:
            word_data["start"] = i * 0.3
            word_data["end"] = word_data["start"] + 0.25
        word_data["source"] = "Interpolated"
        interpolated_count += 1
    print(f"[ok] Interpolated {interpolated_count} words")


def print_alignment_statistics(aligned_words: List[Dict]) -> None:
    print("\n" + "=" * 60 + "\nALIGNMENT STATISTICS")
    sources: Dict[str, int] = {}
    for word in aligned_words:
        source = word.get("source", "Unmatched")
        sources[source] = sources.get(source, 0) + 1
    total = len(aligned_words)
    print(f"Total words: {total}")
    for source in sorted(sources):
        count = sources[source]
        pct = (count / total * 100) if total else 0
        print(f"  {source}: {count} ({pct:.1f}%)")


def make_surgical_mfa(aligned_words: List[Dict], audio_path: str,
                      mfa_cmd=None) -> None:
    """Convenience: build a temp MFAWrapper and run gap alignment."""
    try:
        temp_dir = tempfile.mkdtemp(prefix="mfa_surgical_")
        wrapper = MFAWrapper(work_dir=temp_dir, mfa_cmd=mfa_cmd)
        run_mfa_on_gaps(aligned_words, audio_path, wrapper)
    except Exception as exc:
        print(f"[warn] Surgical MFA skipped: {exc}")
