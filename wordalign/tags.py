"""EXPERIMENTAL: audio-event tag detection and placement.

Matches human-annotated ``[tags]`` (e.g. ``[laughs]``, ``[door knock]``)
to detected audio events from WhisperX segment text and Google's YAMNet
AudioSet classifier, then merges/repositions them so they never collide
with dialogue cues.

Status: functional but the least battle-tested part of the pipeline --
word-level and sentence-level SRT are the production outputs. Enable
with ``--tags``.
"""
from __future__ import annotations

import csv
import gc
import re
from collections import defaultdict
from datetime import timedelta
from typing import Dict, List

import srt

from .config import MAX_DURATION_MS, TAG_MAP
from .utils import TIMECODE_TAG_RE

try:
    import librosa
    import psutil
    import tensorflow as tf
    import tensorflow_hub as hub
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def optimize_tags(tag_entries: List[srt.Subtitle],
                  sentence_entries: List[srt.Subtitle]) -> List[srt.Subtitle]:
    """Merge consecutive identical tags with no dialogue in between."""
    if not tag_entries:
        return []
    tags = sorted(tag_entries, key=lambda x: x.start)
    sentences = sorted(sentence_entries, key=lambda x: x.start)
    optimized: List[srt.Subtitle] = []

    def flush(group: List[srt.Subtitle]) -> None:
        start_time = group[0].start
        raw_end = max(t.end.total_seconds() for t in group)
        final_end = min(raw_end, start_time.total_seconds() + 6.0)
        optimized.append(srt.Subtitle(index=0, start=start_time,
                                      end=timedelta(seconds=final_end),
                                      content=group[0].content))

    current_group = [tags[0]]
    for curr_tag in tags[1:]:
        prev_tag = current_group[-1]
        if curr_tag.content == prev_tag.content:
            dialogue_between = any(
                s.start.total_seconds() > (prev_tag.end.total_seconds() - 0.1)
                and s.end.total_seconds() < (curr_tag.start.total_seconds() + 0.1)
                for s in sentences)
            if not dialogue_between:
                current_group.append(curr_tag)
                continue
        flush(current_group)
        current_group = [curr_tag]
    flush(current_group)
    return optimized


def adjust_tag_timing(tag_entries: List[srt.Subtitle],
                      sentence_entries: List[srt.Subtitle]) -> List[srt.Subtitle]:
    """Nudge tags outside dialogue cues (unless fully contained)."""
    buffer = 0.050
    adjusted: List[srt.Subtitle] = []

    def strictly_inside(i_s, i_e, o_s, o_e):
        return i_s >= o_s and i_e <= o_e

    for tag in tag_entries:
        tag_start = tag.start.total_seconds()
        tag_end = tag.end.total_seconds()
        original_dur = tag_end - tag_start
        new_start, new_end = tag_start, tag_end
        for sent in sentence_entries:
            s_start = sent.start.total_seconds()
            s_end = sent.end.total_seconds()
            if strictly_inside(tag_start, tag_end, s_start, s_end):
                new_start, new_end = tag_start, tag_end
                break
            if new_start < s_start and new_end > s_start:
                new_end = s_start - buffer
            if new_start < s_end and new_end > s_end:
                new_start = s_end + buffer
        if new_end <= new_start:
            if new_end != tag_end:
                new_start = max(0, new_end - original_dur)
            elif new_start != tag_start:
                new_end = new_start + original_dur
        adjusted.append(srt.Subtitle(index=0,
                                     start=timedelta(seconds=new_start),
                                     end=timedelta(seconds=new_end),
                                     content=tag.content))
    return adjusted


def combine_and_sort_srt(sentence_entries, tag_entries):
    optimized_tags = optimize_tags(tag_entries, sentence_entries)
    final_tags = adjust_tag_timing(optimized_tags, sentence_entries)
    all_subs = sentence_entries + final_tags
    all_subs.sort(key=lambda x: x.start)
    for i, sub in enumerate(all_subs):
        sub.index = i + 1
    return all_subs


def _load_yamnet():
    print("Loading YAMNet model...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = tf.keras.utils.get_file(
        "yamnet_class_map.csv",
        "https://raw.githubusercontent.com/tensorflow/models/master/"
        "research/audioset/yamnet/yamnet_class_map.csv")
    with open(class_map_path) as fh:
        class_names = [row["display_name"] for row in csv.DictReader(fh)]
    return model, class_names


def detect_audio_tags_yamnet(audio_path: str, human_tags: set,
                             whisperx_events: List[Dict],
                             confidence: float) -> List[Dict]:
    if not TF_AVAILABLE:
        print("[info] tensorflow/tensorflow-hub not installed; "
              "skipping YAMNet.")
        return []
    whisperx_tags = {e["tag"].lower() for e in whisperx_events}
    print("\n" + "=" * 60 + "\nRUNNING YAMNET FOR REMAINING TAGS")
    if psutil.virtual_memory().percent > 90:
        print("[warn] System memory very high; YAMNet may crash.")
    try:
        model, class_names = _load_yamnet()
    except Exception as exc:
        print(f"[warn] Failed to load YAMNet: {exc}")
        return []
    class_names_lower = {name.lower(): name for name in class_names}
    tags_to_search = set()
    for tag in human_tags:
        ltag = tag.lower()
        if ltag in whisperx_tags or any(
                v in whisperx_tags for v in [ltag + "s", ltag.rstrip("s")]):
            continue
        if ltag in TAG_MAP:
            tags_to_search.add(TAG_MAP[ltag])
        elif ltag in class_names_lower:
            tags_to_search.add(class_names_lower[ltag])
    if not tags_to_search:
        print("[ok] YAMNet: no new tags to search for.")
        return []
    print(f"YAMNet searching for: {', '.join(sorted(tags_to_search))}")
    try:
        audio_16k, _ = librosa.load(audio_path, sr=16000)
    except Exception as exc:
        print(f"[warn] librosa load failed: {exc}")
        return []
    scores, _, _ = model(audio_16k)
    del audio_16k
    gc.collect()
    indices = {i for i, name in enumerate(class_names)
               if name in tags_to_search}
    events: List[Dict] = []
    step = 0.48
    scores_np = scores.numpy()
    for i, frame in enumerate(scores_np):
        for index in indices:
            if frame[index] > confidence:
                events.append({"start": i * step, "end": i * step + step,
                               "tag": class_names[index], "source": "YAMNet"})
    del scores, scores_np, model
    gc.collect()
    if not events:
        return []
    events.sort(key=lambda x: x["start"])
    merged = []
    current = events[0]
    for event in events[1:]:
        if (event["tag"] == current["tag"]
                and (event["start"] - current["end"]) < step * 1.5):
            current["end"] = event["end"]
        else:
            merged.append(current)
            current = event
    merged.append(current)
    max_duration = MAX_DURATION_MS / 1000.0
    for event in merged:
        if event["end"] - event["start"] > max_duration:
            event["end"] = event["start"] + max_duration
    print(f"[ok] YAMNet detected {len(merged)} events.")
    return merged


def match_human_tags_to_detections(human_text: str,
                                   detected_events: List[Dict]) -> List[Dict]:
    print("\nMatching detected events to transcript tags (1-to-1)...")
    human_tags_in_order = [
        {"text": m.group(1).strip().lower(), "original": m.group(1).strip()}
        for m in re.finditer(r"\[(.*?)\]", human_text)
        if not TIMECODE_TAG_RE.match(m.group(1).strip())
        and "END OF AUDIO" not in m.group(1)]
    print(f"  Found {len(human_tags_in_order)} tag instances in transcript.")
    if not human_tags_in_order:
        return []
    detected_pool = defaultdict(list)
    for event in detected_events:
        key = event["tag"].lower()
        detected_pool[key].append(event)
        for human_ver, generic_ver in TAG_MAP.items():
            if generic_ver.lower() == key:
                detected_pool[human_ver].append(event)
    matched_events: List[Dict] = []
    unmatched_count = 0
    for human_tag in human_tags_in_order:
        tag_text = human_tag["text"]
        found = False
        words = re.split(r"[-\s]", tag_text)
        variations = {tag_text}
        for word in words:
            variations.update({word, word.rstrip("s"), word + "s"})
        for var in sorted(variations):
            if detected_pool.get(var):
                event = detected_pool[var].pop(0)
                matched_events.append({"start": event["start"],
                                       "end": event["end"],
                                       "tag": human_tag["original"],
                                       "source": event.get("source")})
                found = True
                break
        if not found:
            unmatched_count += 1
    print(f"  [ok] Matched {len(matched_events)} of "
          f"{len(human_tags_in_order)} tag instances.")
    if unmatched_count:
        print(f"  - No match for {unmatched_count} instances.")
    return matched_events
