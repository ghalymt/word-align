# WordAlign 2.0 User Guide

## Getting Started

### Quick Start (Simple Mode)

1. Open WordAlign
2. Click **"Create subtitles from audio/video"**
3. Choose your audio or video file
4. (Optional) Attach a verbatim transcript if you have one
5. Select a quality preset:
   - **Fast** — Vosk + WhisperX, no MFA
   - **Balanced** — WhisperX + Qwen + Vosk (recommended)
   - **Best Quality** — All engines + MFA + QA
6. Click **Run Pipeline**
7. Wait for processing to complete
8. Review flagged transcript sections
9. Export final subtitles

### Quality Presets

| Preset | Engines | Speed | Use When |
|---|---|---|---|
| Fast | Vosk + WhisperX | ⚡ Fastest | Quick turnaround, good quality |
| Balanced | WhisperX + Qwen + Vosk | ⚖ Moderate | Default choice for most content |
| Best Quality | All + MFA + QA | 🐌 Slowest | Final deliverable, maximum accuracy |
| CPU Only | Vosk only | ⚡ No GPU needed | No CUDA available |

---

## Models Page

Before running, ensure required models are installed:

1. Go to **Models** page
2. Click **Download** next to each required model
3. Or click **Use Existing...** to point to a model you already have
4. The system validates the directory and shows ✓ or ✗

Models are stored in `WordAlign/models/`. You can change this location
in Settings.

---

## Engines Page

View all registered engines and their status:

- **Ready** (green) — engine is installed and working
- **Missing** (red) — dependencies not installed
- **Not found** (yellow) — optional component missing

Enable/disable engines with the toggle switches.

---

## Pipelines Page (Advanced)

For advanced users who want fine-grained control:

- **Transcription Engines** — enable/disable, adjust vote weights
- **Timing Waterfall Order** — drag to reorder timing sources
- **Segmentation** — adjust CPL, lines, duration limits
- **Smart QA** — enable/disable individual QA signals

Changes are saved to a pipeline profile JSON file.

---

## Review Page

After running a pipeline, the Review page shows:

1. **Waveform** — audio visualization with issue markers
2. **Transcript** — word-by-word text with highlighting:
   - Yellow underline = Low confidence
   - Red underline = Engine disagreement
   - Dashed underline = Interpolated timestamps
3. **Issue Panel** (right side) — list of flagged segments
4. **Playback controls** — play, previous/next issue

Click any timestamp to seek the player to that location.

### Issue Categories

| Category | Description |
|---|---|
| Agreement | Engines disagree on this word |
| Repetition | Repeated phrase (possible hallucination) |
| Timing | Impossible rate, overlap, or interpolation run |
| Semantic | LLM flagged as inconsistent (if enabled) |

### Keyboard Shortcuts

| Key | Action |
|---|---|
| Space | Play/pause |
| Enter | Play current issue |
| Ctrl+Enter | Accept suggestion |
| Delete | Dismiss issue |
| F8 | Next issue |

---

## Diagnostics Page

Shows hardware information and provides:

- **Copy System Report** — for bug reports (no private data included)
- **Open Log Folder** — access application logs

---

## Output Files

After a successful run, you'll find:

| File | Contents |
|---|---|
| `*_word_level.srt` | One cue per word — maximum precision |
| `*_sentence_level.srt` | Merged, balanced subtitle cues |
| `*_transcript.txt` | Ensemble transcript with timestamps |
| `*_transcript.docx` | Word transcript with highlighted low-confidence words |
| `*_audio_tags.srt` | Audio events (laughs, music, etc.) |

---

## CLI Usage

The CLI remains fully functional:

```bash
# Reference mode
python -m wordalign interview.mp4 -t interview.txt --vosk-models ~/vosk-models

# Ensemble mode
python -m wordalign interview.mp4 --engines whisperx,qwen,vosk -l en

# With QA
python -m wordalign interview.mp4 -l en --doc docx
```

The CLI uses the same `PipelineRunner` as the GUI — no separate code path.
