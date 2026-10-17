# WordAlign 2.2 User Guide

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
| Robust | Analysis + gated cleanup + 4 voting engines + forced alignment + review queue | 🎧 Slow, GPU | Music, crowd or street noise, phone audio, crosstalk ([guide](robust_flow.md)) |

Parts of the Robust preset that are not installed are skipped with a
message. Run `WordAlign.exe --doctor` (or `wordalign --doctor`) to see what
is installed, and `scripts\install_robust.ps1` to add the rest.

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

## Review queue (step 4)

Every run with the robust stages (the Robust and Maximum Quality presets,
and the CLI by default) sorts the cues into three tiers and writes
`<name>_review.json`:

| Tier | Meaning | What to do |
|---|---|---|
| **Redo** | a large part of the cue is probably wrong | retype it |
| **Quick check** | something about the cue is uncertain | listen once; most need no change |
| **Accept** | nothing suggests an error | skip |

When the job finishes, step 4 *Review* appears under the output files:

1. Pick **To review** (Redo + Quick check), **Redo only** or **All cues**,
   in time order or riskiest first.
2. Each cue shows its tier, the reasons ("voters disagree on 2 words",
   "music under speech", "0.4 s interpolated"...), and its words coloured
   by confidence: red = the engines disagree, amber = weak agreement,
   dotted underline = the timing is a guess. Hover a word to see what the
   other engines heard.
3. **▶ Loop** plays the cue (with a little context) over and over.
4. Fix the text. A chip such as `sale → sailed` applies another engine's
   reading with one click.
5. **✓ Accept** saves the cue; **↺ Undo** returns it to unreviewed.

| Key (in a cue's text box) | Action |
|---|---|
| Esc | Play / stop the loop |
| Ctrl+Enter | Accept and move to the next cue |

Your work is saved in the job folder (`%USERPROFILE%\WordAlign\outputs\<job>`):
`<name>_corrected.srt` holds the subtitles with your fixes; the files the
pipeline wrote are never changed. `<name>_review_edits.json` and
`<name>_review_log.jsonl` record what you did, and
`wordalign-bench fit-triage --jobs "%USERPROFILE%\WordAlign\outputs"`
uses them to tune the tiers to your material. Reloading the page reopens
the last review.

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
| `*_sentence_level.srt` | Merged, balanced subtitle cues — speech only |
| `*_transcript.txt` | Ensemble transcript with timestamps |
| `*_transcript.docx` | Word transcript with highlighted low-confidence words |
| `*_audio_tags.srt` | Sound tags (laughs, music, etc.) with the time each is on screen (`--tags`) |
| `*_combined.srt` | Speech and sound tags together (`--tags`) |
| `*_final.srt` | **The deliverable** — every word and every `[tag]` of the transcript, within the layout and timing rules (2.2.1) |
| `*_review.json` | Review tiers, reasons and word confidences (2.2) |
| `*_corrected.srt` | The final SRT with your corrections from the review queue (GUI) |

### The final SRT

`*_final.srt` is written on every run, after the combined file. It holds:

- **every word and every `[tag]`** of the transcript, in the editor's order
  (a transcript's timecodes and `[END OF AUDIO]` are not tags);
- at most `--max-lines` lines of `--cpl` characters (spaces and punctuation
  count), one line whenever the text fits one, line breaks that keep phrases
  together (subject + verb, auxiliaries, phrasal verbs, names, numbers and
  units — standard subtitling rules for English), sound tags at a cue's edge on a
  line of their own;
- cues of 1 to 7 s (`--min-cue-ms`, `--max-duration-ms`) with at least
  1 ms between them and no overlaps; long transcript lines are split at their
  best break; fast cues use the pause around them (and a little time from a
  slower neighbour) to stay near 20 characters per second.

**Speech timing always wins.** A tag between sentences gets a cue of its own
filling the pause where the transcript puts it (timed by a matching sound
event when there is one; an event that runs into speech is cut back to that
pause). A tag inside a sentence stays inline in that sentence's cue, and a tag
whose pause is under half a second shares the cue before (or after) it.

**What wins when rules collide.** Speech pace can't be controlled, so the
rules adapt to it, in this order:

1. every word and tag of the transcript, in order;
2. sync with the audio -- a cue never starts before its first word or after
   it, and never overlaps the next one;
3. the layout (`--max-lines` × `--cpl`);
4. on-screen time and reading speed (`--min-cue-ms`, 7 s, ~20 characters per
   second): met wherever the pauses allow, and bent where the speaker is
   simply too fast -- a quick "Okay." between two sentences keeps its short
   cue rather than showing early or late.

**Timing style.** Cues start with their first word, run on to the next cue
when the pause between them is up to 1 s (`--chain-gap-ms`), and linger
250 ms after a longer pause (`--lead-out-ms`). A short sentence such as
"Okay." keeps its own cue in sync with the words, even under `--min-cue-ms`
(reported as "kept in sync"); `--no-sync-starts` instead lets it start a
little early, or join a neighbour, to reach the minimum. The run
checks the finished file and prints the result (`Final SRT -> …: all N words
and M tags present, all rules met, …`); the same report is stored under
`final_check` in the job manifest, and anything that still breaks a rule is
listed as a warning.

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

# Bad audio: the robust flow
python -m wordalign street_interview.mp4 --profile robust -l en
```

The CLI uses the same `PipelineRunner` as the GUI — no separate code path.
