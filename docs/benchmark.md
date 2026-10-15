# Measuring the robust flow: `wordalign-bench`

`wordalign-bench` builds a test recording from public data, degrades it in
known ways, runs WordAlign over it in several configurations and scores the
results against the true words and word times. It answers "did this change
make bad audio better or worse, and by how much?".

```bash
pip install -e ".[bench]"          # huggingface_hub, pyarrow, soundfile

wordalign-bench sources --dest bench/sources              # download once (~1 GB)
wordalign-bench build   --sources bench/sources --out bench/suite --quick
wordalign-bench run     --suite bench/suite --out bench/out \
                        --variants v2.1,v2.2,robust --modes reference,free --fresh-cache
wordalign-bench report  --out bench/out                   # re-render report.md
```

`python -m wordalign.robustness ...` is the same command.

## The data

| Source | Used for | License |
|---|---|---|
| [gilkeyio/librispeech-alignments](https://huggingface.co/datasets/gilkeyio/librispeech-alignments) (`test_clean`) | speech with word times; 12 speakers, half female | CC-BY-4.0 |
| [FluidInference/musan](https://huggingface.co/datasets/FluidInference/musan) (`noise/free-sound`) | noise | see MUSAN (free-sound clips) |
| [benjamin-paine/free-music-archive-commercial-16khz-full](https://huggingface.co/datasets/benjamin-paine/free-music-archive-commercial-16khz-full) | music | per-track Creative Commons licenses allowing commercial use |

Everything is cached under `--dest`; later runs need no network.

## The suite

`build` joins one speaker's utterances into a natural passage (0.4–1.2 s
pauses) for each condition block and degrades it; blocks are separated by
1 s of silence and written as one long file, so the pipeline sees what it
sees on a real job. In `bench.txt` every utterance is one line, in the order
the utterances start, as a human transcript writes crosstalk. The outputs are `bench.wav`, `bench.txt` (the true
transcript, for reference mode), `bench.words.json` (true word times) and
`bench.blocks.json`.

| Condition | What is done to the speech |
|---|---|
| `clean` | nothing |
| `noise_10dB`, `noise_5dB`, `noise_0dB` | MUSAN noise at that SNR |
| `music_10dB` … `music_-5dB` | FMA music at that SNR |
| `babble_5dB`, `babble_0dB` | six other LibriSpeech talkers |
| `reverb_0.6s`, `reverb_1.0s` | synthetic room reverb (RT60) |
| `phone_mulaw8k`, `voip_opus12k` | telephone μ-law at 8 kHz; Opus at 12 kbit/s |
| `overlap_5dB`, `overlap_0dB` | a second speaker talking over; their words are scored separately |
| `clipped` | 12 dB of gain, hard-clipped at the original peak |
| `only_music`, `only_noise`, `only_babble`, `silence` | no speech at all: every word written there is invented |

`--quick` uses nine of them (clean, noise 5 dB, music 0 dB, babble 0 dB,
reverb 1.0 s, phone, overlap 5 dB, only music, only noise), about five
minutes of audio. `--conditions a,b,c` picks your own; `--block-seconds`
sets the passage length; `--seed` changes the draw.

## Variants and modes

| Variant | Flags |
|---|---|
| `v2.1` | `--timing fill_only --no-vad --no-hallucination-filter --no-triage` — the 2.1 behaviour, run by the 2.2 code |
| `v2.2` | defaults: analysis, anchors + forced alignment, triage; engines WhisperX, Qwen, Vosk |
| `robust` | `--profile robust` |

Add your own with `--variant NAME="--flags ..."`. Each variant runs in
**reference** mode (the true transcript is given; only timing is scored)
and **free** mode (no transcript; words and timing are scored).
`--fresh-cache` gives every run an empty stage cache so the run times are
honest; without it later variants reuse earlier engine output.

## The metrics

| Metric | Meaning |
|---|---|
| WER | word error rate of the main speaker in the speech blocks (free mode), after lower-casing and stripping punctuation; words that belong to a second talker are neither errors nor required |
| Mean shift | mean distance between a main-speaker word's true start and its output start, over matched words (ms) |
| Within 100 ms | share of matched words that start within 100 ms of the truth |
| Invented words / 10 min | words written in the no-speech blocks, per 10 minutes of no-speech audio (free mode) |
| Triage recall | of the cues that contain an error, the share the review queue flagged (*Quick check* or *Redo*) |
| Accept cues with errors | share of *Accept* cues that do contain an error (lower is better) |
| Run time | wall-clock seconds for the whole file |

`report.md` has the overall table, then per-condition tables for WER and
timing, the second talker in overlap blocks (share of their words found,
and their timing), and the invented words per no-speech block. `report.json` holds
everything, including each run's per-stage seconds; each run's own log and
output files are under `<out>/<variant>/<mode>/`.

## Calibrating the review tiers

The benchmark's triage numbers use the default weights. To fit the weights
to your own work instead, correct some real jobs (the GUI review queue
writes `<name>_corrected.srt`) and run:

```bash
wordalign-bench fit-triage --jobs FOLDER [--dry-run]
```

## Results

See the release notes of each version for the numbers measured on the
release candidate.
