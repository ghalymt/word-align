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

### 2.2.0 (quick suite, RTX 4070 Ti, empty cache per run)

| | 2.1 | 2.2 defaults | robust |
|---|---|---|---|
| Word-start error, reference mode | 868 ms | 69 ms | 70 ms |
| Within 100 ms, reference mode | 83.0% | 90.1% | 90.1% |
| WER, free mode | 16.9% | 15.0% | 15.3% |
| Word-start error, free mode | 225 ms | 73 ms | 106 ms |
| Invented words / 10 min | 300 | 0 | 0 |
| Triage recall / Accept cues with errors, free mode | – | 88% / 5% | 77% / 9% |
| Triage recall / Accept cues with errors, reference mode | – | 50% / 24% | 63% / 25% |

| WER by condition (free mode) | 2.1 | 2.2 | robust |
|---|---|---|---|
| clean | 3.7% | 2.4% | 2.4% |
| noise 5 dB | 0.0% | 0.0% | 0.0% |
| music 0 dB (sung vocals) | 0.0% | 13.3% | 12.0% |
| babble 0 dB | 72.6% | 84.2% | 87.4% |
| reverb 1.0 s | 43.4% | 2.6% | 2.6% |
| phone (μ-law) | 3.8% | 1.3% | 1.3% |
| second talker −5 dB | 1.0% | 1.0% | 1.0% |

| Word-start error by condition (reference mode) | 2.1 | 2.2 | robust |
|---|---|---|---|
| clean | 47 ms | 36 ms | 36 ms |
| noise 5 dB | 28 ms | 32 ms | 32 ms |
| music 0 dB | 182 ms | 85 ms | 85 ms |
| babble 0 dB | 5761 ms | 205 ms | 205 ms |
| reverb 1.0 s | 44 ms | 39 ms | 39 ms |
| phone (μ-law) | 27 ms | 25 ms | 25 ms |
| second talker −5 dB | 46 ms | 62 ms | 62 ms |

What this says: anchors + forced alignment fix the large timing failures
(babble, music, reverb), the VAD/voting stages stop invented text on
non-speech, and the review queue catches most erroneous cues when there is
no transcript. Under loud music with sung vocals the voters transcribe the
lyrics too, and with a reference transcript the review queue still misses
about half of the cues whose timing is off by more than 250 ms. On this
synthetic suite the robust profile's extra voters and cleanup did not beat
the defaults; run the benchmark on recordings like yours before choosing.
