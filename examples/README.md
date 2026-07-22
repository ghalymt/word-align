# Quick trial

Any public-domain clip works. For example, download a LibriVox recording
(public domain audiobooks), keep the first two minutes, and run:

```bash
ffmpeg -i chapter1.mp3 -t 120 -ar 16000 -ac 1 clip.wav

# transcript-free:
python -m wordalign clip.wav -l en --engines whisperx,vosk

# with the book text as reference:
python -m wordalign clip.wav -t clip_text.txt
```

Compare `clip_word_level.srt` against the audio in a subtitle editor
(e.g. Subtitle Edit / Aegisub) — every word cue should sit on its spoken
onset within a few tens of milliseconds.
