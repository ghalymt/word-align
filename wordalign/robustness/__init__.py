"""Bad-audio benchmark suite (``wordalign-bench``).

Every robustness change is accepted or rejected on numbers measured here:

* ``degrade``  -- noise, music, babble, crowd, reverb, codecs, overlap and
  clipping applied to clean labelled speech at controlled levels;
* ``corpus``   -- labelled items (audio + word timings or a transcript);
* ``suite``    -- builds a suite folder with a manifest;
* ``metrics``  -- WER, word-time shift, invented words on non-speech,
  triage recall and safety;
* ``runner``   -- runs pipeline variants over a suite and writes a report.

See docs/benchmark.md for the workflow.
"""
