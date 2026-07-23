"""word-align: word-accurate subtitle timing via multi-engine alignment."""
# 0.9.2: abbreviation-aware sentence splitting (St./Mr./Dr. no longer break a
# cue at the period) and source-aware cue-overlap resolution that never moves
# a Vosk timestamp. 0.9.1 fixed the Windows multiprocessing entry-point guard
# and verified reference mode end-to-end on an RTX 4070 Ti. Held below 1.0.0
# until the ensemble mode's NeMo voters are exercised on real audio too.
__version__ = "0.9.2"
