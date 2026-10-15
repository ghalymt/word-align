"""Stage 1-2 of the robust flow: measures, VAD planning, quality map, cleanup plan."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.audio import measures
from wordalign.audio.analysis import ChunkInfo, QualityMap, fixed_chunks
from wordalign.audio.io import read_wav, write_wav
from wordalign.audio.prep import _crossfade_insert, plan_cleanup
from wordalign.audio.vad import (HOP_S, _cut_long, max_prob, mean_prob,
                                 plan_asr_chunks, speech_segments)

SR = 16000


def _speechlike(seconds, rng):
    """Bursts of Laplacian-amplitude "syllables" with short pauses."""
    n = int(seconds * SR)
    x = np.zeros(n, dtype=np.float32)
    t = 0
    while t < n:
        length = int(rng.uniform(0.12, 0.3) * SR)
        env = np.hanning(min(length, n - t))
        x[t:t + len(env)] = rng.laplace(0, 0.2, len(env)) * env
        t += length + int(rng.uniform(0.02, 0.15) * SR)
    return x


class TestMeasures(unittest.TestCase):

    def test_wada_snr_falls_as_noise_rises(self):
        rng = np.random.default_rng(1)
        clean = _speechlike(8.0, rng)
        estimates = []
        for snr in (30, 15, 5, 0):
            noise = rng.standard_normal(clean.size).astype(np.float32)
            gain = np.sqrt(np.mean(clean ** 2) / (np.mean(noise ** 2) * 10 ** (snr / 10)))
            estimates.append(measures.wada_snr(clean + gain * noise))
        self.assertEqual(estimates, sorted(estimates, reverse=True), estimates)
        self.assertGreater(estimates[0] - estimates[-1], 15)

    def test_silence_has_no_snr(self):
        self.assertTrue(np.isnan(measures.wada_snr(np.zeros(SR, np.float32))))

    def test_narrowband_detects_telephone_audio(self):
        rng = np.random.default_rng(2)
        wide = rng.standard_normal(4 * SR).astype(np.float32)
        spec = np.fft.rfft(wide)
        freqs = np.fft.rfftfreq(wide.size, 1 / SR)
        spec[freqs > 3600] = 0
        narrow = np.fft.irfft(spec, wide.size).astype(np.float32)
        self.assertFalse(measures.is_narrowband(wide))
        self.assertTrue(measures.is_narrowband(narrow))

    def test_clipping_ratio(self):
        t = np.arange(SR) / SR
        sine = 0.8 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        clipped = np.clip(sine * 4, -0.8, 0.8)
        self.assertLess(measures.clipping_ratio(sine), 0.001)
        self.assertGreater(measures.clipping_ratio(clipped), 0.3)


class TestVadPlanning(unittest.TestCase):

    def _probs(self, spans, seconds):
        n = int(seconds / HOP_S)
        p = np.full(n, 0.02, dtype=np.float32)
        for s, e in spans:
            p[int(s / HOP_S):int(e / HOP_S)] = 0.95
        return p

    def test_segments_follow_the_speech(self):
        probs = self._probs([(1.0, 3.0), (5.0, 6.0)], 8.0)
        segs = speech_segments(probs, duration=8.0)
        self.assertEqual(len(segs), 2)
        self.assertAlmostEqual(segs[0][0], 1.0, delta=0.1)
        self.assertAlmostEqual(segs[1][1], 6.0, delta=0.1)

    def test_short_blips_are_dropped(self):
        probs = self._probs([(1.0, 1.1), (3.0, 5.0)], 6.0)
        segs = speech_segments(probs, duration=6.0, min_speech_s=0.25)
        self.assertEqual(len(segs), 1)

    def test_chunks_respect_max_len_and_gaps(self):
        spans = [(i * 4.0, i * 4.0 + 3.0) for i in range(20)]       # 80 s
        probs = self._probs(spans + [(85.0, 88.0)], 90.0)
        segs = speech_segments(probs, duration=90.0)
        chunks = plan_asr_chunks(segs, probs, 90.0, max_len=28.0, max_gap=2.0)
        for s, e in chunks:
            self.assertLessEqual(e - s, 28.0 + 0.5)
        # The 5 s silence before 85 s is never inside a chunk.
        self.assertTrue(any(abs(s - 85.0) < 0.5 for s, _ in chunks), chunks)
        for s, e in chunks:
            self.assertFalse(s < 82.0 < e and s < 84.0 < e, (s, e))

    def test_long_speech_is_cut_at_the_quietest_frame(self):
        probs = np.full(int(60 / HOP_S), 0.9, dtype=np.float32)
        probs[int(20.0 / HOP_S)] = 0.1                    # a pause at 20 s
        pieces = _cut_long(0.0, 60.0, probs, 28.0, HOP_S)
        self.assertAlmostEqual(pieces[0][1], 20.0, delta=0.1)
        self.assertEqual(pieces[-1][1], 60.0)

    def test_mean_and_max_probability(self):
        probs = self._probs([(1.0, 2.0)], 4.0)
        self.assertGreater(mean_prob(probs, 1.0, 2.0), 0.9)
        self.assertLess(mean_prob(probs, 2.5, 3.5), 0.1)
        self.assertGreater(max_prob(probs, 0.5, 1.5), 0.9)
        self.assertEqual(mean_prob(probs, 100, 101), 0.0)


class TestSileroModel(unittest.TestCase):
    """Runs only when the Silero ONNX model is available."""

    def setUp(self):
        from wordalign.audio.vad import find_vad_model
        from wordalign.models.paths import _project_models_root
        self.model = (os.environ.get("WORDALIGN_TEST_VAD_MODEL")
                      or find_vad_model(str(_project_models_root())))
        if not self.model:
            self.skipTest("Silero VAD model not installed")
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime not installed")

    def test_speech_and_silence(self):
        from wordalign.audio.io import load_audio
        from wordalign.audio.vad import SileroVAD
        clip = Path(__file__).resolve().parent / "golden" / "audio" / "espeak_en.flac"
        try:
            audio = load_audio(str(clip))
        except Exception as exc:
            self.skipTest(f"ffmpeg unavailable: {exc}")
        padded = np.concatenate([np.zeros(3 * SR, np.float32), audio])
        probs = SileroVAD(self.model).probabilities(padded)
        self.assertLess(float(np.mean(probs[:int(2.5 / HOP_S)])), 0.2)
        self.assertGreater(float(np.mean(probs[int(3.5 / HOP_S):] > 0.5)), 0.3)


class TestQualityMap(unittest.TestCase):

    def _map(self):
        q = QualityMap(duration=60.0, vad_probs=np.full(int(60 / HOP_S), 0.8, np.float32))
        q.speech = [(0.0, 60.0)]
        q.chunks = [ChunkInfo(0.0, 20.0, snr_db=25.0),
                    ChunkInfo(20.0, 40.0, snr_db=4.0, music=0.6),
                    ChunkInfo(40.0, 60.0, snr_db=1.0, narrowband=True)]
        q.overlaps = [(45.0, 52.0)]
        return q

    def test_conditions(self):
        q = self._map()
        self.assertEqual(q.chunks[0].conditions, [])
        self.assertEqual(q.chunks[1].conditions, ["music", "low_snr"])
        self.assertIn("very_low_snr", q.chunks[2].conditions)
        self.assertIn("overlap", q.conditions_at(44.0, 50.0))
        self.assertEqual(q.conditions_at(1.0, 5.0), [])

    def test_cache_round_trip(self):
        q = self._map()
        q.sed_groups = {"music": np.linspace(0, 1, 1500).astype(np.float16)}
        back = QualityMap.from_cache(q.to_cache())
        self.assertEqual(len(back.chunks), 3)
        self.assertEqual(back.chunks[1].conditions, q.chunks[1].conditions)
        self.assertAlmostEqual(back.group_mean("music", 0, 60), 0.5, delta=0.01)
        self.assertTrue(back.has_vad)
        self.assertEqual(back.summary()["chunks"], 3)

    def test_fixed_chunks_cover_the_file(self):
        chunks = fixed_chunks(100.0, 28.0)
        self.assertEqual(chunks[0][0], 0.0)
        self.assertEqual(chunks[-1][1], 100.0)
        self.assertTrue(all(e - s <= 28.0 for s, e in chunks))


class TestCleanupPlan(unittest.TestCase):

    def test_gates(self):
        q = QualityMap(duration=100.0)
        q.chunks = [ChunkInfo(0, 10, snr_db=20.0),                    # clean
                    ChunkInfo(10, 20, snr_db=4.0, music=0.7),         # music, low SNR
                    ChunkInfo(20, 30, snr_db=15.0, music=0.7),        # music, high SNR
                    ChunkInfo(30, 40, snr_db=1.0, crowd=0.6),         # loud crowd
                    ChunkInfo(40, 50, snr_db=6.0),                    # steady noise
                    ChunkInfo(50, 60, snr_db=2.0, narrowband=True),   # phone: never
                    ChunkInfo(60, 70, snr_db=6.0, crowd=0.6)]         # moderate crowd
        plan = dict(plan_cleanup(q, "auto"))
        self.assertEqual(plan, {1: "vocals", 3: "crowd", 4: "denoise"})
        self.assertEqual(plan_cleanup(q, "off"), [])
        everything = dict(plan_cleanup(q, "all"))
        self.assertEqual(everything[2], "vocals")
        self.assertEqual(everything[6], "crowd")
        self.assertNotIn(5, everything)

    def test_crossfade_keeps_the_rest_untouched(self):
        out = np.ones(SR, dtype=np.float32)
        piece = np.zeros(SR // 2, dtype=np.float32)
        _crossfade_insert(out, piece, SR // 4)
        self.assertTrue(np.all(out[:SR // 4] == 1.0))
        self.assertTrue(np.all(out[3 * SR // 4:] == 1.0))
        self.assertEqual(float(out[SR // 2]), 0.0)
        self.assertTrue(np.all(np.diff(out[SR // 4:SR // 4 + 800]) <= 0))


class TestWavIO(unittest.TestCase):

    def test_round_trip(self):
        x = (0.5 * np.sin(np.linspace(0, 100, SR))).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.wav")
            write_wav(path, x)
            y = read_wav(path)
        self.assertEqual(len(y), len(x))
        self.assertLess(float(np.max(np.abs(x - y))), 1e-3)


if __name__ == "__main__":
    unittest.main()
