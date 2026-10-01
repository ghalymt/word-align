"""Stage cache: failed or cancelled engine runs are never replayed."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.config import PipelineConfig
from wordalign.core.cache import StageCache
from wordalign.core.pipeline import PipelineRunner


class TestStageCacheEngineRuns(unittest.TestCase):
    """PipelineRunner._cached_engine must not replay failed/cancelled runs."""

    ARGS = ("whisperx", "large-v3", "en", {"device": "cpu"})
    WORDS = [{"word": "hello", "start": 0.0, "end": 0.4, "conf": 0.9}]

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache_dir = Path(tmp.name) / "cache"
        self.audio = Path(tmp.name) / "audio.wav"
        self.audio.write_bytes(b"RIFF-not-really-audio")
        patcher = patch.dict(os.environ, {"WORDALIGN_CACHE": str(self.cache_dir)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.runner = PipelineRunner(PipelineConfig(str(self.audio)))
        self.calls = 0

    def _run(self, result, cancel=False):
        def loader():
            self.calls += 1
            if cancel:
                self.runner.ctx.cancel_event.set()
            return result
        return self.runner._cached_engine(*self.ARGS, loader)

    def _cache_files(self):
        return list(self.cache_dir.glob("*/*.json"))

    def test_empty_whisperx_result_is_not_cached(self):
        self._run(([], []))
        self.assertEqual(self._cache_files(), [])
        output, hit = self._run((self.WORDS, []))
        self.assertFalse(hit)
        self.assertEqual(self.calls, 2)
        self.assertEqual(output[0], self.WORDS)

    def test_empty_word_list_is_not_cached(self):
        self._run([])
        self.assertEqual(self._cache_files(), [])
        self._run([])
        self.assertEqual(self.calls, 2)

    def test_cancelled_partial_result_is_not_cached(self):
        self._run(self.WORDS, cancel=True)
        self.assertEqual(self._cache_files(), [])
        self.runner.ctx.cancel_event.clear()
        output, hit = self._run(self.WORDS)
        self.assertFalse(hit)
        self.assertEqual(self.calls, 2)

    def test_cancel_requested_flag_blocks_caching(self):
        self.runner.ctx.cancel_requested = True
        self._run(self.WORDS)
        self.assertEqual(self._cache_files(), [])

    def test_existing_empty_entry_is_a_cache_miss(self):
        # Entries written by earlier versions: the engine failed, `([], [])`
        # was stored, and every later run read it back for 30 days.
        from wordalign.core.fingerprint import engine_cache_key, fingerprint_audio
        plugin, model, language, settings = self.ARGS
        key = engine_cache_key(fingerprint_audio(str(self.audio)), plugin,
                               "2.0", model, None, language, settings)
        StageCache().put(key, [[], []])
        output, hit = self._run((self.WORDS, []))
        self.assertFalse(hit)
        self.assertEqual(self.calls, 1)
        self.assertEqual(output[0], self.WORDS)

    def test_successful_result_is_still_cached(self):
        self._run((self.WORDS, []))
        output, hit = self._run(([], []))
        self.assertTrue(hit)
        self.assertEqual(self.calls, 1)
        self.assertEqual(output[0], self.WORDS)



if __name__ == "__main__":
    unittest.main()
