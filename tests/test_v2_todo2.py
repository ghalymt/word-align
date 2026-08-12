"""Tests for the second TODO-completion pass (benchmark, GPU scheduler,
crash recovery, code-switching, diarization, exe build, native GUI)."""
import sys
import os
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.core.types import WordResult
from wordalign.core.database import ProjectStore, ProjectRecord, JobRecord


def W(text, t, conf=0.9, dur=0.3, source="Vosk"):
    return WordResult(
        id=f"w{int(t*100):06d}", text=text,
        normalized_text=text.lower().strip(".,!?"),
        start=t, end=t+dur, confidence=conf,
        timing_source=source, matched=True,
    )


# ========== Benchmark ==========

def test_benchmark_compute_wer_perfect():
    from wordalign.benchmark import compute_wer
    info = compute_wer(["hello", "world"], ["hello", "world"])
    assert info["wer"] == 0.0

def test_benchmark_compute_wer_bad():
    from wordalign.benchmark import compute_wer
    info = compute_wer(["hello", "world", "test"], ["goodbye", "world"])
    assert info["wer"] > 0.0

def test_benchmark_engine():
    from wordalign.benchmark import benchmark_engine
    def fake_transcribe(audio_path, **kwargs):
        return [W("hello", 0.0), W("world", 0.3)]
    result = benchmark_engine(
        "fake", fake_transcribe,
        ["hello", "world"], "fake.wav", model_id="fake-v1")
    assert result.engine_id == "fake"
    assert result.wer == 0.0
    assert result.coverage == 1.0
    assert result.runtime_seconds >= 0.0

def test_benchmark_confidence_stats():
    from wordalign.benchmark import _confidence_stats
    words = [W("a", 0.0, conf=0.95), W("b", 0.3, conf=0.5)]
    stats = _confidence_stats(words)
    assert stats["mean"] == 0.725
    assert stats["histogram"]["0.8-1.0"] == 1
    assert stats["histogram"]["0.4-0.6"] == 1

def test_benchmark_create_profile():
    from wordalign.benchmark import BenchmarkResult, create_profile_from_benchmark
    results = [BenchmarkResult(engine_id="vosk", wer=0.1, coverage=0.9),
               BenchmarkResult(engine_id="whisperx", wer=0.05, coverage=0.95)]
    profile = create_profile_from_benchmark(results)
    assert profile["name"] == "Benchmark whisperx"
    assert profile["transcription"]["engines"][0]["plugin"] == "whisperx"

def test_benchmark_format_report():
    from wordalign.benchmark import BenchmarkResult, format_benchmark_report
    results = [BenchmarkResult(engine_id="vosk", wer=0.1, runtime_seconds=5.0,
                               peak_memory_mb=100.0, coverage=0.9, confidence_mean=0.8)]
    report = format_benchmark_report(results)
    assert "WORD-ALIGN BENCHMARK" in report
    assert "vosk" in report


# ========== GPU Scheduler ==========

def test_gpu_scheduler_acquire_release():
    from wordalign.core.gpu_scheduler import GPUScheduler
    sched = GPUScheduler(max_concurrent=1)
    slot = sched.acquire("whisperx", model_id="large-v3", estimated_mb=4096)
    assert slot.engine_id == "whisperx"
    assert sched.active_count == 1
    sched.release("whisperx")
    assert sched.active_count == 0

def test_gpu_scheduler_blocks_second():
    from wordalign.core.gpu_scheduler import GPUScheduler
    sched = GPUScheduler(max_concurrent=1)
    sched.acquire("whisperx")
    try:
        sched.acquire("qwen", timeout_seconds=0.1)
        assert False, "should have timed out"
    except TimeoutError:
        pass
    sched.release("whisperx")

def test_gpu_scheduler_context_manager():
    from wordalign.core.gpu_scheduler import GPUScheduler
    sched = GPUScheduler()
    with sched:
        sched.acquire("vosk")
        assert sched.active_count == 1
    assert sched.active_count == 0

def test_detect_oom():
    from wordalign.core.gpu_scheduler import detect_oom
    assert detect_oom(RuntimeError("CUDA out of memory"))
    assert detect_oom(RuntimeError("torch.cuda.OutOfMemoryError"))
    assert not detect_oom(RuntimeError("file not found"))

def test_suggest_recovery():
    from wordalign.core.gpu_scheduler import suggest_recovery
    rec = suggest_recovery("qwen", RuntimeError("CUDA out of memory"),
                           chunk_seconds=60.0)
    assert rec.suggest_smaller_chunks is True
    assert rec.suggest_cpu_fallback is True
    assert "qwen" in rec.message


# ========== Crash Recovery ==========

def test_recovery_no_incomplete():
    from wordalign.core.recovery import CrashRecovery
    db = tempfile.mktemp(suffix=".db")
    store = ProjectStore(db_path=db)
    store.create_project(ProjectRecord(id="p1", name="P", audio_path="/a.wav"))
    store.create_job(JobRecord(id="j1", project_id="p1", status="completed",
                               started_at=time.time()-1000))
    rec = CrashRecovery(store)
    report = rec.find_incomplete_jobs()
    assert report.count == 0
    store.close()
    os.unlink(db)

def test_recovery_detects_stale():
    from wordalign.core.recovery import CrashRecovery
    db = tempfile.mktemp(suffix=".db")
    store = ProjectStore(db_path=db)
    store.create_project(ProjectRecord(id="p1", name="P", audio_path="/a.wav"))
    store.create_job(JobRecord(id="j1", project_id="p1", status="running",
                               started_at=time.time()-3600))
    store.record_stage("j1", "vosk", "completed", started_at=time.time()-3600,
                       completed_at=time.time()-3500, duration_seconds=100.0)
    rec = CrashRecovery(store)
    report = rec.find_incomplete_jobs()
    assert report.count == 1
    assert report.incomplete[0].last_completed_stage == "vosk"
    store.close()
    os.unlink(db)

def test_recovery_skips_fresh_job():
    from wordalign.core.recovery import CrashRecovery
    db = tempfile.mktemp(suffix=".db")
    store = ProjectStore(db_path=db)
    store.create_project(ProjectRecord(id="p1", name="P", audio_path="/a.wav"))
    store.create_job(JobRecord(id="j1", project_id="p1", status="running",
                               started_at=time.time()))
    rec = CrashRecovery(store)
    report = rec.find_incomplete_jobs()
    assert report.count == 0  # too fresh
    store.close()
    os.unlink(db)

def test_recovery_resume_plan():
    from wordalign.core.recovery import CrashRecovery
    db = tempfile.mktemp(suffix=".db")
    store = ProjectStore(db_path=db)
    store.create_project(ProjectRecord(id="p1", name="P", audio_path="/a.wav"))
    store.create_job(JobRecord(id="j1", project_id="p1", status="running",
                               started_at=time.time()-3600))
    store.record_stage("j1", "vosk", "completed", started_at=time.time()-3600,
                       completed_at=time.time()-3500, duration_seconds=100.0)
    rec = CrashRecovery(store)
    plan = rec.resume_plan("j1")
    assert plan["resume_stage"] == "vosk"
    store.close()
    os.unlink(db)


# ========== Code-Switching ==========

def test_detect_script_arabic():
    from wordalign.qa.codeswitch import detect_script
    assert detect_script("مرحبا") == "ar"
    assert detect_script("hello") is None

def test_detect_script_cjk():
    from wordalign.qa.codeswitch import detect_script
    assert detect_script("你好") == "zh"
    assert detect_script("こんにちは") == "ja"
    assert detect_script("안녕") == "ko"

def test_detect_word_language():
    from wordalign.qa.codeswitch import detect_word_language
    assert detect_word_language("the") == "en"
    assert detect_word_language("مرحبا") == "ar"
    assert detect_word_language("bonjour", fallback="fr") == "fr"  # not in stopwords

def test_detect_language_span():
    from wordalign.qa.codeswitch import detect_language_span
    words = [W("hello", i * 0.3) for i in range(10)]
    words += [W("مرحبا", 3.0 + i * 0.3) for i in range(10)]
    result = detect_language_span(words)
    langs = {w.language for w in result}
    assert "ar" in langs

def test_language_stats():
    from wordalign.qa.codeswitch import language_stats
    words = [W("the", 0.0), W("مرحبا", 0.5)]
    words[0].language = "en"
    words[1].language = "ar"
    stats = language_stats(words)
    assert stats.get("en") == 1
    assert stats.get("ar") == 1


# ========== Diarization ==========

def test_diarization_adapter_descriptor():
    from wordalign.engines.adapters.diarization_adapter import DiarizationAdapter
    from wordalign.plugins.capabilities import Capability
    adapter = DiarizationAdapter()
    desc = adapter.descriptor()
    assert desc.engine_id == "diarize"
    assert Capability.DIARIZE in desc.capabilities

def test_diarization_gap_fallback():
    from wordalign.engines.adapters.diarization_adapter import DiarizationAdapter
    words = [W("hello", 0.0, dur=0.3), W("world", 0.4, dur=0.3),
             W("next", 3.0, dur=0.3), W("segment", 3.4, dur=0.3)]
    adapter = DiarizationAdapter()
    result = adapter._diarize_by_gaps(words, gap_threshold=0.8)
    assert result[0].speaker == "SPEAKER_0"
    assert result[2].speaker == "SPEAKER_1"

def test_format_speaker_transcript():
    from wordalign.engines.adapters.diarization_adapter import format_speaker_transcript
    words = [W("hello", 0.0), W("world", 0.3)]
    words[0].speaker = "SPEAKER_0"
    words[1].speaker = "SPEAKER_0"
    out = format_speaker_transcript(words)
    assert "SPEAKER_0" in out
    assert "hello world" in out


# ========== Build files ==========

def test_build_spec_exists():
    assert Path("build.spec").exists() or (Path(__file__).parent.parent / "build.spec").exists()

def test_build_py_exists():
    assert Path("build.py").exists() or (Path(__file__).parent.parent / "build.py").exists()

def test_build_script_exists():
    p = Path(__file__).parent.parent / "scripts" / "build_windows.ps1"
    assert p.exists()

def test_native_gui_imports():
    import wordalign.gui.native as native
    assert callable(native.main)

def test_native_workers_format():
    from wordalign.gui.native.workers import _format_event
    from wordalign.core.events import StageCompleted, PipelineStarted
    assert _format_event(PipelineStarted(mode="reference", language="en")) == "[start] mode=reference language=en"
    assert _format_event(StageCompleted(stage="vosk", duration_seconds=1.5)) == "[vosk] done (1.5s)"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except AssertionError as exc:
                failures += 1
                print(f"[FAIL] {name}: {exc}")
            except Exception as exc:
                failures += 1
                print(f"[ERROR] {name}: {type(exc).__name__}: {exc}")
    print("\n" + ("ALL TESTS PASSED" if not failures
                  else f"{failures} TEST(S) FAILED"))
    sys.exit(1 if failures else 0)
