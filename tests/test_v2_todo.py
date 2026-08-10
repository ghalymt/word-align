"""Tests for the new modules added in the TODO completion pass.

Tests: stage cache, SQLite store, GUI backend, thin CLI, segmenter config,
LLM QA providers, local re-alignment, waveform, job manifest, MFA multilingual.
"""
import sys
import os
import json
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.core.cache import StageCache
from wordalign.core.fingerprint import fingerprint_audio, engine_cache_key, AudioFingerprint
from wordalign.core.database import ProjectStore, ProjectRecord, JobRecord
from wordalign.core.config import SegmentationConfig, PipelineProfile
from wordalign.core.manifest import generate_manifest, save_manifest
from wordalign.segment import set_config, get_config, set_layout, _max_cpl, _max_lines
from wordalign.qa.providers import LLMQAProvider, LocalModelProvider, OpenAICompatibleProvider, NoOpProvider
from wordalign.qa.realignment import detect_edits, compute_local_timestamps, apply_realignment, EditRegion
from wordalign.qa.issues import RiskScore, TranscriptIssue
from wordalign.core.types import WordResult


def W(text, t, conf=0.9, dur=0.3, source="Vosk"):
    return WordResult(
        id=f"w{int(t*100):06d}", text=text,
        normalized_text=text.lower().strip(".,!?"),
        start=t, end=t+dur, confidence=conf,
        timing_source=source, matched=True,
    )


# ========== Stage Cache ==========

def test_stage_cache_put_get():
    cache_dir = tempfile.mkdtemp()
    cache = StageCache(cache_dir=cache_dir)
    cache.put("test_key_1", {"words": ["hello", "world"]})
    data = cache.get("test_key_1")
    assert data is not None
    assert data["words"] == ["hello", "world"]

def test_stage_cache_miss():
    cache_dir = tempfile.mkdtemp()
    cache = StageCache(cache_dir=cache_dir)
    assert cache.get("nonexistent") is None

def test_stage_cache_has():
    cache_dir = tempfile.mkdtemp()
    cache = StageCache(cache_dir=cache_dir)
    cache.put("has_key", {"data": 42})
    assert cache.has("has_key")
    assert not cache.has("missing_key")

def test_stage_cache_invalidate():
    cache_dir = tempfile.mkdtemp()
    cache = StageCache(cache_dir=cache_dir)
    cache.put("del_key", {"x": 1})
    cache.invalidate("del_key")
    assert not cache.has("del_key")

def test_stage_cache_clear():
    cache_dir = tempfile.mkdtemp()
    cache = StageCache(cache_dir=cache_dir)
    cache.put("a", {1: 2})
    cache.put("b", {3: 4})
    deleted = cache.clear()
    assert deleted == 2
    assert not cache.has("a")
    assert not cache.has("b")

def test_audio_fingerprint():
    # Create a temp file with known content
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(b"fake audio data for fingerprinting test")
    tmp.close()
    try:
        fp = fingerprint_audio(tmp.name)
        assert isinstance(fp, AudioFingerprint)
        assert fp.file_size > 0
        assert len(fp.cache_key) == 16
    finally:
        os.unlink(tmp.name)

def test_engine_cache_key_deterministic():
    fp = AudioFingerprint(file_hash="abc123", file_size=1000, file_mtime=123456, duration_hint=10.0)
    k1 = engine_cache_key(fp, "vosk", "1.0", "vosk-model-en", None, "en", {"chunk": 60})
    k2 = engine_cache_key(fp, "vosk", "1.0", "vosk-model-en", None, "en", {"chunk": 60})
    assert k1 == k2

def test_engine_cache_key_different_settings():
    fp = AudioFingerprint(file_hash="abc123", file_size=1000, file_mtime=123456, duration_hint=10.0)
    k1 = engine_cache_key(fp, "vosk", "1.0", None, None, "en", {"chunk": 60})
    k2 = engine_cache_key(fp, "vosk", "1.0", None, None, "en", {"chunk": 30})
    assert k1 != k2


# ========== SQLite Project Store ==========

def test_project_store_create_and_get():
    db_path = tempfile.mktemp(suffix=".db")
    store = ProjectStore(db_path=db_path)
    project = ProjectRecord(
        id="test_proj_1", name="Test Project",
        audio_path="/fake/audio.wav",
        transcript_path="/fake/transcript.txt",
        language="en", mode="reference",
    )
    store.create_project(project)
    retrieved = store.get_project("test_proj_1")
    assert retrieved is not None
    assert retrieved.name == "Test Project"
    assert retrieved.audio_path == "/fake/audio.wav"
    store.close()
    os.unlink(db_path)

def test_project_store_list():
    db_path = tempfile.mktemp(suffix=".db")
    store = ProjectStore(db_path=db_path)
    for i in range(3):
        store.create_project(ProjectRecord(
            id=f"proj_{i}", name=f"Project {i}",
            audio_path=f"/fake/audio_{i}.wav",
        ))
    projects = store.list_projects()
    assert len(projects) == 3
    store.close()
    os.unlink(db_path)

def test_project_store_update():
    db_path = tempfile.mktemp(suffix=".db")
    store = ProjectStore(db_path=db_path)
    store.create_project(ProjectRecord(
        id="proj_update", name="Original",
        audio_path="/fake/audio.wav",
    ))
    store.update_project("proj_update", name="Updated", language="fr")
    retrieved = store.get_project("proj_update")
    assert retrieved.name == "Updated"
    assert retrieved.language == "fr"
    store.close()
    os.unlink(db_path)

def test_project_store_delete():
    db_path = tempfile.mktemp(suffix=".db")
    store = ProjectStore(db_path=db_path)
    store.create_project(ProjectRecord(
        id="proj_del", name="To Delete",
        audio_path="/fake/audio.wav",
    ))
    store.delete_project("proj_del")
    assert store.get_project("proj_del") is None
    store.close()
    os.unlink(db_path)

def test_project_store_job():
    db_path = tempfile.mktemp(suffix=".db")
    store = ProjectStore(db_path=db_path)
    store.create_project(ProjectRecord(
        id="proj_job", name="Job Test",
        audio_path="/fake/audio.wav",
    ))
    store.create_job(JobRecord(
        id="job_1", project_id="proj_job", status="running",
        started_at=time.time(),
    ))
    store.update_job("job_1", status="completed",
                     completed_at=time.time(),
                     word_count=100, segment_count=20)
    job = store.get_job("job_1")
    assert job.status == "completed"
    assert job.word_count == 100
    store.close()
    os.unlink(db_path)

def test_project_store_stages():
    db_path = tempfile.mktemp(suffix=".db")
    store = ProjectStore(db_path=db_path)
    store.create_project(ProjectRecord(
        id="proj_stages", name="Stages Test",
        audio_path="/fake/audio.wav",
    ))
    store.create_job(JobRecord(
        id="job_stages", project_id="proj_stages", status="running",
        started_at=time.time(),
    ))
    stage_id = store.record_stage("job_stages", "vosk", "completed",
                                  started_at=time.time()-5,
                                  completed_at=time.time(),
                                  duration_seconds=5.0)
    stages = store.list_stages("job_stages")
    assert len(stages) == 1
    assert stages[0]["stage_name"] == "vosk"
    store.close()
    os.unlink(db_path)

def test_project_store_qa_issues():
    db_path = tempfile.mktemp(suffix=".db")
    store = ProjectStore(db_path=db_path)
    store.create_project(ProjectRecord(
        id="proj_qa", name="QA Test",
        audio_path="/fake/audio.wav",
    ))
    store.create_job(JobRecord(
        id="job_qa", project_id="proj_qa", status="completed",
        started_at=time.time(),
    ))
    store.save_qa_issue("job_qa", {
        "id": "issue_001",
        "category": "agreement",
        "severity": "high",
        "confidence": 0.8,
        "start": 1.0, "end": 2.0,
        "word_ids": ["w000100", "w000200"],
        "original_text": "hello world",
        "explanation": "Low confidence",
    })
    issues = store.list_qa_issues("job_qa")
    assert len(issues) == 1
    assert issues[0]["category"] == "agreement"
    store.close()
    os.unlink(db_path)

def test_project_store_settings():
    db_path = tempfile.mktemp(suffix=".db")
    store = ProjectStore(db_path=db_path)
    store.set_setting("vosk_models_dir", "/path/to/models")
    val = store.get_setting("vosk_models_dir")
    assert val == "/path/to/models"
    store.close()
    os.unlink(db_path)

def test_project_store_glossary():
    db_path = tempfile.mktemp(suffix=".db")
    store = ProjectStore(db_path=db_path)
    store.add_glossary_entry("GitHub", correct_form="GitHub", context="brand name")
    entries = store.list_glossary()
    assert len(entries) == 1
    assert entries[0]["term"] == "github"
    store.close()
    os.unlink(db_path)


# ========== Segmenter Config ==========

def test_set_config_from_segmentation_config():
    cfg = SegmentationConfig(max_cpl=32, max_lines=2, max_duration_ms=5000, min_cue_ms=600)
    set_config(cfg)
    current = get_config()
    assert current["max_cpl"] == 32
    assert current["max_lines"] == 2
    assert current["max_duration_ms"] == 5000
    assert current["min_cue_ms"] == 600
    # Restore defaults
    set_layout(max_cpl=42, max_lines=2, max_duration_ms=7000)

def test_get_config_returns_dict():
    set_layout(max_cpl=42, max_lines=2, max_duration_ms=7000)
    cfg = get_config()
    assert isinstance(cfg, dict)
    assert cfg["max_cpl"] == 42
    assert "min_cue_ms" in cfg


# ========== LLM QA Providers ==========

def test_noop_provider():
    provider = NoOpProvider()
    result = provider.review_chunk([], [], [])
    assert result == []

def test_local_model_provider_init():
    provider = LocalModelProvider(model_name="Qwen/Qwen2.5-1.5B")
    assert provider.model_name == "Qwen/Qwen2.5-1.5B"
    assert provider._model is None  # lazy loading

def test_openai_compatible_provider_init():
    provider = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    assert provider.base_url == "http://localhost:1234/v1"
    assert provider.model == "local-model"


# ========== Local Re-alignment ==========

def test_detect_edits_no_changes():
    original = [W("hello", 0.0), W("world", 0.3)]
    edited = [W("hello", 0.0), W("world", 0.3)]
    regions = detect_edits(original, edited)
    assert len(regions) == 0

def test_detect_edits_with_change():
    original = [W("hello", 0.0), W("wrld", 0.3), W("test", 0.6)]
    edited = [W("hello", 0.0), W("world", 0.3), W("test", 0.6)]
    regions = detect_edits(original, edited)
    assert len(regions) == 1
    assert regions[0].old_text == ["wrld"]
    assert regions[0].new_text == ["world"]

def test_compute_local_timestamps():
    region = EditRegion(
        start_idx=0, end_idx=2,
        anchor_before=0.0, anchor_after=1.0,
        old_text=["a", "b"], new_text=["c", "d"],
    )
    timestamps = compute_local_timestamps(region)
    assert len(timestamps) == 2
    assert timestamps[0][0] == 0.0
    assert timestamps[1][1] == 1.0

def test_apply_realignment():
    words = [W("hello", 0.0), W("wrld", 0.3), W("test", 0.6)]
    edited = [W("hello", 0.0), W("world", 0.3), W("test", 0.6)]
    regions = detect_edits(words, edited)
    result = apply_realignment(words, regions)
    assert result[1].text == "world"
    assert result[1].timing_source == "realignment"
    assert result[1].confidence == 0.5


# ========== Job Manifest ==========

def test_generate_manifest():
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(b"fake audio data for manifest test")
    tmp.close()
    try:
        manifest = generate_manifest(
            tmp.name,
            profile_name="fast",
            engines_used=["vosk", "whisperx"],
            models_used=["vosk-model-en-us-0.22", "large-v3"],
            stage_durations={"vosk": 5.2, "whisperx": 12.8},
            warnings=["Vosk model not found, using fallback"],
        )
        assert manifest["manifest_version"] == 1
        assert manifest["profile"] == "fast"
        assert "vosk" in manifest["engines"]
        assert manifest["stages"]["vosk"] == 5.2
        assert len(manifest["warnings"]) == 1
        assert manifest["input"]["audio_size_bytes"] > 0
    finally:
        os.unlink(tmp.name)

def test_save_manifest():
    tmp = tempfile.mktemp(suffix=".json")
    manifest = {"test": True, "version": "2.0"}
    path = save_manifest(manifest, tmp)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["test"] is True
    os.unlink(tmp)


# ========== MFA Multilingual ==========

def test_mfa_language_map():
    from wordalign.engines.mfa_engine import get_mfa_models, MFA_LANGUAGE_MAP
    en_models = get_mfa_models("en")
    assert en_models["acoustic"] == "english_us_arpa"
    fr_models = get_mfa_models("fr")
    assert fr_models["acoustic"] == "french_mfa"
    # Unknown language defaults to English
    unknown = get_mfa_models("xx")
    assert unknown["acoustic"] == "english_us_arpa"

def test_mfa_supported_languages():
    from wordalign.engines.mfa_engine import MFA_LANGUAGE_MAP
    assert "en" in MFA_LANGUAGE_MAP
    assert "ar" in MFA_LANGUAGE_MAP
    assert "zh" in MFA_LANGUAGE_MAP
    assert "fr" in MFA_LANGUAGE_MAP
    assert "de" in MFA_LANGUAGE_MAP
    assert "es" in MFA_LANGUAGE_MAP


# ========== Thin CLI ==========

def test_cli_v2_imports():
    from wordalign.cli_v2 import _parse_args, main
    assert callable(_parse_args)
    assert callable(main)

def test_cli_v2_parse_args():
    from wordalign.cli_v2 import _parse_args
    cfg = _parse_args(["test.wav", "-t", "transcript.txt", "--max-cpl", "32"])
    assert cfg.audio_path == "test.wav"
    assert cfg.transcript_path == "transcript.txt"
    assert cfg.max_cpl == 32


# ========== GUI Backend ==========

def test_gui_server_imports():
    from wordalign.gui.server import PipelineAPIHandler, run_server
    assert callable(run_server)

def test_waveform_imports():
    from wordalign.gui.waveform import generate_peaks, save_peaks
    assert callable(generate_peaks)
    assert callable(save_peaks)


# ========== Pipeline Profile Integration ==========

def test_pipeline_profile_with_segmentation_config():
    profile = PipelineProfile.preset("cpu_only")
    cfg = profile.segmentation
    assert cfg.max_cpl == 42
    assert cfg.max_lines == 2
    # set_config should accept it
    set_config(cfg)
    current = get_config()
    assert current["max_cpl"] == 42
    set_layout(max_cpl=42, max_lines=2, max_duration_ms=7000)


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
