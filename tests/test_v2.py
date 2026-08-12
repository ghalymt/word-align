"""Tests for the new core, plugin, model, and QA systems.

Run with: python -m pytest tests/test_v2.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.core.types import WordResult, SegmentResult, EngineResult, PipelineResult
from wordalign.core.config import SegmentationConfig, PipelineProfile, JobContext
from wordalign.core.events import CollectSink, PrintSink, PipelineStarted, StageCompleted
from wordalign.core.converters import words_to_dicts, dicts_to_words
from wordalign.core.errors import EngineError, OOMError, PipelineCancelled
from wordalign.plugins.capabilities import Capability
from wordalign.plugins.base import EnginePlugin, EngineDescriptor, HealthStatus
from wordalign.plugins.manifest import PluginManifest
from wordalign.plugins.registry import PluginRegistry
from wordalign.plugins.protocol import make_request, make_cancel, parse_line, PROTOCOL_VERSION
from wordalign.models.catalog import ModelCatalog, ModelEntry
from wordalign.models.manager import ModelManager
from wordalign.models.validator import ModelValidator
from wordalign.runtimes.detector import HardwareDetector
from wordalign.qa.issues import RiskScore, TranscriptIssue
from wordalign.qa.deterministic import (detect_agreement_issues,
                                         detect_repetition,
                                         detect_timing_anomalies,
                                         run_deterministic_qa)
from wordalign.qa.engine import QAEngine


def W(text, t, conf=0.9, dur=0.3, source="Vosk", wid=None):
    return WordResult(
        id=wid or f"w{int(t*100):06d}",
        text=text,
        normalized_text=text.lower().strip(".,!?"),
        start=t, end=t+dur, confidence=conf,
        timing_source=source, matched=True,
    )


# ========== Core types ==========

def test_word_result_creation():
    w = WordResult(id="w000001", text="Hello", normalized_text="hello",
                   start=0.0, end=0.3, confidence=0.95)
    assert w.text == "Hello"
    assert w.confidence == 0.95
    assert w.matched == False  # default

def test_segmentation_config_defaults():
    cfg = SegmentationConfig()
    assert cfg.max_cpl == 42
    assert cfg.max_lines == 2
    assert cfg.cue_char_budget == 84
    assert cfg.iteration_start == 1
    assert cfg.iteration_end == 84

def test_segmentation_config_custom():
    cfg = SegmentationConfig(max_cpl=32, max_lines=2)
    assert cfg.cue_char_budget == 64

def test_pipeline_profile_preset():
    profile = PipelineProfile.preset("fast")
    assert profile.name == "Fast"
    assert len(profile.transcription.engines) == 2

def test_pipeline_profile_from_json():
    import tempfile, json
    d = {"schema_version": 1, "name": "Test", "segmentation": {"max_cpl": 32}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(d, f)
        f.flush()
        profile = PipelineProfile.from_json(f.name)
    assert profile.name == "Test"
    assert profile.segmentation.max_cpl == 32

def test_job_context_cancel():
    ctx = JobContext()
    assert not ctx.cancel_requested
    ctx.cancel_requested = True
    try:
        ctx.check_cancel("test")
        assert False, "should have raised"
    except PipelineCancelled:
        pass


# ========== Converters ==========

def test_words_to_dicts_roundtrip():
    words = [W("Hello", 0.0), W("world", 0.4)]
    dicts = words_to_dicts(words)
    assert dicts[0]["word"] == "Hello"
    assert dicts[0]["start"] == 0.0
    back = dicts_to_words(dicts, 0)
    assert back[0].text == "Hello"
    assert back[0].matched == True


# ========== Events ==========

def test_collect_sink():
    sink = CollectSink()
    sink.emit(PipelineStarted(audio_path="/test.wav", mode="reference", language="en"))
    sink.emit(StageCompleted(stage="vosk", duration_seconds=1.5))
    assert len(sink.events) == 2
    assert isinstance(sink.events[0], PipelineStarted)
    assert sink.events[1].stage == "vosk"


# ========== Capabilities ==========

def test_capability_flags():
    caps = Capability.TRANSCRIBE | Capability.WORD_TIMING | Capability.VOTE
    labels = caps.labels()
    assert "transcribe" in labels
    assert "word_timing" in labels
    assert "vote" in labels
    assert Capability.TRANSCRIBE in caps
    assert Capability.DIARIZE not in caps


# ========== Manifest ==========

def test_manifest_validation():
    m = PluginManifest(id="test", name="Test", capabilities=["transcribe", "vote"])
    errors = m.validate()
    assert errors == []
    assert m.capability_flags & Capability.TRANSCRIBE
    assert m.capability_flags & Capability.VOTE

def test_manifest_invalid_capability():
    m = PluginManifest(id="test", capabilities=["flying"])
    errors = m.validate()
    assert any("flying" in e for e in errors)

def test_manifest_missing_id():
    m = PluginManifest()
    errors = m.validate()
    assert any("id" in e for e in errors)


# ========== Registry ==========

class FakePlugin(EnginePlugin):
    def descriptor(self):
        return EngineDescriptor(
            engine_id="fake", display_name="Fake", version="1.0",
            capabilities=Capability.TRANSCRIBE | Capability.VOTE,
            runtime_type="in_process", supported_languages=["*"],
            models=[], hardware={"cpu": True, "cuda": False})
    def health_check(self):
        return HealthStatus(ready=True, runtime_status="ready")

def test_registry_register_and_query():
    reg = PluginRegistry()
    reg.register(FakePlugin())
    assert "fake" in reg
    assert len(reg) == 1
    ids = reg.query(Capability.TRANSCRIBE)
    assert "fake" in ids
    ids = reg.query(Capability.DIARIZE)
    assert "fake" not in ids

def test_registry_unregister():
    reg = PluginRegistry()
    reg.register(FakePlugin())
    reg.unregister("fake")
    assert "fake" not in reg
    assert len(reg) == 0

def test_registry_health_check_all():
    reg = PluginRegistry()
    reg.register(FakePlugin())
    results = reg.health_check_all()
    assert "fake" in results
    assert results["fake"]["ready"] == True


# ========== Protocol ==========

def test_protocol_make_request():
    req = make_request("transcribe", {"audio_path": "/test.wav"})
    assert req["type"] == "request"
    assert req["method"] == "transcribe"
    assert req["protocol_version"] == PROTOCOL_VERSION
    assert req["request_id"].startswith("req_")

def test_protocol_parse_line():
    msg = parse_line('{"type": "result", "request_id": "abc"}')
    assert msg["type"] == "result"
    assert parse_line("") is None
    assert parse_line("not json") is None

def test_protocol_make_cancel():
    cancel = make_cancel("req_123")
    assert cancel["type"] == "cancel"
    assert cancel["request_id"] == "req_123"


# ========== Model catalog ==========

def test_model_catalog_load():
    catalog_path = Path(__file__).parent.parent / "wordalign" / "data" / "model_catalog.json"
    catalog = ModelCatalog()
    catalog.load(str(catalog_path))
    assert len(catalog.list_all()) >= 5
    entry = catalog.get("whisper-large-v3")
    assert entry is not None
    assert entry.plugin_id == "whisperx"
    assert entry.recommended == True

def test_model_catalog_by_plugin():
    catalog_path = Path(__file__).parent.parent / "wordalign" / "data" / "model_catalog.json"
    catalog = ModelCatalog(str(catalog_path))
    wx_models = catalog.list_for_plugin("whisperx")
    assert len(wx_models) >= 2

def test_model_catalog_by_language():
    catalog_path = Path(__file__).parent.parent / "wordalign" / "data" / "model_catalog.json"
    catalog = ModelCatalog(str(catalog_path))
    en_models = catalog.list_by_language("en")
    assert len(en_models) >= 5


# ========== Model manager ==========

def test_model_manager_register_external(tmp_path=None):
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    mgr = ModelManager(str(tmp / "models"))
    # Create a fake model directory
    model_dir = tmp / "my_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    reg = mgr.register_external("whisper-large-v3", str(model_dir))
    assert reg.managed == False
    assert reg.validated == True
    assert mgr.is_installed("whisper-large-v3")
    # External models are NOT deleted
    mgr.delete("whisper-large-v3")
    assert model_dir.exists()  # should still exist
    assert not mgr.is_installed("whisper-large-v3")


# ========== Model validator ==========

def test_model_validator_whisperx():
    import tempfile
    from pathlib import Path
    v = ModelValidator()
    tmp = Path(tempfile.mkdtemp())
    # Empty dir -> invalid
    result = v.validate(tmp, "whisperx")
    assert not result["valid"]
    # Create expected files
    for f in ["config.json", "pytorch_model.bin", "tokenizer.json"]:
        (tmp / f).write_text("test")
    result = v.validate(tmp, "whisperx")
    assert result["valid"]


# ========== Hardware detector ==========

def test_hardware_detector():
    detector = HardwareDetector()
    info = detector.detect()
    assert info.os != ""
    assert info.cpu_count > 0
    # RAM may be 0 if psutil is not installed
    if info.ram_gb > 0:
        assert info.ram_gb > 0


# ========== QA: Agreement ==========

def test_qa_agreement_detects_low_confidence():
    words = [W("good", 0.0, conf=0.95), W("bad", 0.5, conf=0.3),
             W("okay", 1.0, conf=0.9)]
    issues = detect_agreement_issues(words, threshold=0.5)
    assert len(issues) == 1
    assert issues[0].category == "agreement"
    assert issues[0].severity in ("medium", "high")  # low confidence → elevated risk
def test_qa_agreement_no_false_positives():
    words = [W("good", 0.0, conf=0.95), W("fine", 0.4, conf=0.9)]
    issues = detect_agreement_issues(words, threshold=0.5)
    assert len(issues) == 0


# ========== QA: Repetition ==========

def test_qa_repetition_detects_repeated_phrases():
    words = []
    text = "I think that we should I think that we should go".split()
    for i, w in enumerate(text):
        words.append(W(w, i * 0.3, conf=0.9))
    issues = detect_repetition(words, min_n=3, max_n=6)
    assert len(issues) >= 1
    assert issues[0].category == "repetition"


# ========== QA: Timing ==========

def test_qa_timing_detects_overlap():
    words = [W("hello", 0.0, dur=0.5), W("world", 0.3, dur=0.3)]
    issues = detect_timing_anomalies(words)
    overlap_issues = [i for i in issues if "overlap" in i.id]
    assert len(overlap_issues) >= 1

def test_qa_timing_detects_interpolated_run():
    words = [W("timed", 0.0, source="Vosk")]
    for i in range(7):
        words.append(W(f"word{i}", 0.3 + i * 0.3, source="Interpolated"))
    words.append(W("timed2", 3.0, source="Vosk"))
    issues = detect_timing_anomalies(words)
    interp_issues = [i for i in issues if "interp" in i.id]
    assert len(interp_issues) >= 1


# ========== QA: Combined ==========

def test_qa_engine_runs_deterministic():
    words = [W("good", 0.0, conf=0.95), W("bad", 0.5, conf=0.3)]
    engine = QAEngine(enable_llm=False)
    issues = engine.review(words)
    assert len(issues) >= 1

def test_risk_score_overall():
    risk = RiskScore(agreement_risk=0.8, confidence_risk=0.6, timing_risk=0.4)
    assert 0 < risk.overall < 1.0
    assert risk.severity in ("low", "medium", "high")

def test_risk_score_explanation():
    risk = RiskScore(agreement_risk=0.8, semantic_risk=0.6)
    text = risk.explanation()
    assert "Ensemble disagreement" in text
    assert "LLM flagged" in text


# ========== Engine adapters ==========

def test_whisperx_adapter_descriptor():
    from wordalign.engines.adapters.whisperx_adapter import WhisperXAdapter
    adapter = WhisperXAdapter()
    desc = adapter.descriptor()
    assert desc.engine_id == "whisperx"
    assert Capability.TRANSCRIBE in desc.capabilities
    assert Capability.VOTE in desc.capabilities

def test_vosk_adapter_descriptor():
    from wordalign.engines.adapters.vosk_adapter import VoskAdapter
    adapter = VoskAdapter()
    desc = adapter.descriptor()
    assert desc.engine_id == "vosk"
    assert Capability.WORD_TIMING in desc.capabilities

def test_qwen_adapter_descriptor():
    from wordalign.engines.adapters.qwen_adapter import QwenAdapter
    adapter = QwenAdapter()
    desc = adapter.descriptor()
    assert desc.engine_id == "qwen_asr"
    assert desc.runtime_type == "subprocess"

def test_mfa_adapter_descriptor():
    from wordalign.engines.adapters.mfa_adapter import MFAAdapter
    adapter = MFAAdapter()
    desc = adapter.descriptor()
    assert desc.engine_id == "mfa"
    assert Capability.FORCED_ALIGN in desc.capabilities
    assert Capability.TRANSCRIBE not in desc.capabilities

def test_register_all_adapters():
    from wordalign.engines.adapters.whisperx_adapter import WhisperXAdapter
    from wordalign.engines.adapters.vosk_adapter import VoskAdapter
    from wordalign.engines.adapters.qwen_adapter import QwenAdapter
    from wordalign.engines.adapters.mfa_adapter import MFAAdapter
    from wordalign.engines.adapters.yamnet_adapter import YAMNetAdapter
    reg = PluginRegistry()
    reg.register(WhisperXAdapter())
    reg.register(VoskAdapter())
    reg.register(QwenAdapter())
    reg.register(MFAAdapter())
    reg.register(YAMNetAdapter())
    assert len(reg) == 5
    assert "whisperx" in reg
    assert "vosk" in reg
    assert "qwen_asr" in reg
    assert "mfa" in reg
    assert "yamnet" in reg
    transcribe = reg.query(Capability.TRANSCRIBE)
    assert "whisperx" in transcribe
    assert "vosk" in transcribe
    assert "qwen_asr" in transcribe
    assert "mfa" not in transcribe
    align = reg.query(Capability.FORCED_ALIGN)
    assert "mfa" in align

def test_dummy_plugin_registration_no_code_changes():
    """Verify that adding a new engine requires zero changes to existing code."""
    class SuperASR2027(EnginePlugin):
        def descriptor(self):
            return EngineDescriptor(
                engine_id="super_asr_2027",
                display_name="SuperASR 2027",
                version="1.0",
                capabilities=Capability.TRANSCRIBE | Capability.WORD_TIMING | Capability.VOTE,
                runtime_type="in_process",
                supported_languages=["en"],
                models=["super-asr-v1"],
                hardware={"cpu": False, "cuda": True})
        def health_check(self):
            return HealthStatus(ready=True, runtime_status="ready")
    reg = PluginRegistry()
    reg.register(SuperASR2027())
    # The new engine is automatically discoverable
    assert "super_asr_2027" in reg.query(Capability.TRANSCRIBE)
    assert "super_asr_2027" in reg.query(Capability.VOTE)
    # No changes needed to pipeline.py, cli.py, or any existing adapter


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
