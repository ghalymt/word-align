import sys; sys.path.insert(0, '.')

from wordalign.core.types import WordResult, SegmentResult, EngineResult, PipelineResult
from wordalign.core.config import SegmentationConfig, PipelineProfile, JobContext
from wordalign.core.events import EventSink, PrintSink, CollectSink, PipelineStarted, PipelineCompleted
from wordalign.core.errors import EngineError, OOMError, PipelineCancelled
from wordalign.core.converters import words_to_dicts, dicts_to_words
from wordalign.core.pipeline import PipelineRunner
from wordalign.plugins.capabilities import Capability
from wordalign.plugins.base import EnginePlugin, EngineDescriptor, HealthStatus
from wordalign.plugins.manifest import PluginManifest
from wordalign.plugins.registry import PluginRegistry
from wordalign.plugins.protocol import SubprocessRunner, make_request, parse_line
from wordalign.engines.adapters.whisperx_adapter import WhisperXAdapter
from wordalign.engines.adapters.vosk_adapter import VoskAdapter
from wordalign.engines.adapters.qwen_adapter import QwenAdapter
from wordalign.engines.adapters.mfa_adapter import MFAAdapter
from wordalign.engines.adapters.yamnet_adapter import YAMNetAdapter
from wordalign.models.catalog import ModelCatalog, ModelEntry
from wordalign.models.manager import ModelManager
from wordalign.models.validator import ModelValidator
from wordalign.models.downloader import DownloadManager
from wordalign.runtimes.detector import HardwareDetector, HardwareInfo
from wordalign.runtimes.manager import RuntimeManager, RuntimeStatus
from wordalign.qa.engine import QAEngine
from wordalign.qa.deterministic import run_deterministic_qa
from wordalign.qa.issues import TranscriptIssue, RiskScore

# Plugin registry
reg = PluginRegistry()
reg.register(WhisperXAdapter())
reg.register(VoskAdapter())
reg.register(QwenAdapter())
reg.register(MFAAdapter())
reg.register(YAMNetAdapter())
print("Registered plugins:", reg.list_plugins())
print("TRANSCRIBE engines:", reg.query(Capability.TRANSCRIBE))
print("FORCED_ALIGN engines:", reg.query(Capability.FORCED_ALIGN))
print("VOTE engines:", reg.query(Capability.VOTE))
print("AUDIO_EVENTS engines:", reg.query(Capability.AUDIO_EVENTS))

# Model catalog
cat = ModelCatalog("wordalign/data/model_catalog.json")
print("Models in catalog:", len(cat.list_all()))
print("Recommended:", [m.display_name for m in cat.list_recommended()])

# Profiles
for preset in ["fast", "balanced", "maximum_quality", "cpu_only"]:
    p = PipelineProfile.preset(preset)
    print("Profile %s: %d engines, CPL=%d" % (preset, len(p.transcription.engines), p.segmentation.max_cpl))

# QA
words = [
    WordResult(id="w000001", text="hello", normalized_text="hello", start=0.0, end=0.3, confidence=0.95),
    WordResult(id="w000002", text="bad", normalized_text="bad", start=0.3, end=0.6, confidence=0.2),
    WordResult(id="w000003", text="world", normalized_text="world", start=0.6, end=0.9, confidence=0.9),
]
issues = run_deterministic_qa(words)
print("QA issues found:", len(issues))
for issue in issues:
    print("  [%s] %s: %s - %s" % (issue.severity, issue.category, issue.original_text, issue.explanation[:60]))

# Dummy plugin (pluggability test)
class SuperASR2027(EnginePlugin):
    def descriptor(self):
        return EngineDescriptor(engine_id="super_asr_2027", display_name="SuperASR 2027", version="1.0",
            capabilities=Capability.TRANSCRIBE|Capability.WORD_TIMING|Capability.VOTE,
            runtime_type="in_process", supported_languages=["en"], models=["super-v1"],
            hardware={"cpu": False, "cuda": True})
    def health_check(self):
        return HealthStatus(ready=True, runtime_status="ready")

reg.register(SuperASR2027())
assert "super_asr_2027" in reg.query(Capability.TRANSCRIBE)
print("Dummy plugin registered:", "super_asr_2027" in reg)
print("Total plugins:", len(reg))

print()
print("ALL VERIFICATIONS PASSED")
