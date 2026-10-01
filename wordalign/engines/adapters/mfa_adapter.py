"""MFA adapter."""
from __future__ import annotations

from ...models.paths import ModelPaths
from ...plugins.base import EngineDescriptor, EnginePlugin, HealthStatus
from ...plugins.capabilities import Capability


class MFAAdapter(EnginePlugin):

    def descriptor(self) -> EngineDescriptor:
        return EngineDescriptor(
            engine_id="mfa",
            display_name="Montreal Forced Aligner",
            version="1.0",
            capabilities=Capability.FORCED_ALIGN,
            runtime_type="external",
            supported_languages=["en"],
            models=["english_us_arpa"],
            hardware={"cpu": True, "cuda": False},
            help_text="Gold-standard forced alignment. English only (for now).",
        )

    def health_check(self) -> HealthStatus:
        import os
        import shutil
        mfa = os.environ.get("WORDALIGN_MFA") or shutil.which("mfa")
        if not mfa:
            return HealthStatus(ready=False, runtime_status="missing",
                                missing_components=["mfa"],
                                message="Install Montreal Forced Aligner or set WORDALIGN_MFA")
        model_root = ModelPaths().resolve("mfa")
        required = (
            model_root / "acoustic" / "english_us_arpa.zip"
            if model_root else None,
            model_root / "dictionary" / "english_us_arpa.dict"
            if model_root else None,
            model_root / "g2p" / "english_us_arpa.zip"
            if model_root else None,
        )
        if not all(path and path.is_file() for path in required):
            return HealthStatus(ready=False, runtime_status="missing",
                                missing_components=["mfa_models"],
                                message="Install the English MFA models under models/mfa")
        return HealthStatus(ready=True, runtime_status="ready",
                            message=f"MFA: {mfa}")

    def align(self, request) -> dict:
        from ..mfa_engine import MFAWrapper
        from ...align import run_mfa_on_gaps
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="mfa_surgical_")
        wrapper = MFAWrapper(work_dir=temp_dir,
                             mfa_cmd=request.options.get("mfa_cmd"))
        # run_mfa_on_gaps mutates aligned_words in place
        run_mfa_on_gaps(request.options["aligned_words"],
                        request.audio_path, wrapper)
        return {"engine_id": "mfa",
                "aligned_words": request.options["aligned_words"]}
