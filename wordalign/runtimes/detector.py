"""Hardware detector — replaces standalone preflight.py with a reusable API."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class HardwareInfo:
    """System hardware capabilities."""
    os: str = ""
    os_version: str = ""
    architecture: str = ""
    cpu_count: int = 0
    cpu_brand: str = ""
    ram_gb: float = 0.0
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    cuda_available: bool = False
    cuda_version: str = ""
    ffmpeg_available: bool = False
    ffmpeg_path: str = ""
    ffprobe_available: bool = False
    disk_free_gb: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"OS: {self.os} {self.os_version} ({self.architecture})",
            f"CPU: {self.cpu_brand} ({self.cpu_count} cores)",
            f"RAM: {self.ram_gb:.1f} GB",
        ]
        if self.cuda_available:
            lines.append(f"GPU: {self.gpu_name} ({self.gpu_vram_gb:.1f} GB VRAM)")
            lines.append(f"CUDA: {self.cuda_version}")
        else:
            lines.append("GPU: None / CUDA not available")
        lines.append(f"FFmpeg: {'yes' if self.ffmpeg_available else 'no'}")
        lines.append(f"Disk free: {self.disk_free_gb:.1f} GB")
        return "\n".join(lines)

    def can_run_model(self, min_vram: float, min_ram: float) -> tuple[bool, list[str]]:
        """Check if this hardware can run a model with the given requirements."""
        issues = []
        if min_vram > 0 and self.gpu_vram_gb < min_vram:
            issues.append(f"VRAM: {self.gpu_vram_gb:.1f} GB < required {min_vram} GB")
        if min_ram > 0 and self.ram_gb < min_ram:
            issues.append(f"RAM: {self.ram_gb:.1f} GB < required {min_ram} GB")
        return (len(issues) == 0, issues)


class HardwareDetector:
    """Detects system hardware capabilities."""

    def detect(self) -> HardwareInfo:
        info = HardwareInfo()
        info.os = platform.system()
        info.os_version = platform.version()
        info.architecture = platform.machine()
        info.cpu_count = os.cpu_count() or 1
        info.cpu_brand = platform.processor() or "Unknown"

        # RAM
        try:
            import psutil
            info.ram_gb = psutil.virtual_memory().total / (1024**3)
            usage = psutil.disk_usage(os.getcwd())
            info.disk_free_gb = usage.free / (1024**3)
        except ImportError:
            pass

        # CUDA / GPU
        try:
            import torch
            if torch.cuda.is_available():
                info.cuda_available = True
                info.cuda_version = str(torch.version.cuda or "")
                idx = torch.cuda.current_device()
                info.gpu_name = torch.cuda.get_device_name(idx)
                info.gpu_vram_gb = (
                    torch.cuda.get_device_properties(idx).total_memory / (1024**3))
        except ImportError:
            pass
        if not info.cuda_available:
            self._detect_cuda_backend(info)

        # FFmpeg
        info.ffmpeg_path = shutil.which("ffmpeg") or ""
        info.ffmpeg_available = bool(info.ffmpeg_path)
        info.ffprobe_available = bool(shutil.which("ffprobe"))

        return info

    def _detect_cuda_backend(self, info: HardwareInfo) -> None:
        script = (
            "import json, torch; "
            "available=torch.cuda.is_available(); "
            "index=torch.cuda.current_device() if available else 0; "
            "print(json.dumps({'available': available, "
            "'name': torch.cuda.get_device_name(index) if available else '', "
            "'vram': torch.cuda.get_device_properties(index).total_memory / (1024**3) "
            "if available else 0.0, "
            "'cuda': str(torch.version.cuda or '')}))"
        )
        for variable in ("WORDALIGN_WHISPERX_PYTHON", "WORDALIGN_QWEN_PYTHON"):
            python = os.environ.get(variable)
            if not python or not Path(python).is_file():
                continue
            try:
                probe = subprocess.run(
                    [python, "-c", script], capture_output=True, text=True,
                    timeout=30, check=False)
                if probe.returncode != 0:
                    continue
                payload = json.loads(probe.stdout.strip().splitlines()[-1])
            except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
                continue
            if payload.get("available"):
                info.cuda_available = True
                info.gpu_name = str(payload.get("name", ""))
                info.gpu_vram_gb = float(payload.get("vram", 0.0))
                info.cuda_version = str(payload.get("cuda", ""))
                return

    def check_ffmpeg_works(self) -> bool:
        """Verify ffprobe actually executes, not just exists on PATH."""
        if not shutil.which("ffprobe"):
            return False
        try:
            subprocess.run(["ffprobe", "-version"],
                           capture_output=True, text=True, timeout=10)
            return True
        except Exception:
            return False
