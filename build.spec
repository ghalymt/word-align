# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building a portable WordAlign Windows executable.

Usage:
    pyinstaller build.spec

Produces dist/WordAlign/ containing WordAlign.exe plus bundled Python,
FFmpeg, and all pure-Python dependencies. End users need no Python install.
"""
import importlib.util
import os
import shutil
from pathlib import Path

def _resolve_tool(name):
    candidates = []
    if os.name == "nt":
        candidates.extend([
            os.path.join(r"C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg\bin", name),
            os.path.join(r"C:\ffmpeg\bin", name),
        ])
    resolved = shutil.which(name)
    if resolved:
        candidates.append(resolved)
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


ffmpeg_files = []
for name in ("ffmpeg.exe", "ffprobe.exe", "ffplay.exe"):
    path = _resolve_tool(name)
    if path:
        ffmpeg_files.append((path, "."))

vosk_binaries = []
vosk_spec = importlib.util.find_spec("vosk")
if vosk_spec and vosk_spec.origin:
    vosk_src = Path(vosk_spec.origin).resolve().parent
    if vosk_src.is_dir():
        for path in vosk_src.iterdir():
            if path.suffix.lower() in {".dll", ".so", ".dylib"}:
                vosk_binaries.append((str(path), "vosk"))

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=ffmpeg_files + vosk_binaries,
    datas=[
        ("wordalign/data/model_catalog.json", "wordalign/data"),
        ("wordalign/profiles/*.json", "wordalign/profiles"),
        ("wordalign/gui/app.html", "wordalign/gui"),
        # Worker scripts invoked by subprocess from qwen/whisperx engines.
        # Not auto-included by Analysis because they are only referenced by
        # file path (Path(__file__).with_name(...)), never imported.
        ("wordalign/engines/_qwen_worker.py", "wordalign/engines"),
        ("wordalign/engines/_whisperx_worker.py", "wordalign/engines"),
    ],
    hiddenimports=[
        "srt",
        "langdetect",
        "textgrid",
        "qwen_asr",               # optional Qwen3-ASR voter (graceful skip if import fails)
        "qwen_asr.inference.qwen3_asr",
        "wordalign.core",
        "wordalign.core.cache",
        "wordalign.core.config",
        "wordalign.core.converters",
        "wordalign.core.database",
        "wordalign.core.errors",
        "wordalign.core.events",
        "wordalign.core.fingerprint",
        "wordalign.core.gpu_scheduler",
        "wordalign.core.manifest",
        "wordalign.core.pipeline",
        "wordalign.core.recovery",
        "wordalign.core.types",
        "wordalign.engines.adapters",
        "wordalign.engines.adapters.diarization_adapter",
        "wordalign.engines.adapters.mfa_adapter",
        "wordalign.engines.adapters.nemo_adapter",
        "wordalign.engines.adapters.qwen_adapter",
        "wordalign.engines.adapters.vosk_adapter",
        "wordalign.engines.adapters.whisperx_adapter",
        "wordalign.engines.adapters.yamnet_adapter",
        "wordalign.gui.server",
        "wordalign.gui.waveform",
        "wordalign.models",
        "wordalign.models.catalog",
        "wordalign.models.downloader",
        "wordalign.models.manager",
        "wordalign.models.validator",
        "wordalign.plugins",
        "wordalign.plugins.base",
        "wordalign.plugins.capabilities",
        "wordalign.plugins.manifest",
        "wordalign.plugins.protocol",
        "wordalign.plugins.registry",
        "wordalign.qa",
        "wordalign.qa.codeswitch",
        "wordalign.qa.deterministic",
        "wordalign.qa.engine",
        "wordalign.qa.issues",
        "wordalign.qa.providers",
        "wordalign.qa.realignment",
        "wordalign.runtimes",
        "wordalign.runtimes.detector",
        "wordalign.runtimes.manager",
    ],
    hookspath=[],
    runtime_hooks=['scripts/runtime_hook.py'],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,    # one-folder: binaries live in _internal/
    name="WordAlign",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,             # CLI application — console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="WordAlign",
)
