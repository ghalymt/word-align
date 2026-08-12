# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building a portable WordAlign Windows executable.

Usage:
    pyinstaller build.spec

Produces dist/WordAlign/ containing WordAlign.exe plus bundled Python,
FFmpeg, and all pure-Python dependencies. End users need no Python install.
"""
import os

# Bundle FFmpeg binaries next to the exe if present on this machine
ffmpeg_bin = r"C:\ProgramData\chocolatey\bin"
ffmpeg_files = []
if os.path.isdir(ffmpeg_bin):
    for name in ("ffmpeg.exe", "ffprobe.exe", "ffplay.exe"):
        p = os.path.join(ffmpeg_bin, name)
        if os.path.exists(p):
            ffmpeg_files.append((p, "."))

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=ffmpeg_files,
    datas=[
        ("wordalign/data/model_catalog.json", "wordalign/data"),
        ("wordalign/profiles/*.json", "wordalign/profiles"),
        ("wordalign/gui/app.html", "wordalign/gui"),
    ],
    hiddenimports=[
        "srt",
        "langdetect",
        "textgrid",
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
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="WordAlign",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,          # CLI application — console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
