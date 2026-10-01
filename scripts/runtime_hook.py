"""PyInstaller runtime hook for WordAlign.

Runs at the very start of the frozen .exe, BEFORE any other import.
Fixes the vosk DLL lookup so the bundled app can actually transcribe.

The problem chain we're working around:
  1. ``vosk/__init__.py`` does ``os.path.dirname(__file__).add_dll_directory``
     and ``_ffi.dlopen(.../vosk/libvosk.dll)`` at import time. In a
     one-file PyInstaller bundle, ``libvosk.dll`` lives at ``_MEIPASS``
     (the bundle root), NOT inside ``_MEIPASS/vosk/``, so both calls
     fail.
  2. ``libvosk.dll`` itself depends on the MinGW C++ runtime
     (``libgcc_s_seh-1.dll``, ``libstdc++-6.dll``,
     ``libwinpthread-1.dll``) which PyInstaller places in the bundle
     root too — so even after fixing #1 the load fails with
     error 0x7e (MOD_NOT_FOUND).

We fix both by copying every DLL that ships alongside the source
``vosk`` package into ``_MEIPASS/vosk/`` so the bundled import-time
lookup succeeds with no external dependency on the host's
site-packages directory.
"""
import importlib.util
import os
import shutil
import sys

if getattr(sys, "frozen", False):
    meipass = getattr(sys, "_MEIPASS", None)

    # (1) Robust os.add_dll_directory: ignore non-existent paths.
    if hasattr(os, "add_dll_directory"):
        _real_add = os.add_dll_directory
        def _safe_add(path):
            if not path or not os.path.isdir(path):
                return None
            return _real_add(path)
        os.add_dll_directory = _safe_add

    if meipass:
        # (2) Make the bundle root DLL-searchable.
        os.environ["PATH"] = meipass + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(meipass)
        except Exception:
            pass

        # (2b) Locate the host's real ffmpeg/ffprobe and expose the
        # absolute path to the pipeline via env vars. The pipeline
        # (wordalign/utils.py) will prefer these over a bare "ffmpeg"
        # PATH lookup — the bare lookup would hit torchaudio's
        # bundled shim (which is broken in a one-file PyInstaller
        # build) before the real binary.
        ffmpeg_dirs = [
            meipass,
            os.path.dirname(os.path.abspath(sys.executable)),
            os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "_internal"),
        ]
        for tool, envvar in (("ffmpeg.exe", "WORDALIGN_FFMPEG"),
                              ("ffprobe.exe", "WORDALIGN_FFPROBE")):
            src_tool = shutil.which(tool)
            if not src_tool:
                for directory in ffmpeg_dirs:
                    candidate = os.path.join(directory, tool)
                    if os.path.isfile(candidate):
                        src_tool = candidate
                        break
            if src_tool:
                os.environ[envvar] = src_tool
                dst_tool = os.path.join(meipass, tool)
                if not os.path.isfile(dst_tool):
                    try:
                        shutil.copy2(src_tool, dst_tool)
                    except OSError:
                        pass

        vosk_dir = os.path.join(meipass, "vosk")
        os.makedirs(vosk_dir, exist_ok=True)

        # (3) Locate the source vosk package — it ships libvosk.dll and
        # its MinGW dependencies as siblings of __init__.py. We copy
        # every DLL we find there into the bundled vosk/ subdir so the
        # import-time lookup in vosk/__init__.py succeeds.
        candidates = []
        try:
            vosk_spec = importlib.util.find_spec("vosk")
            if vosk_spec and vosk_spec.origin:
                candidates.append(os.path.dirname(vosk_spec.origin))
        except Exception:
            pass
        candidates.append(os.path.join(meipass, "vosk"))

        for src_dir in candidates:
            if not src_dir or not os.path.isdir(src_dir):
                continue
            for name in os.listdir(src_dir):
                if not name.lower().endswith(".dll"):
                    continue
                src = os.path.join(src_dir, name)
                if not os.path.isfile(src):
                    continue
                dst = os.path.join(vosk_dir, name)
                if not os.path.isfile(dst):
                    try:
                        shutil.copy2(src, dst)
                    except OSError:
                        pass

        # (4) Make sure vosk/ is on the DLL search path now that it
        # has its siblings in place.
        try:
            os.add_dll_directory(vosk_dir)
        except Exception:
            pass
