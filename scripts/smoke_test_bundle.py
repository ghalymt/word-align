"""Smoke-test a built WordAlign bundle (used by the release workflow).

    python scripts/smoke_test_bundle.py --exe dist/WordAlign/WordAlign.exe
        [--vosk-models DIR] [--expect-version 2.1.0]

The bundle runs with a bare system PATH, so it has to bring its own
Python, FFmpeg and Vosk libraries -- exactly what an end user's machine
looks like. Checks, in order:

1. ``--version`` prints the expected version, and ``--help`` lists the
   2.1 flags (catches modules PyInstaller left out);
2. a reference-mode run on the test clip writes SRT, WebVTT and ASS;
3. with ``--vosk-models``: a Vosk run times most words with Vosk (the
   bundled libvosk and its DLLs load and work);
4. the GUI starts the way ``Launch WordAlign.vbs`` starts it, serves the
   page and the API, and refuses a cross-origin POST.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIP = ROOT / "tests" / "golden" / "audio" / "espeak_en.flac"
CLIP_TEXT = ROOT / "tests" / "golden" / "audio" / "espeak_en.txt"
ENGINES_OFF = ["--no-vosk", "--no-qwen", "--no-whisperx", "--no-mfa"]


def bare_env(workdir: Path) -> dict:
    """The environment of a machine with nothing installed."""
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith(("PYTHON", "WORDALIGN_", "VIRTUAL_ENV",
                                        "CONDA"))}
    if os.name == "nt":
        root = os.environ.get("SystemRoot", r"C:\Windows")
        env["PATH"] = os.pathsep.join([os.path.join(root, "System32"), root])
    env["WORDALIGN_CACHE"] = str(workdir / "cache")
    # Never open a real browser from the GUI check.
    env["BROWSER"] = "wordalign-smoke-test-no-browser"
    return env


def run(cmd, env, timeout=600) -> str:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run([str(c) for c in cmd], env=env, timeout=timeout,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(output)
        raise SystemExit(f"FAILED (exit {proc.returncode})")
    return output


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAILED: {message}")
    print(f"  ok: {message}")


def source_version() -> str:
    text = (ROOT / "wordalign" / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'__version__ = "([^"]+)"', text).group(1)


def check_cli(exe: Path, env: dict, version: str) -> None:
    out = run([exe, "--version"], env)
    check(version in out, f"--version reports {version}")
    out = run([exe, "--help"], env)
    for flag in ("--formats", "--batch", "--diarize", "--max-lines"):
        check(flag in out, f"--help lists {flag}")


def check_reference_run(exe: Path, env: dict, workdir: Path) -> None:
    out_dir = workdir / "reference"
    run([exe, CLIP, "-t", CLIP_TEXT, "-o", out_dir, "-l", "en",
         "--formats", "srt,vtt,ass", *ENGINES_OFF], env)
    for ext in ("srt", "vtt", "ass"):
        path = out_dir / f"{CLIP.stem}_sentence_level.{ext}"
        check(path.is_file() and path.stat().st_size > 0,
              f"reference run wrote {path.name}")


def check_vosk_run(exe: Path, env: dict, workdir: Path, models: Path) -> None:
    out_dir = workdir / "vosk"
    run([exe, CLIP, "-t", CLIP_TEXT, "-o", out_dir, "-l", "en",
         "--engines", "vosk", "--no-mfa", "--vosk-models", models], env)
    manifest = json.loads((out_dir / f"{CLIP.stem}_job_manifest.json")
                          .read_text(encoding="utf-8"))
    sources = manifest["word_stats"]["sources"]
    total = manifest["word_stats"]["total"]
    timed = total - sources.get("Interpolated", 0)
    print(f"  word sources: {sources}")
    check(total > 0 and timed / total >= 0.5,
          f"Vosk timed {timed}/{total} words")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def check_gui(exe: Path, env: dict) -> None:
    port = free_port()
    env = dict(env, WORDALIGN_PORT=str(port))
    base = f"http://127.0.0.1:{port}"
    # No arguments: the way Launch WordAlign.vbs starts the app.
    proc = subprocess.Popen([str(exe)], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    try:
        deadline = time.time() + 120
        while True:
            try:
                with urllib.request.urlopen(base + "/api/health", timeout=5) as r:
                    health = json.load(r)
                break
            except (urllib.error.URLError, ConnectionError, OSError):
                if proc.poll() is not None or time.time() > deadline:
                    print(proc.stdout.read() if proc.stdout else "")
                    raise SystemExit("FAILED: the GUI server did not start")
                time.sleep(1)
        check(health.get("status") == "ok", "GUI /api/health answers")
        with urllib.request.urlopen(base + "/", timeout=10) as r:
            page = r.read().decode("utf-8", "replace")
        check("<html" in page.lower(), "GUI page is served")
        request = urllib.request.Request(
            base + "/api/cache/clear", data=b"{}", method="POST",
            headers={"Origin": "https://example.com",
                     "Content-Type": "application/json"})
        try:
            urllib.request.urlopen(request, timeout=10)
            status = 200
        except urllib.error.HTTPError as exc:
            status = exc.code
        check(status == 403, "cross-origin POST is refused")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--exe", required=True, type=Path,
                        help="WordAlign.exe of the built bundle")
    parser.add_argument("--vosk-models", type=Path,
                        help="folder holding vosk-model-en-us-0.22 (any "
                             "English Vosk model under that name)")
    parser.add_argument("--expect-version", default=None,
                        help="default: the version in wordalign/__init__.py")
    args = parser.parse_args(argv)

    exe = args.exe.resolve()
    if not exe.is_file():
        raise SystemExit(f"FAILED: {exe} not found")
    version = args.expect_version or source_version()
    with tempfile.TemporaryDirectory(prefix="wordalign_smoke_") as tmp:
        workdir = Path(tmp)
        env = bare_env(workdir)
        print("[1/4] CLI"); check_cli(exe, env, version)
        print("[2/4] reference run"); check_reference_run(exe, env, workdir)
        if args.vosk_models:
            print("[3/4] Vosk run")
            check_vosk_run(exe, env, workdir, args.vosk_models.resolve())
        else:
            print("[3/4] Vosk run skipped (no --vosk-models)")
        print("[4/4] GUI"); check_gui(exe, env)
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
