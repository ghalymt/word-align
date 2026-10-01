"""Entry point for the portable WordAlign executable.

This file lives at the repository root so PyInstaller can run it as a
bare script.  It imports from the ``wordalign`` package using absolute
imports, avoiding the relative-import problem that breaks ``__main__.py``
inside a frozen bundle.

The portable layout (E:\\Wordalign\\models\\) is the default source of truth
for models. Explicit backend, LLM, and cache overrides supplied by the user
are preserved; stale Hugging Face cache variables are scrubbed before the
local model tree is resolved.

Each heavy backend (Qwen3-ASR, WhisperX, MFA) lives in its own venv under
``WordAlign/venvs/<backend>/`` to keep conflicting torch/transformers pins
from polluting the main bundle. If the user hasn't set the corresponding
``WORDALIGN_*_PYTHON`` / ``WORDALIGN_MFA`` env var, we auto-resolve it to
the bundled venv here so the rest of the code can stay env-var-driven.
"""
import multiprocessing
import os
import sys
from pathlib import Path


multiprocessing.freeze_support()

# Stale model-cache variables are scrubbed so a previous install cannot
# silently redirect the portable model tree. Explicit backend, LLM, and
# cache overrides remain available to the user.
for _key in (
    "HF_HOME",
    "HF_HUB_CACHE",
):
    os.environ.pop(_key, None)


def _bundle_root() -> Path:
    """Locate the portable WordAlign folder.

    - Frozen onefile: sys.executable is the launcher, parent = bundle root.
    - Frozen onedir:  sys.executable is in bundle root.
    - Dev (running from source): the repo root (one above ``wordalign/``).
    """
    exe = Path(sys.executable).resolve()
    if getattr(sys, "frozen", False):
        return exe.parent
    # Dev mode: assume launcher.py lives at the repo root.
    return Path(__file__).resolve().parent


def _resolve_backend_venv(bundle: Path, sub: str, env_var: str) -> None:
    """If *env_var* isn't set, look for ``bundle/venvs/<sub>/python`` in
    either the Windows (``Scripts/python.exe``) or Unix
    (``bin/python``) layout, and point *env_var* at it. Silently no-ops
    if the venv doesn't exist (the engine layer will surface a clearer
    "backend missing" message at use time)."""
    if os.environ.get(env_var):
        return
    candidates = [
        bundle / "venvs" / sub / "Scripts" / "python.exe",   # Windows venv
        bundle / "venvs" / sub / "bin" / "python",           # Unix venv
        bundle / "venvs" / sub / "python.exe",               # Windows flat
        bundle / "venvs" / sub / "python",                   # Unix flat
    ]
    for py in candidates:
        if py.is_file():
            os.environ[env_var] = str(py)
            return


def _resolve_mfa(bundle: Path) -> None:
    if os.environ.get("WORDALIGN_MFA"):
        return
    mfa_root = bundle / "venvs" / "mfa"
    if ((mfa_root / "Scripts" / "mfa.exe").is_file()
            or (mfa_root / "bin" / "mfa").is_file()):
        os.environ["WORDALIGN_MFA"] = str(mfa_root)
        return
    mfa = mfa_root / ("mfa.exe" if os.name == "nt" else "mfa")
    if mfa.is_file():
        os.environ["WORDALIGN_MFA"] = str(mfa)


_BUNDLE = _bundle_root()
_resolve_backend_venv(_BUNDLE, "qwen", "WORDALIGN_QWEN_PYTHON")
_resolve_backend_venv(_BUNDLE, "whisperx", "WORDALIGN_WHISPERX_PYTHON")
_resolve_mfa(_BUNDLE)

# Debug: surface venv auto-resolution on stderr so we can verify the bundle
# layout at runtime without needing to dig through the frozen _internal/.
print(f"[launcher] bundle root: {_BUNDLE}", file=sys.stderr)
print(f"[launcher] WORDALIGN_QWEN_PYTHON     = {os.environ.get('WORDALIGN_QWEN_PYTHON')!r}", file=sys.stderr)
print(f"[launcher] WORDALIGN_WHISPERX_PYTHON = {os.environ.get('WORDALIGN_WHISPERX_PYTHON')!r}", file=sys.stderr)
print(f"[launcher] WORDALIGN_MFA             = {os.environ.get('WORDALIGN_MFA')!r}", file=sys.stderr)

def main() -> int:
    if len(sys.argv) > 1:
        from wordalign.cli_v2 import main as cli_main
        return cli_main()
    from wordalign.gui.server import run_server
    run_server(port=int(os.environ.get("WORDALIGN_PORT", "5575")),
               open_browser=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
