"""Entry point for the portable WordAlign executable.

This file lives at the repository root so PyInstaller can run it as a
bare script.  It imports from the ``wordalign`` package using absolute
imports, avoiding the relative-import problem that breaks ``__main__.py``
inside a frozen bundle.
"""
import sys

from wordalign.cli_v2 import main

if __name__ == "__main__":
    # If CLI args were passed (e.g. from a terminal), run the CLI pipeline.
    # If double-clicked (no args), launch the GUI automatically.
    if len(sys.argv) > 1:
        from wordalign.cli_v2 import main
        sys.exit(main())
    else:
        from wordalign.gui.server import run_server
        run_server(port=5575, open_browser=True)
        sys.exit(0)
