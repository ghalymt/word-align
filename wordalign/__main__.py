"""Package entry point: ``python -m wordalign``.

The ``if __name__ == "__main__"`` guard is load-bearing, not decorative.
The Vosk stage uses a ``ProcessPoolExecutor``; on Windows (and anywhere the
start method is ``spawn``) each worker boots a fresh interpreter that
re-imports this module as ``__mp_main__``. Without the guard, ``main()`` would
re-run inside every worker — restarting the whole pipeline recursively. With
it, the guard is False in the children and only the parent runs ``main()``.
"""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
