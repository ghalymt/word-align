"""PySide6 native GUI for WordAlign.

A real Qt desktop application with:
- Native file dialogs
- Background pipeline worker (QThread)
- Progress + stage events in real time
- QA issues list with accept/dismiss actions
- Model path configuration

Requires PySide6:  pip install PySide6

Launch:  python -m wordalign.gui.native
"""
from __future__ import annotations

import sys


def main() -> int:
    """Launch the native GUI."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("[error] PySide6 is not installed.")
        print("        Install it with:  pip install PySide6")
        print("        Then run again:   python -m wordalign.gui.native")
        return 1

    from .main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("WordAlign")
    app.setOrganizationName("WordAlign")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
