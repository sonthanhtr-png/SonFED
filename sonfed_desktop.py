from __future__ import annotations

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        sys.stderr.write("PySide6 is not installed. Run: pip install PySide6\n")
        return 1

    from ui.fed_terminal import SonFEDTerminal

    app = QApplication(sys.argv)
    window = SonFEDTerminal()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
