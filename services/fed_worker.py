from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal

from ai.desktop_engine import SonFEDDesktopEngine
from shared.file_bus import DEFAULT_SHARED_DIR


class SonFEDWorker(QThread):
    snapshot = Signal(dict)
    log = Signal(str)

    def __init__(self, interval_seconds: int = 15, parent=None) -> None:
        super().__init__(parent)
        self.interval_seconds = max(5, int(interval_seconds))
        self.engine = SonFEDDesktopEngine(DEFAULT_SHARED_DIR)
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self.log.emit("SonFED AI worker started.")
        while self._running:
            try:
                payload = self.engine.refresh()
                self.snapshot.emit(payload)
                signal = payload.get("signal", {})
                self.log.emit(f"Signal {signal.get('action', 'WAIT')} | {signal.get('confidence', 0)}% | wrote shared folder.")
            except Exception as exc:
                self.log.emit(f"SonFED refresh error: {exc}\n{traceback.format_exc(limit=2)}")
            self.msleep(self.interval_seconds * 1000)
        self.log.emit("SonFED AI worker stopped.")
