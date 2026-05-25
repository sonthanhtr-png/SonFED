from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from .utils import ROOT


load_dotenv()


TELEGRAM_STATE_PATH = ROOT / "data" / "telegram_state.json"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class TelegramQueue:
    def __init__(self, enabled: bool = True, state_path: str | Path = TELEGRAM_STATE_PATH, cooldown_seconds: int = 300) -> None:
        self.enabled = enabled
        self.state_path = Path(state_path)
        self.cooldown_seconds = int(cooldown_seconds)
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "") or str(self._state().get("chat_id", ""))
        self._queue: queue.Queue[tuple[str, str, int]] = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _save_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _key(text: str, event_key: str | None = None) -> str:
        return hashlib.sha1((event_key or text).encode("utf-8", errors="ignore")).hexdigest()

    def mark_sent(self, key: str) -> None:
        state = self._state()
        sent = state.setdefault("sent", {})
        sent[key] = time.time()
        if len(sent) > 500:
            state["sent"] = dict(sorted(sent.items(), key=lambda item: item[1])[-500:])
        self._save_state(state)

    def should_send(self, text: str, event_key: str | None = None, cooldown_seconds: int | None = None) -> tuple[bool, str]:
        key = self._key(text, event_key)
        state = self._state()
        last = float(state.get("sent", {}).get(key, 0) or 0)
        cooldown = self.cooldown_seconds if cooldown_seconds is None else int(cooldown_seconds)
        return (time.time() - last >= cooldown), key

    def send_now(self, text: str) -> tuple[bool, str]:
        if not self.enabled:
            return False, "Telegram đang tắt."
        if not self.token:
            return False, "Chưa cấu hình TELEGRAM_BOT_TOKEN."
        if not self.chat_id:
            self.chat_id = self.discover_chat_id()
        if not self.chat_id:
            return False, "Chưa có TELEGRAM_CHAT_ID. Hãy mở bot Telegram và gửi /start một lần."
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text},
                timeout=10,
            )
            resp.raise_for_status()
            return True, "Đã gửi Telegram."
        except Exception as exc:
            return False, f"Lỗi gửi Telegram: {exc}"

    def discover_chat_id(self) -> str:
        if not self.token:
            return ""
        try:
            resp = requests.get(f"https://api.telegram.org/bot{self.token}/getUpdates", timeout=8)
            resp.raise_for_status()
            updates = resp.json().get("result", [])
        except Exception:
            return ""
        for item in reversed(updates):
            message = item.get("message") or item.get("edited_message") or {}
            chat = message.get("chat", {})
            chat_id = chat.get("id")
            if chat_id:
                state = self._state()
                state["chat_id"] = str(chat_id)
                self._save_state(state)
                return str(chat_id)
        return ""

    def send_queued(self, text: str, event_key: str | None = None, cooldown_seconds: int | None = None, retries: int = 2) -> tuple[bool, str]:
        ok, key = self.should_send(text, event_key, cooldown_seconds)
        if not ok:
            return False, "Telegram cooldown: bỏ qua tin trùng hoặc gửi quá gần."
        self.mark_sent(key)
        self._queue.put((text, key, int(retries)))
        return True, "Đã đưa tin Telegram vào queue."

    def _worker_loop(self) -> None:
        while True:
            text, key, retries = self._queue.get()
            try:
                ok, _ = self.send_now(text)
                if not ok and retries > 0:
                    time.sleep(2)
                    self._queue.put((text, key, retries - 1))
            finally:
                self._queue.task_done()


def send_telegram(text: str) -> tuple[bool, str]:
    return TelegramQueue(enabled=True).send_now(text)


def send_telegram_queued(text: str, enabled: bool = True, event_key: str | None = None, cooldown_seconds: int = 300) -> tuple[bool, str]:
    return TelegramQueue(enabled=enabled, cooldown_seconds=cooldown_seconds).send_queued(text, event_key, cooldown_seconds)


def build_alert(gold_price: float, gold_analysis: dict, macro: dict, signal: dict, levels: dict) -> str:
    return "\n".join([
        "SonFED Alert",
        "",
        f"Giá vàng: {_safe_float(gold_price):.2f}",
        f"SonFED Pressure Index: {macro.get('score')} - {macro.get('interpretation')}",
        f"Hỗ trợ: {levels.get('support')}",
        f"Kháng cự: {levels.get('resistance')}",
        "",
        "Kết luận:",
        gold_analysis.get("summary", ""),
        "",
        "Chiến lược:",
        f"{signal.get('strategy')} - {signal.get('action')} - độ tin cậy {signal.get('confidence')}%",
        "",
        "Lưu ý:",
        signal.get("reason", ""),
    ])
