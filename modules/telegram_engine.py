from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

load_dotenv()


def send_telegram(text: str) -> tuple[bool, str]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False, "Chưa cấu hình TELEGRAM_BOT_TOKEN hoặc TELEGRAM_CHAT_ID."
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
        resp.raise_for_status()
        return True, "Đã gửi Telegram."
    except Exception as exc:
        return False, f"Lỗi gửi Telegram: {exc}"


def build_alert(gold_price: float, gold_analysis: dict, macro: dict, signal: dict, levels: dict) -> str:
    return "\n".join([
        "SonFED Alert",
        "",
        f"Giá vàng: {gold_price:.2f}",
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
