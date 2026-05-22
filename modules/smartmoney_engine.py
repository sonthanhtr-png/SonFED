from __future__ import annotations

import pandas as pd


def smartmoney_notes(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["Chưa đủ dữ liệu smart money."]
    last = df.iloc[-1]
    notes = []
    if bool(last.get("LIQUIDITY_SWEEP_UP", False)):
        notes.append("Quét thanh khoản phía trên: cẩn trọng với BUY đuổi.")
    if bool(last.get("LIQUIDITY_SWEEP_DOWN", False)):
        notes.append("Quét thanh khoản phía dưới: có thể xuất hiện nhịp hồi kỹ thuật.")
    if bool(last.get("BOS_UP", False)):
        notes.append("Phá cấu trúc tăng kèm volume: hạn chế SELL sớm.")
    if bool(last.get("BOS_DOWN", False)):
        notes.append("Phá cấu trúc giảm kèm volume: ưu tiên chờ hồi để SELL.")
    return notes or ["Chưa có dấu hiệu quét thanh khoản hoặc phá cấu trúc rõ."]
