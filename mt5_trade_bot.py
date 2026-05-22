from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

from modules.utils import load_json, save_json

ROOT = Path(__file__).resolve().parent
CONFIG = load_json(ROOT / "config.json", {})
SHARED = Path(CONFIG.get("paths", {}).get("shared_dir", "C:\\SonFED\\shared"))
try:
    SHARED.mkdir(parents=True, exist_ok=True)
except Exception:
    SHARED = ROOT / "shared"
    SHARED.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    rows = load_json(SHARED / "bot_log.json", [])
    if not isinstance(rows, list):
        rows = []
    rows.append(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {message}")
    save_json(SHARED / "bot_log.json", rows[-200:])


def signal_is_fresh(signal: dict, max_age: int) -> bool:
    try:
        created = datetime.strptime(signal.get("time", ""), "%Y-%m-%d %H:%M:%S")
        return (datetime.now() - created).total_seconds() <= max_age
    except Exception:
        return False


def write_status(symbol: str) -> None:
    if mt5 is None or not mt5.initialize():
        save_json(SHARED / "trade_status.json", {"symbol": symbol, "position": "NONE", "status": "MT5_NOT_CONNECTED", "open_positions": 0})
        return
    positions = mt5.positions_get(symbol=symbol) or []
    profit = sum(float(p.profit) for p in positions)
    side = "NONE"
    if positions:
        side = "BUY" if positions[0].type == mt5.POSITION_TYPE_BUY else "SELL"
    save_json(SHARED / "trade_status.json", {
        "symbol": symbol,
        "position": side,
        "open_positions": len(positions),
        "profit": profit,
        "drawdown_percent": 0,
        "status": "OPEN" if positions else "FLAT",
        "open_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


def spread_points(symbol: str) -> float:
    if mt5 is None:
        return 9999
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if not tick or not info:
        return 9999
    return abs(tick.ask - tick.bid) / info.point


def place_order(signal: dict) -> None:
    symbol = signal.get("symbol", "XAUUSD")
    action = signal.get("action")
    if mt5 is None or not mt5.initialize():
        log("MT5 chưa kết nối, bỏ qua lệnh.")
        return
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if not tick or not info:
        log(f"Không đọc được symbol {symbol}.")
        return
    order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
    price = tick.ask if action == "BUY" else tick.bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.01,
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": 260521,
        "comment": "SonFED",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    log(f"Gửi lệnh {action}: {result}")


def can_trade(signal: dict) -> tuple[bool, str]:
    cfg = CONFIG.get("trade", {})
    if signal.get("action") not in {"BUY", "SELL"}:
        return False, "Không phải tín hiệu BUY/SELL."
    if not signal.get("allow_auto_trade", False):
        return False, "Tín hiệu không cho phép tự động giao dịch."
    if signal.get("confidence", 0) < cfg.get("min_confidence", 70):
        return False, "Độ tin cậy thấp."
    if not signal_is_fresh(signal, cfg.get("signal_max_age_seconds", 300)):
        return False, "Tín hiệu quá cũ."
    if spread_points(signal.get("symbol", "XAUUSD")) > cfg.get("max_spread_points", 35):
        return False, "Spread cao."
    reasons = signal.get("risk_check", {}).get("reasons", [])
    if any("tin lớn" in str(r).lower() for r in reasons):
        return False, "Có tin lớn gần thời điểm hiện tại."
    return True, "Đủ điều kiện."


def main() -> None:
    log("MT5 Trade Bot khởi động.")
    while True:
        signal = load_json(SHARED / "signal.json", {})
        symbol = signal.get("symbol", CONFIG.get("trade", {}).get("symbol", "XAUUSD"))
        write_status(symbol)
        ok, reason = can_trade(signal)
        save_json(SHARED / "risk_status.json", {"spread_points": spread_points(symbol), "allow": ok, "reason": reason, "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        if ok:
            place_order(signal)
            time.sleep(30)
        else:
            log(f"Bỏ qua tín hiệu: {reason}")
            time.sleep(5)


if __name__ == "__main__":
    main()
