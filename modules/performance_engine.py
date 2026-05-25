from __future__ import annotations

from typing import Any

from .trade_statistics import BASE_CAPITAL, account_statistics, enrich_positions


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _money(value: Any) -> str:
    number = _safe_float(value)
    sign = "+" if number > 0 else ""
    return f"{sign}{number:.2f}$"


def _pct(value: Any) -> str:
    number = _safe_float(value)
    sign = "+" if number > 0 else ""
    return f"{sign}{number:.1f}%"


def build_account_report(trade_status: dict[str, Any], config: dict[str, Any]) -> str:
    account = trade_status.get("account", {})
    positions = trade_status.get("positions", [])
    stats = account_statistics(account, positions, _safe_float(config.get("telegram", {}).get("base_capital"), BASE_CAPITAL))
    if isinstance(trade_status.get("statistics"), dict):
        stats.update(trade_status["statistics"])
    profit_icon = "📈" if _safe_float(stats.get("profit_percent")) >= 0 else "📉"
    return "\n".join(
        [
            "========================",
            "📊 SONFED REPORT",
            "========================",
            "",
            f"💰 Balance: {_safe_float(stats.get('balance')):.2f}$",
            f"📈 Equity: {_safe_float(stats.get('equity')):.2f}$",
            f"🔥 Floating PnL: {_money(stats.get('floating_pnl'))}",
            "",
            f"🟢 BUY Orders: {int(_safe_float(stats.get('buy_orders')))}",
            f"🔴 SELL Orders: {int(_safe_float(stats.get('sell_orders')))}",
            f"📦 Total BUY Volume: {_safe_float(stats.get('buy_volume')):.2f}",
            f"📦 Total SELL Volume: {_safe_float(stats.get('sell_volume')):.2f}",
            "",
            f"🏆 Winrate: {_safe_float(stats.get('winrate')):.1f}%",
            f"⚖️ RR Avg: {_safe_float(stats.get('rr_avg')):.2f}",
            f"✅ Win: {int(_safe_float(stats.get('wins')))}",
            f"❌ Loss: {int(_safe_float(stats.get('losses')))}",
            "",
            f"📉 Drawdown: {_safe_float(stats.get('drawdown_percent')):.1f}%",
            f"📅 Today PnL: {_money(stats.get('today_pnl', 0))}",
            f"📅 Week PnL: {_money(stats.get('week_pnl', 0))}",
            f"{profit_icon} Profit on {_safe_float(config.get('telegram', {}).get('base_capital'), BASE_CAPITAL):.0f}$: {_pct(stats.get('profit_percent'))}",
            "",
            f"🤖 Bot Status: {'RUNNING' if trade_status else 'UNKNOWN'}",
            f"🔄 Auto Refresh: {'ON' if config.get('auto_refresh', {}).get('enabled') else 'OFF'}",
            f"⚡ Auto Trade: {'ON' if config.get('trade', {}).get('allow_auto_trade') else 'OFF'}",
            "",
            "========================",
        ]
    )


def build_orders_report(positions: list[dict[str, Any]]) -> str:
    rows = enrich_positions(positions)
    if not rows:
        return "📭 Không có lệnh đang mở."
    lines = ["📌 OPEN ORDERS", ""]
    for pos in rows:
        icon = "🟢" if pos["type"] == "BUY" else "🔴"
        lines.extend(
            [
                f"{icon} {pos['type']} {pos['symbol']}",
                f"Lot: {_safe_float(pos.get('lot')):.2f}",
                f"Entry: {_safe_float(pos.get('entry')):.2f}",
                f"Current: {_safe_float(pos.get('current')):.2f}",
                f"PnL: {_money(pos.get('profit'))}",
                f"SL: {_safe_float(pos.get('sl')):.2f}",
                f"TP: {_safe_float(pos.get('tp')):.2f}",
                f"Age: {pos['age_minutes']} phút",
                "",
            ]
        )
    return "\n".join(lines).strip()


def build_ai_change_alert(previous: str, current: str, reason: str) -> str:
    return "\n".join(["⚠️ AI CHANGED SIGNAL", f"{previous} → {current}", "", "Reason:", reason or "AI đổi bias theo dữ liệu mới."])
