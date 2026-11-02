# -*- coding: utf-8 -*-
"""Симулятор long-only бота.

Правила (строго):
- signal_t -> вход на баре t+1.
- Только покупка (long-only).
- All-in при срабатывании сигнала.
- Издержки + слиппедж на каждую сторону оборота.
- Порог задаётся снаружи (подобран на валидации), здесь не меняется.
"""

from __future__ import annotations

import numpy as np

import config


def simulate(prices, signals, threshold: float) -> dict:
    """Симулировать long-only стратегию.

    prices: array-like цен закрытия (длина n).
    signals: array-like вероятностей/скоров (длина n) ИЛИ бинарных меток.
    threshold: порог для сигнала "вверх".

    Возвращает dict с equity_curve, trades, метриками.
    """
    prices = np.asarray(prices, dtype=float)
    signals = np.asarray(signals, dtype=float)
    n = len(prices)
    if len(signals) != n:
        raise ValueError("prices и signals должны быть одинаковой длины")

    cost_rate = config.TOTAL_COST_PER_SIDE
    cash = float(config.START_CAPITAL)
    shares = 0.0
    in_pos = False
    trades = []
    equity_curve = []
    positions = np.zeros(n, dtype=int)

    for i in range(n):
        price = prices[i]
        if not np.isfinite(price) or price <= 0:
            positions[i] = 1 if in_pos else 0
            mtm = cash + shares * price if np.isfinite(price) else cash
            equity_curve.append(mtm)
            continue

        desired = False
        if i > 0 and np.isfinite(signals[i - 1]):
            desired = bool(signals[i - 1] >= threshold)

        # Исполнение перехода на этом баре по цене close[i].
        if desired and not in_pos:
            invest = cash
            fee = invest * cost_rate
            shares = (invest - fee) / price if invest > 0 else 0.0
            cash = 0.0
            in_pos = True
            trades.append({
                "entry_i": i,
                "entry_price": float(price),
                "entry_cash": float(invest),
                "exit_i": None,
                "exit_price": None,
                "pnl": None,
            })
        elif (not desired) and in_pos:
            gross = shares * price
            fee = gross * cost_rate
            cash = gross - fee
            shares = 0.0
            in_pos = False
            if trades and trades[-1]["exit_i"] is None:
                trades[-1]["exit_i"] = i
                trades[-1]["exit_price"] = float(price)
                trades[-1]["pnl"] = float(cash - trades[-1]["entry_cash"])

        positions[i] = 1 if in_pos else 0
        mtm = cash + shares * price
        equity_curve.append(mtm)

    # Закрыть открытую позицию в конце (с издержками) для честного PnL.
    if in_pos and n > 0:
        price = prices[-1]
        if np.isfinite(price) and price > 0:
            gross = shares * price
            fee = gross * cost_rate
            cash = gross - fee
            shares = 0.0
            in_pos = False
            if trades and trades[-1]["exit_i"] is None:
                trades[-1]["exit_i"] = n - 1
                trades[-1]["exit_price"] = float(price)
                trades[-1]["pnl"] = float(cash - trades[-1]["entry_cash"])
            if equity_curve:
                equity_curve[-1] = cash

    equity_curve = np.asarray(equity_curve, dtype=float)
    final_equity = float(equity_curve[-1]) if n else float(config.START_CAPITAL)

    closed = [t for t in trades if t["exit_i"] is not None]
    pnls = [t["pnl"] for t in closed if t["pnl"] is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    winrate = (len(wins) / len(pnls)) if pnls else 0.0
    profit_factor = (sum(wins) / (abs(sum(losses)) + 1e-12)) if losses else (float("inf") if wins else 0.0)
    peak = np.maximum.accumulate(equity_curve) if n else np.array([config.START_CAPITAL])
    dd = (equity_curve - peak) / (peak + 1e-12)
    max_dd = float(dd.min()) if n else 0.0
    exposure = float(positions.mean()) if n else 0.0

    total_pnl = final_equity - config.START_CAPITAL
    total_ret = total_pnl / config.START_CAPITAL

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "final_equity": final_equity,
        "total_pnl": total_pnl,
        "total_return": total_ret,
        "num_trades": len(trades),
        "num_closed": len(closed),
        "winrate": winrate,
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else None,
        "max_drawdown": max_dd,
        "exposure": exposure,
        "avg_trade": float(np.mean(pnls)) if pnls else 0.0,
    }


def buy_and_hold(prices) -> dict:
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    if n < 2 or not np.isfinite(prices[0]) or prices[0] <= 0:
        return {"final_equity": float(config.START_CAPITAL), "total_return": 0.0,
                "equity_curve": np.array([config.START_CAPITAL]), "max_drawdown": 0.0}
    entry = prices[0]
    cost_rate = config.TOTAL_COST_PER_SIDE
    shares = (config.START_CAPITAL * (1 - cost_rate)) / entry
    equity = shares * prices  # mark-to-market (до выходных издержек)
    final = shares * prices[-1] * (1 - cost_rate)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-12)
    max_dd = float(dd.min()) if n else 0.0
    return {
        "final_equity": float(final),
        "total_return": float(final / config.START_CAPITAL - 1),
        "equity_curve": equity,
        "max_drawdown": max_dd,
    }


def cash_benchmark() -> dict:
    return {"final_equity": float(config.START_CAPITAL), "total_return": 0.0,
            "equity_curve": np.array([config.START_CAPITAL]), "max_drawdown": 0.0}