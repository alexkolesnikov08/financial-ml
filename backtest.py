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
import pandas as pd

import config


def simulate(prices, signals, threshold: float = 0.5, thr_lo=None, time_mask=None,
             regime_mask=None, atr=None, trend_mask=None, min_hold=0, cooldown=0,
             trail_mult=None, consensus_signals=None, min_consensus=None,
             edge_mu_win=None, edge_mu_loss=None) -> dict:
    """Симулировать long-only стратегию.

    prices: array-like цен закрытия (длина n).
    signals: array-like вероятностей/скоров (длина n) ИЛИ бинарных меток.
    threshold: порог для сигнала "вверх" (thr_hi).
    Опциональные параметры (None -> текущее поведение):
      thr_lo, time_mask, regime_mask, atr, trend_mask, min_hold, cooldown,
      trail_mult, consensus_signals, min_consensus, edge_mu_win, edge_mu_loss.
    """
    prices = np.asarray(prices, dtype=float)
    signals = np.asarray(signals, dtype=float)
    n = len(prices)
    if len(signals) != n:
        raise ValueError("prices и signals должны быть одинаковой длины")

    time_mask = None if time_mask is None else np.asarray(time_mask, dtype=bool)
    regime_mask = None if regime_mask is None else np.asarray(regime_mask, dtype=bool)
    trend_mask = None if trend_mask is None else np.asarray(trend_mask, dtype=bool)
    atr = None if atr is None else np.asarray(atr, dtype=float)
    consensus = None if consensus_signals is None else np.asarray(consensus_signals)

    # Проверки длин масок/ATR/consensus.
    if time_mask is not None and time_mask.shape[0] != n:
        raise ValueError("time_mask должна иметь длину n")
    if regime_mask is not None and regime_mask.shape[0] != n:
        raise ValueError("regime_mask должна иметь длину n")
    if trend_mask is not None and trend_mask.shape[0] != n:
        raise ValueError("trend_mask должна иметь длину n")
    if atr is not None and atr.shape[0] != n:
        raise ValueError("atr должна иметь длину n")
    if consensus is not None:
        if consensus.ndim == 2:
            if consensus.shape[0] != n:
                raise ValueError("consensus_signals (2D) должна иметь n строк")
        else:
            if consensus.shape[0] != n:
                raise ValueError("consensus_signals (1D) должна иметь длину n")

    cost_rate = config.TOTAL_COST_PER_SIDE
    cash = float(config.START_CAPITAL)
    shares = 0.0
    in_pos = False
    trades = []
    equity_curve = []
    positions = np.zeros(n, dtype=int)

    last_exit_i = None
    entry_i = None
    peak_price = None
    last_valid_price = None
    prev_equity = float(config.START_CAPITAL)

    lo_thr = threshold if thr_lo is None else float(thr_lo)

    for i in range(n):
        price = prices[i]
        if np.isfinite(price) and price > 0:
            last_valid_price = price
        if not np.isfinite(price) or price <= 0:
            positions[i] = 1 if in_pos else 0
            if in_pos:
                if last_valid_price is not None and np.isfinite(last_valid_price):
                    mtm = cash + shares * last_valid_price
                elif equity_curve:
                    mtm = equity_curve[-1]
                else:
                    mtm = cash
            else:
                mtm = cash
            equity_curve.append(mtm)
            prev_equity = mtm
            continue

        if in_pos:
            if peak_price is None or price > peak_price:
                peak_price = price

        # Решение на основе signal[i-1] (signal_t -> вход t+1).
        sig = signals[i - 1] if (i > 0 and i - 1 < len(signals)) else np.nan
        if i > 0 and np.isfinite(sig):
            if in_pos:
                desired = bool(sig >= lo_thr)
            else:
                desired = bool(sig >= threshold)
        else:
            desired = False

        # min_hold: не выходим раньше срока.
        if in_pos and (not desired) and min_hold and entry_i is not None and (i - entry_i) < int(min_hold):
            desired = True

        # cooldown: не входим сразу после выхода.
        if (not in_pos) and desired and cooldown and last_exit_i is not None and (i - last_exit_i) < int(cooldown):
            desired = False

        # Маски и фильтры для ВХОДА.
        if (not in_pos) and desired:
            allowed = True
            if time_mask is not None and i < len(time_mask):
                allowed = allowed and bool(time_mask[i])
            if regime_mask is not None and i < len(regime_mask):
                allowed = allowed and bool(regime_mask[i])
            if trend_mask is not None and i < len(trend_mask):
                allowed = allowed and bool(trend_mask[i])
            if consensus is not None and min_consensus is not None:
                if i > 0 and i - 1 < consensus.shape[0]:
                    idx = i - 1
                else:
                    idx = min(i, consensus.shape[0] - 1)
                if consensus.ndim == 2:
                    row = consensus[idx]
                else:
                    row = consensus[idx]
                cnt = int(np.sum(np.asarray(row) >= threshold))
                allowed = allowed and (cnt >= int(min_consensus))
            if edge_mu_win is not None and edge_mu_loss is not None and np.isfinite(sig):
                edge = float(edge_mu_win) * float(sig) - float(edge_mu_loss) * (1.0 - float(sig))
                allowed = allowed and (edge > 0.0)
            if not allowed:
                desired = False

        # Trailing stop.
        if (in_pos and trail_mult is not None and atr is not None
                and i < len(atr) and np.isfinite(atr[i]) and float(atr[i]) > 0.0
                and float(trail_mult) > 0.0):
            if peak_price is not None and price <= peak_price - float(trail_mult) * float(atr[i]):
                desired = False

        # Regime forced exit.
        if in_pos and regime_mask is not None and i < len(regime_mask) and not bool(regime_mask[i]):
            desired = False

        # Исполнение перехода.
        if desired and not in_pos:
            invest = cash
            fee = invest * cost_rate
            shares = (invest - fee) / price if invest > 0 else 0.0
            cash = 0.0
            in_pos = True
            entry_i = i
            peak_price = price
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
            last_exit_i = i
            if trades and trades[-1]["exit_i"] is None:
                trades[-1]["exit_i"] = i
                trades[-1]["exit_price"] = float(price)
                trades[-1]["pnl"] = float(cash - trades[-1]["entry_cash"])

        positions[i] = 1 if in_pos else 0
        mtm = cash + shares * price
        equity_curve.append(mtm)

    # Закрыть открытую позицию в конце.
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
            # positions учитывает закрытие на последнем баре
            if n > 0:
                positions[n - 1] = 0

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


def make_time_mask(prices_index_or_df, no_trade_first_min: int = 30,
                   no_trade_last_min: int = 30, open_hour: int = 9,
                   open_minute: int = 30, close_hour: int = 16,
                   close_minute: int = 0, tz: str = config.TZ) -> np.ndarray:
    """Маска разрешённых баров для входа (исключаем открытие/закрытие)."""
    if hasattr(prices_index_or_df, "index"):
        idx = prices_index_or_df.index
    else:
        idx = prices_index_or_df
    idx = pd.Index(idx)
    try:
        local = idx.tz_convert(tz) if idx.tz is not None else idx
    except Exception:
        local = idx
    open_min = open_hour * 60 + open_minute
    close_min = close_hour * 60 + close_minute
    mins = local.hour * 60 + local.minute if hasattr(local, "hour") else np.array([0]*len(idx))
    since_open = mins - open_min
    to_close = close_min - mins
    mask = (since_open >= no_trade_first_min) & (to_close >= no_trade_last_min)
    return np.asarray(mask, dtype=bool)