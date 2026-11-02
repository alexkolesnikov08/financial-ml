# -*- coding: utf-8 -*-
"""Оценка бота и бенчмарков. Сводные таблицы и сохранение артефактов."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import config


def _clean_value(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, np.ndarray):
        return [_clean_value(x) for x in v.tolist()]
    return v


def summarize_holdout(result: dict, name: str) -> dict:
    return {
        "name": name,
        "final_equity": result["final_equity"],
        "total_return": result["total_return"],
        "num_trades": result["num_trades"],
        "num_closed": result["num_closed"],
        "winrate": result["winrate"],
        "profit_factor": result["profit_factor"],
        "max_drawdown": result["max_drawdown"],
        "exposure": result["exposure"],
        "avg_trade": result["avg_trade"],
    }


def aggregate_holdouts(per_holdout: list[dict]) -> pd.DataFrame:
    """Усреднить final_equity по нескольким hold-out периодам (по summary)."""
    rows = []
    names = list(per_holdout[0]["summary"]["name"]) if per_holdout else []
    for name in names:
        eqs = []
        for r in per_holdout:
            sub = r["summary"][r["summary"]["name"] == name]
            if not sub.empty:
                eqs.append(float(sub["final_equity"].iloc[0]))
        if eqs:
            rows.append({
                "name": name,
                "final_equity_mean": float(np.mean(eqs)),
                "final_equity_std": float(np.std(eqs)),
                "final_equity_min": float(np.min(eqs)),
                "n_holdouts": len(eqs),
            })
    return pd.DataFrame(rows).sort_values("final_equity_mean", ascending=False) if rows else pd.DataFrame()


def save_results(results: dict, path: Path | None = None) -> Path:
    if path is None:
        path = config.REPORTS_DIR / "results.json"
    with open(path, "w") as f:
        json.dump(results, f, default=_clean_value, indent=2, allow_nan=False)
    return path


def to_dataframe(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)