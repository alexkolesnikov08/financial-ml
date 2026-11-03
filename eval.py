# -*- coding: utf-8 -*-
"""Оценка бота и бенчмарков. Сводные таблицы и сохранение артефактов."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

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


def classification_metrics(y_true: Any, y_pred: Any) -> dict:
    """Классификационные метрики на hold-out: accuracy/precision/recall/F1.

    Ожидается бинарная задача с положительным классом 1.
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.shape != yp.shape:
        raise ValueError("y_true и y_pred должны иметь одинаковую длину")
    if yt.size == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    accuracy = float(np.mean(yt == yp))

    tp = float(np.sum((yt == 1) & (yp == 1)))
    fp = float(np.sum((yt == 0) & (yp == 1)))
    fn = float(np.sum((yt == 1) & (yp == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _resolve_holdout_arrays(result: dict, y_true: Any, y_pred: Any):
    """Достать y_true/y_pred из аргументов или result."""
    if y_true is None:
        for key in ("y_true", "y_hold", "y", "y_actual", "actual"):
            if isinstance(result, dict) and key in result:
                y_true = result[key]
                break
    if y_pred is None:
        for key in ("y_pred", "pred", "predictions", "yhat", "proba"):
            if isinstance(result, dict) and key in result:
                y_pred = result[key]
                break
    return y_true, y_pred


def summarize_holdout(result: dict, name: str, y_true: Any = None,
                     y_pred: Any = None, **kwargs) -> dict:
    out = {
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

    # Поддержка передачи через kwargs.
    if y_true is None:
        y_true = kwargs.get("y_true", kwargs.get("y_hold", kwargs.get("y")))
    if y_pred is None:
        y_pred = kwargs.get("y_pred", kwargs.get("pred", kwargs.get("predictions")))
    y_true, y_pred = _resolve_holdout_arrays(result, y_true, y_pred)

    if y_true is not None and y_pred is not None:
        try:
            out.update(classification_metrics(y_true, y_pred))
        except Exception:
            pass
    return out


def regime_metrics(y_true: Any, y_pred: Any, regimes: Any = None, **kwargs) -> dict:
    """Оценка classification_metrics по режимам рынка.

    ``regimes`` может быть:
      - dict name -> bool-маска (длина n);
      - массив строк-меток режима длины n;
      - массив/список масок.
    Дополнительно ищет маски в kwargs (regime_masks/masks).
    Возвращает dict regime -> metrics (с полем n).
    """
    if regimes is None:
        for key in ("regimes", "regime_masks", "masks", "regime"):
            if key in kwargs:
                regimes = kwargs[key]
                break

    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    n = yt.shape[0]
    if yp.shape[0] != n:
        m = min(n, yp.shape[0])
        yt = yt[:m]
        yp = yp[:m]
        n = m

    out: dict = {}

    if isinstance(regimes, dict):
        for name, mask in regimes.items():
            m = np.asarray(mask, dtype=bool).ravel()
            if m.shape[0] != n:
                # Если маска короче/длиннее - попробовать выровнять.
                if m.shape[0] < n:
                    mm = np.zeros(n, dtype=bool)
                    mm[:m.shape[0]] = m
                    m = mm
                else:
                    m = m[:n]
            sub_yt = yt[m]
            sub_yp = yp[m]
            metrics = classification_metrics(sub_yt, sub_yp) if sub_yt.size else {
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            }
            metrics = dict(metrics)
            metrics["n"] = int(m.sum())
            out[str(name)] = metrics
        return out

    arr = np.asarray(regimes) if regimes is not None else None
    if arr is None:
        return out

    # Массив строк-меток.
    if arr.ndim == 1 and arr.dtype.kind in ("U", "S", "O"):
        labels = [str(x) for x in arr.tolist()]
        if len(labels) != n:
            labels = labels[:n]
        for label in sorted(set(labels)):
            m = np.asarray([l == label for l in labels], dtype=bool)
            sub_yt = yt[m]
            sub_yp = yp[m]
            metrics = classification_metrics(sub_yt, sub_yp) if sub_yt.size else {
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            }
            metrics = dict(metrics)
            metrics["n"] = int(m.sum())
            out[label] = metrics
        return out

    # 2D: каждая строка/столбец - маска.
    if arr.ndim == 2:
        # Если строк столько же, сколько режимов? Предполагаем столбцы - маски.
        if arr.shape[0] == n:
            for j in range(arr.shape[1]):
                m = arr[:, j].astype(bool).ravel()
                name = f"regime_{j}"
                sub_yt = yt[m]
                sub_yp = yp[m]
                metrics = classification_metrics(sub_yt, sub_yp) if sub_yt.size else {
                    "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                }
                metrics = dict(metrics)
                metrics["n"] = int(m.sum())
                out[name] = metrics
            return out
        if arr.shape[1] == n:
            for j in range(arr.shape[0]):
                m = arr[j].astype(bool).ravel()
                name = f"regime_{j}"
                sub_yt = yt[m]
                sub_yp = yp[m]
                metrics = classification_metrics(sub_yt, sub_yp) if sub_yt.size else {
                    "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                }
                metrics = dict(metrics)
                metrics["n"] = int(m.sum())
                out[name] = metrics
            return out

    # 1D bool-маска -> один режим.
    if arr.ndim == 1 and (arr.dtype == bool or arr.dtype == np.bool_):
        m = arr.astype(bool).ravel()[:n]
        if m.shape[0] < n:
            mm = np.zeros(n, dtype=bool)
            mm[:m.shape[0]] = m
            m = mm
        sub_yt = yt[m]
        sub_yp = yp[m]
        metrics = classification_metrics(sub_yt, sub_yp) if sub_yt.size else {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }
        metrics = dict(metrics)
        metrics["n"] = int(m.sum())
        out["regime"] = metrics
        return out

    # Список масок (list of arrays).
    if isinstance(regimes, (list, tuple)):
        for j, m in enumerate(regimes):
            mm = np.asarray(m, dtype=bool).ravel()
            if mm.shape[0] != n:
                if mm.shape[0] < n:
                    tmp = np.zeros(n, dtype=bool)
                    tmp[:mm.shape[0]] = mm
                    mm = tmp
                else:
                    mm = mm[:n]
            sub_yt = yt[mm]
            sub_yp = yp[mm]
            metrics = classification_metrics(sub_yt, sub_yp) if sub_yt.size else {
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            }
            metrics = dict(metrics)
            metrics["n"] = int(mm.sum())
            out[f"regime_{j}"] = metrics
        return out

    return out


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