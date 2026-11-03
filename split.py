# -*- coding: utf-8 -*-
"""Временные сплиты: walk-forward фолды и multi-holdout.

Никакого shuffle. Все индексы - позиции в отсортированном по времени DataFrame.
"""

from __future__ import annotations

from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd

import config


Fold = Tuple[np.ndarray, np.ndarray]  # (train_idx, val_idx)


def time_folds(n: int, n_folds: int = config.N_FOLDS,
               val_fraction: float = config.VAL_FRACTION,
               mode: str = config.CV_MODE,
               min_train_fraction: float = config.MIN_TRAIN_FRACTION) -> List[Fold]:
    """Сгенерировать список (train, val) индексов для walk-forward CV.

    Валидационные блоки покрывают хвост [min_train, n) без пропусков,
    поэтому последний бар попадает в валидацию. mode: expanding/rolling.
    """
    if n < n_folds + 2:
        raise ValueError(f"Слишком мало строк для {n_folds} фолдов: {n}")
    min_train = max(1, int(n * min_train_fraction))
    remaining = n - min_train
    if remaining < n_folds:
        raise ValueError("Слишком мало данных для валидации")
    # Размеры валидационных блоков (сумма = remaining).
    base = remaining // n_folds
    rem = remaining % n_folds
    sizes = [base + (1 if i < rem else 0) for i in range(n_folds)]

    folds: List[Fold] = []
    val_start = min_train
    for i in range(n_folds):
        val_end = val_start + sizes[i]
        if mode == "expanding":
            train_start = 0
        else:  # rolling: фиксированное окно размером min_train
            train_start = max(0, val_start - min_train)
        train_idx = np.arange(train_start, val_start)
        val_idx = np.arange(val_start, val_end)
        if len(train_idx) < 5 or len(val_idx) < 1:
            val_start = val_end
            continue
        folds.append((train_idx, val_idx))
        val_start = val_end
    if not folds:
        raise RuntimeError("Не удалось построить ни одного фолда")
    return folds


def _normalize_holdout_sizes(holdout_days: Any, n_holdouts: int) -> List[int]:
    """Привести holdout_days (int или последовательность) к списку длины n_holdouts."""
    if isinstance(holdout_days, (list, tuple, set, np.ndarray, pd.Series)):
        sizes = [int(x) for x in holdout_days]
    else:
        sizes = [int(holdout_days)]
    if n_holdouts <= 0:
        raise ValueError("n_holdouts должен быть > 0")
    if len(sizes) == 1 and n_holdouts > 1:
        sizes = sizes * n_holdouts
    if len(sizes) < n_holdouts:
        sizes = sizes + [sizes[-1]] * (n_holdouts - len(sizes))
    if len(sizes) > n_holdouts:
        sizes = sizes[:n_holdouts]
    if any(s <= 0 for s in sizes):
        raise ValueError("Размеры hold-out должны быть положительными")
    return sizes


def make_holdouts(df: pd.DataFrame, n_holdouts: int | None = None,
                  holdout_days: Any = config.HOLDOUT_DAYS) -> list[dict]:
    """Сделать список untouched hold-out периодов с конца истории.

    Поддерживает 4-6 (и более) hold-out и разные размеры в днях:
    ``holdout_days`` может быть int или последовательностью int (по одному
    размеру на hold-out). Окна идут с конца истории назад, без перекрытий.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index должен быть DatetimeIndex")
    if isinstance(holdout_days, (list, tuple, set, np.ndarray, pd.Series)) and n_holdouts is None:
        n_holdouts = len(holdout_days)
    if n_holdouts is None:
        n_holdouts = config.N_HOLDOUTS
    if n_holdouts < 1:
        raise ValueError("n_holdouts должен быть >= 1")
    sizes = _normalize_holdout_sizes(holdout_days, n_holdouts)

    idx = df.index
    end = idx.max()
    holdouts = []
    cumulative_days = 0
    for i in range(n_holdouts):
        size = sizes[i]
        h_end = end - pd.Timedelta(days=cumulative_days)
        h_start = h_end - pd.Timedelta(days=size)
        if i == 0:
            holdout_mask = idx >= h_start
        else:
            holdout_mask = (idx >= h_start) & (idx < h_end)
        train_mask = idx < h_start
        cumulative_days += size
        name = f"holdout_{i}_last_{cumulative_days - size + 1}-{cumulative_days}d"
        holdouts.append({
            "name": name,
            "holdout_start": h_start,
            "holdout_end": end if i == 0 else h_end,
            "holdout_days": int(size),
            "train_mask": train_mask.values if hasattr(train_mask, "values") else np.asarray(train_mask),
            "holdout_mask": holdout_mask.values if hasattr(holdout_mask, "values") else np.asarray(holdout_mask),
        })
    return holdouts


def _resolve_holdout_mask(df: pd.DataFrame, holdout: Any) -> np.ndarray:
    """Извлечь булеву маску hold-out из dict/массива."""
    if isinstance(holdout, dict):
        for key in ("holdout_mask", "mask", "test_mask", "val_mask"):
            if key in holdout:
                return np.asarray(holdout[key], dtype=bool).ravel()
        # Индексы строк hold-out.
        for key in ("holdout_idx", "indices", "idx"):
            if key in holdout:
                pos = np.asarray(holdout[key], dtype=int).ravel()
                m = np.zeros(len(df), dtype=bool)
                m[pos] = True
                return m
        raise ValueError("В holdout нет holdout_mask/indices")
    arr = np.asarray(holdout)
    if arr.dtype == bool or arr.dtype == np.bool_:
        return arr.ravel()
    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
        m = np.zeros(len(df), dtype=bool)
        m[arr] = True
        return m
    raise TypeError("holdout должен быть dict или маской/индексами")


def regime_split(df: pd.DataFrame, holdout: Any) -> dict[str, np.ndarray]:
    """Разбить hold-out период на режимы: trend / breakout / sideways.

    Режимы определяются по EMA50 (тренд) и волатильности (ATR/доходности).
    Возвращает маски, выровненные по строкам hold-out (длина = число баров
    hold-out), а не по всему df.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        # Допускаем не-DatetimeIndex, но лучше предупредить молча.
        pass
    mask = _resolve_holdout_mask(df, holdout)
    if mask.shape[0] != len(df):
        raise ValueError("Длина маски hold-out не совпадает с df")
    positions = np.where(mask)[0]
    sub = df.iloc[positions].copy()

    if len(sub) == 0:
        empty = np.zeros(0, dtype=bool)
        return {"trend": empty.copy(), "breakout": empty.copy(), "sideways": empty.copy()}

    close = pd.Series(sub["close"]).astype(float).reset_index(drop=True)
    if "ema_50" in sub.columns:
        ema = pd.Series(sub["ema_50"]).astype(float).reset_index(drop=True)
    else:
        ema = close.ewm(span=50, adjust=False, min_periods=1).mean()
    ema = ema.fillna(close)

    trend_strength = ((close - ema).abs() / ema.replace(0, np.nan)).fillna(0.0)

    if "atr_pct" in sub.columns:
        vol = pd.Series(sub["atr_pct"]).astype(float).reset_index(drop=True)
    elif "atr_14" in sub.columns:
        vol = (pd.Series(sub["atr_14"]).astype(float).reset_index(drop=True)
               / close.replace(0, np.nan))
    else:
        vol = close.pct_change().rolling(10, min_periods=1).std().fillna(0.0)
    vol = vol.fillna(0.0)

    vol_values = np.asarray(vol, dtype=float)
    if vol_values.size:
        try:
            thr_vol = float(np.nanquantile(vol_values, 0.75))
        except Exception:
            med = float(np.nanmedian(vol_values))
            thr_vol = med * 1.5 if med > 0 else 0.0
        if not np.isfinite(thr_vol) or thr_vol <= 0:
            med = float(np.nanmedian(vol_values))
            thr_vol = med * 1.5 if med > 0 else 0.0
    else:
        thr_vol = 0.0
    breakout = (vol_values > thr_vol) if vol_values.size else np.zeros(0, dtype=bool)

    trend = (~breakout) & (np.asarray(trend_strength, dtype=float) > 0.002)
    sideways = ~(breakout | trend)

    return {
        "trend": np.asarray(trend, dtype=bool),
        "breakout": np.asarray(breakout, dtype=bool),
        "sideways": np.asarray(sideways, dtype=bool),
    }