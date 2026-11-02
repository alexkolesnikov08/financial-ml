# -*- coding: utf-8 -*-
"""Временные сплиты: walk-forward фолды и multi-holdout.

Никакого shuffle. Все индексы - позиции в отсортированном по времени DataFrame.
"""

from __future__ import annotations

from typing import List, Tuple

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


def make_holdouts(df: pd.DataFrame, n_holdouts: int = config.N_HOLDOUTS,
                  holdout_days: int = config.HOLDOUT_DAYS) -> list[dict]:
    """Сделать список untouched hold-out периодов с конца истории."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index должен быть DatetimeIndex")
    idx = df.index
    end = idx.max()
    holdouts = []
    for i in range(n_holdouts):
        h_end = end - pd.Timedelta(days=i * holdout_days)
        h_start = h_end - pd.Timedelta(days=holdout_days)
        if i == 0:
            holdout_mask = idx >= h_start
        else:
            holdout_mask = (idx >= h_start) & (idx < h_end)
        train_mask = idx < h_start
        name = f"holdout_{i}_last_{i * holdout_days + 1}-{(i + 1) * holdout_days}d"
        holdouts.append({
            "name": name,
            "holdout_start": h_start,
            "holdout_end": end if i == 0 else h_end,
            "train_mask": train_mask.values if hasattr(train_mask, "values") else np.asarray(train_mask),
            "holdout_mask": holdout_mask.values if hasattr(holdout_mask, "values") else np.asarray(holdout_mask),
        })
    return holdouts