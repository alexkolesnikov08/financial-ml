# -*- coding: utf-8 -*-
"""Тюнинг гиперпараметров строго внутри train/val через walk-forward.

Никакого доступа к hold-out. Одинаковые фолды для всех моделей.
"""

from __future__ import annotations

import itertools
import random
from typing import Any

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

import config
import models as M

random.seed(config.RANDOM_STATE)
np.random.seed(config.RANDOM_STATE)


def _product(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    out = []
    for combo in itertools.product(*vals):
        out.append(dict(zip(keys, combo)))
    return out


def _sample_combos(grid: dict, n: int) -> list[dict]:
    combos = _product(grid)
    if len(combos) <= n:
        return combos
    rng = random.Random(config.RANDOM_STATE)
    return rng.sample(combos, n)


def evaluate_params(factory, params, X, y, folds, metric: str = config.TUNE_METRIC,
                    prices=None):
    """Оценить params по фолдам. Возвращает среднюю метрику (выше = лучше)."""
    scores = []
    Xa = X.values if hasattr(X, "values") else np.asarray(X)
    ya = y.values if hasattr(y, "values") else np.asarray(y)
    pa = None if prices is None else (prices.values if hasattr(prices, "values") else np.asarray(prices))
    for tr, va in folds:
        est = factory(params if params else {})
        try:
            est.fit(Xa[tr], ya[tr])
            proba = est.predict_proba(Xa[va])[:, 1]
        except Exception as e:
            return -np.inf
        # защита от вырожденных proba
        proba = np.clip(proba, 1e-6, 1 - 1e-6)
        if metric == "neg_log_loss":
            s = -log_loss(ya[va], proba, labels=[0, 1])
        elif metric == "roc_auc":
            try:
                s = roc_auc_score(ya[va], proba)
            except Exception:
                s = 0.5
        elif metric == "bot_pnl":
            if pa is None:
                s = -log_loss(ya[va], proba, labels=[0, 1])
            else:
                import backtest as _BT
                try:
                    _r = _BT.simulate(pa[va], proba, 0.5)
                    s = _r["total_return"]
                except Exception:
                    s = -np.inf
        else:
            s = -log_loss(ya[va], proba, labels=[0, 1])
        scores.append(s)
    if not scores:
        return -np.inf
    return float(np.mean(scores))


def tune_model(name: str, spec: dict, X, y, folds, n_iter: int = config.N_TUNING_ITER,
               metric: str = config.TUNE_METRIC, prices=None) -> tuple[dict, float]:
    """Подобрать лучшие гиперпараметры для модели name. Возвращает (best_params, best_score)."""
    grid = spec.get("grid", {})
    factory = spec["factory"]
    combos = _sample_combos(grid, n_iter)
    best_params = {}
    best_score = -np.inf
    for params in combos:
        sc = evaluate_params(factory, params, X, y, folds, metric, prices=prices)
        if sc > best_score:
            best_score = sc
            best_params = dict(params)
    # Если все провалились, дефолт пустой.
    if best_score == -np.inf:
        best_params = {}
        best_score = evaluate_params(factory, {}, X, y, folds, metric, prices=prices)
    return best_params, best_score


def tune_all(X, y, folds, zoo: dict | None = None, n_iter: int = config.N_TUNING_ITER,
             metric: str = config.TUNE_METRIC, prices=None) -> dict:
    """Тюнить все модели в zoo. Возвращает dict name -> {'params':..., 'score':...}."""
    if zoo is None:
        zoo = M.make_zoo()
    result = {}
    for name, spec in zoo.items():
        try:
            bp, bs = tune_model(name, spec, X, y, folds, n_iter, metric, prices=prices)
            result[name] = {"params": bp, "score": bs}
            print(f"[tune] {name}: score={bs:.4f} params={bp}")
        except Exception as e:
            print(f"[tune] {name}: FAILED {e}")
            result[name] = {"params": {}, "score": -np.inf}
    return result