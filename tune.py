# -*- coding: utf-8 -*-
"""Тюнинг гиперпараметров строго внутри train/val через walk-forward.

Никакого доступа к hold-out. Одинаковые фолды для всех моделей.
"""

from __future__ import annotations

import itertools
import os
import pickle
import random
from typing import Any

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

import config
import models as M

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    _HAS_OPTUNA = True
except Exception:  # pragma: no cover
    optuna = None
    TPESampler = None
    MedianPruner = None
    _HAS_OPTUNA = False

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


def _fold_score(factory, params, Xa, ya, pa, tr, va, metric: str) -> float:
    """Оценка одного фолда. Возвращает метрику (выше = лучше)."""
    est = factory(params if params else {})
    try:
        est.fit(Xa[tr], ya[tr])
        proba = est.predict_proba(Xa[va])[:, 1]
    except Exception:
        return -np.inf
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
    return float(s)


def evaluate_params(factory, params, X, y, folds, metric: str = config.TUNE_METRIC,
                    prices=None):
    """Оценить params по фолдам. Возвращает среднюю метрику (выше = лучше)."""
    Xa = X.values if hasattr(X, "values") else np.asarray(X)
    ya = y.values if hasattr(y, "values") else np.asarray(y)
    pa = None if prices is None else (prices.values if hasattr(prices, "values") else np.asarray(prices))
    scores = []
    for tr, va in folds:
        sc = _fold_score(factory, params, Xa, ya, pa, tr, va, metric)
        if not np.isfinite(sc):
            return -np.inf
        scores.append(sc)
    if not scores:
        return -np.inf
    return float(np.mean(scores))


def _suggest_param(trial, name, values):
    """Сэмплировать гиперпараметр из сетки spec'а через Optuna."""
    vals = list(values)
    vals = [v.item() if hasattr(v, "item") and not isinstance(v, (str, bool)) else v for v in vals]

    # Булевы.
    if all(isinstance(v, bool) for v in vals):
        return trial.suggest_categorical(name, vals)
    # None в списке.
    if any(v is None for v in vals):
        return trial.suggest_categorical(name, vals)
    # Строки.
    if all(isinstance(v, str) for v in vals):
        return trial.suggest_categorical(name, vals)
    # Целые.
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in vals):
        uniq = sorted(set(vals))
        if len(uniq) >= 2:
            diffs = np.diff(uniq)
            if np.all(diffs == diffs[0]):
                step = int(diffs[0])
                if step >= 1:
                    return trial.suggest_int(name, int(uniq[0]), int(uniq[-1]), step=step)
        return trial.suggest_categorical(name, vals)
    # Вещественные.
    if all(isinstance(v, (float, np.floating)) for v in vals):
        uniq = sorted(set(vals))
        if len(uniq) >= 2:
            diffs = np.diff(uniq)
            if np.allclose(diffs, diffs[0]):
                step = float(diffs[0])
                if step > 0:
                    return trial.suggest_float(name, float(uniq[0]), float(uniq[-1]), step=step)
        return trial.suggest_categorical(name, vals)
    # Смешанные/прочие.
    return trial.suggest_categorical(name, vals)


def _make_pruner(name):
    if name is None:
        return None
    name = str(name).lower()
    if name in ("none", "off", "false", ""):
        return None
    if name == "median":
        if not _HAS_OPTUNA:
            return None
        return MedianPruner()
    # По умолчанию median.
    if _HAS_OPTUNA:
        return MedianPruner()
    return None


def tune_model_optuna(name: str, spec: dict, X, y, folds, n_trials, timeout,
                      metric: str = config.TUNE_METRIC, prices=None,
                      pruner=None, study_path=None) -> tuple[dict, float]:
    """Optuna-тюнинг (TPE + MedianPruner) с walk-forward и pruning по фолдам."""
    if not _HAS_OPTUNA:
        raise RuntimeError("optuna не установлен")

    grid = spec.get("grid", {}) or {}
    factory = spec["factory"]

    Xa = X.values if hasattr(X, "values") else np.asarray(X)
    ya = y.values if hasattr(y, "values") else np.asarray(y)
    pa = None if prices is None else (prices.values if hasattr(prices, "values") else np.asarray(prices))

    pruner_name = pruner if pruner is not None else getattr(config, "OPTUNA_PRUNER", "median")
    _pruner = _make_pruner(pruner_name)

    def objective(trial):
        params = {}
        for key, values in grid.items():
            params[key] = _suggest_param(trial, key, values)

        scores = []
        for fold_idx, (tr, va) in enumerate(folds):
            sc = _fold_score(factory, params, Xa, ya, pa, tr, va, metric)
            if not np.isfinite(sc):
                # Сообщаем и пруним/завершаем.
                trial.report(-np.inf, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                return -np.inf
            scores.append(sc)
            running = float(np.mean(scores))
            trial.report(running, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if not scores:
            return -np.inf
        return float(np.mean(scores))

    sampler = TPESampler(seed=config.RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=_pruner)
    nt = n_trials if n_trials is not None else getattr(config, "OPTUNA_N_TRIALS", 20)
    to = timeout if timeout is not None else getattr(config, "OPTUNA_TIMEOUT", None)

    study.optimize(objective, n_trials=int(nt) if nt is not None else 1,
                  timeout=to, show_progress_bar=False)

    try:
        if study.best_trial is not None:
            best_params = dict(study.best_params)
            best_score = float(study.best_value)
        else:
            best_params = {}
            best_score = -np.inf
            best_score = evaluate_params(factory, {}, X, y, folds, metric, prices=prices)
    except Exception:
        best_params = {}
        best_score = -np.inf
        best_score = evaluate_params(factory, {}, X, y, folds, metric, prices=prices)

    # Если все провалились.
    if best_score == -np.inf:
        best_params = {}
        best_score = evaluate_params(factory, {}, X, y, folds, metric, prices=prices)

    if study_path is not None:
        try:
            path = str(study_path)
            is_dir = os.path.isdir(path) or path.endswith(("/", "\\"))
            if not is_dir and not os.path.exists(path):
                if os.path.splitext(path)[1] == "":
                    is_dir = True
            if is_dir:
                os.makedirs(path, exist_ok=True)
                path = os.path.join(path, f"{name}_study.pkl")
            else:
                parent = os.path.dirname(path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(study, f)
        except Exception as e:
            print(f"[optuna] save study failed: {e}")

    return best_params, best_score


def tune_model(name: str, spec: dict, X, y, folds, n_iter: int = config.N_TUNING_ITER,
               metric: str = config.TUNE_METRIC, prices=None,
               use_optuna=None, n_trials=None, timeout=None, pruner=None,
               study_path=None) -> tuple[dict, float]:
    """Подобрать лучшие гиперпараметры для модели name. Возвращает (best_params, best_score)."""
    if use_optuna is None:
        use_optuna = getattr(config, "USE_OPTUNA", False)

    if use_optuna and _HAS_OPTUNA:
        nt = n_trials if n_trials is not None else getattr(config, "OPTUNA_N_TRIALS", None)
        to = timeout if timeout is not None else getattr(config, "OPTUNA_TIMEOUT", None)
        return tune_model_optuna(name, spec, X, y, folds, nt if nt is not None else n_iter,
                                 to, metric, prices=prices, pruner=pruner,
                                 study_path=study_path)

    # Старый random search.
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
    if best_score == -np.inf:
        best_params = {}
        best_score = evaluate_params(factory, {}, X, y, folds, metric, prices=prices)
    return best_params, best_score


def tune_all(X, y, folds, zoo: dict | None = None, n_iter: int = config.N_TUNING_ITER,
             metric: str = config.TUNE_METRIC, prices=None,
             use_optuna=None, n_trials=None, timeout=None, pruner=None,
             study_path=None) -> dict:
    """Тюнить все модели в zoo. Возвращает dict name -> {'params':..., 'score':...}."""
    if zoo is None:
        zoo = M.make_zoo()
    result = {}
    for name, spec in zoo.items():
        try:
            bp, bs = tune_model(name, spec, X, y, folds, n_iter=n_iter, metric=metric,
                                 prices=prices, use_optuna=use_optuna,
                                 n_trials=n_trials, timeout=timeout, pruner=pruner,
                                 study_path=study_path)
            result[name] = {"params": bp, "score": bs}
            print(f"[tune] {name}: score={bs:.4f} params={bp}")
        except Exception as e:
            print(f"[tune] {name}: FAILED {e}")
            result[name] = {"params": {}, "score": -np.inf}
    return result