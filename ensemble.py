# -*- coding: utf-8 -*-
"""Ансамбли: все варианты голосования + стекинг с несколькими мета-моделями.

Вероятности базовых моделей должны быть калиброваны. Стекинг обучается
ТОЛЬКО на OOF-матрице (никакого доступа к hold-out при обучении меты).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import config
import models as M


def _stack(probas: dict) -> np.ndarray:
    names = list(probas.keys())
    return np.column_stack([np.asarray(probas[n], dtype=float) for n in names])


def soft_mean(probas: dict) -> np.ndarray:
    M_ = _stack(probas)
    return M_.mean(axis=1)


def weighted_soft(probas: dict, weights: dict | None = None) -> np.ndarray:
    names = list(probas.keys())
    M_ = _stack(probas)
    if weights is None:
        w = np.ones(len(names))
    else:
        w = np.array([weights.get(n, 1.0) for n in names], dtype=float)
    w = w / (w.sum() + 1e-12)
    return M_ @ w


def geom_mean(probas: dict) -> np.ndarray:
    M_ = _stack(probas)
    M_ = np.clip(M_, 1e-6, 1 - 1e-6)
    logm = np.log(M_).mean(axis=1)
    return np.exp(logm)


def rank_avg(probas: dict) -> np.ndarray:
    M_ = _stack(probas)
    ranks = np.full_like(M_, np.nan)
    for j in range(M_.shape[1]):
        col = M_[:, j]
        finite = np.isfinite(col)
        if finite.sum() == 0:
            continue
        r = rankdata(col[finite]) / (finite.sum() + 1)
        ranks[finite, j] = r
    return ranks.mean(axis=1)


def hard_vote(probas: dict) -> np.ndarray:
    M_ = _stack(probas)
    # NaN -> NaN (не считаем как класс 0)
    out = np.full(M_.shape[0], np.nan, dtype=float)
    finite_rows = np.all(np.isfinite(M_), axis=1)
    if finite_rows.any():
        bins = (M_[finite_rows] >= 0.5).astype(int)
        out[finite_rows] = bins.mean(axis=1)
    return out


VOTING_METHODS = {
    "soft_mean": soft_mean,
    "weighted_soft": weighted_soft,
    "geom_mean": geom_mean,
    "rank_avg": rank_avg,
    "hard_vote": hard_vote,
}


def make_weights_from_oof(oof_dict: dict, y) -> dict:
    """Веса из OOF: обратно пропорциональны logloss."""
    from sklearn.metrics import log_loss
    ya = y.values if hasattr(y, "values") else np.asarray(y)
    w = {}
    for name, oof in oof_dict.items():
        mask = np.isfinite(oof)
        if mask.sum() < 10:
            w[name] = 1.0
            continue
        try:
            ll = log_loss(ya[mask], np.clip(oof[mask], 1e-6, 1 - 1e-6))
            w[name] = 1.0 / (ll + 1e-6)
        except Exception:
            w[name] = 1.0
    return w


# ---------- Stacking ----------

def build_meta_matrix(oof_dict: dict) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Построить матрицу мета-признаков из OOF. Возвращает (X_meta, names, mask_finite)."""
    names = list(oof_dict.keys())
    M_ = _stack(oof_dict)
    mask = np.all(np.isfinite(M_), axis=1)
    return M_[mask], names, mask


def train_stacker(meta_name: str, meta_spec: dict, oof_dict: dict, y,
                  folds=None, params=None, tune: bool = False) -> dict:
    """Обучить мета-модель на OOF-матрице. Возвращает dict с fitted estimator и именами базовых."""
    Xmeta, names, mask = build_meta_matrix(oof_dict)
    ya = (y.values if hasattr(y, "values") else np.asarray(y))[mask]
    factory = meta_spec["factory"]
    p = dict(params) if params else {}
    if tune and meta_spec.get("grid"):
        # Тюнинг меты на временных фолдах OOF-строк (позиции внутри mask).
        from split import time_folds
        from tune import evaluate_params
        n = len(ya)
        mfolds = time_folds(n, n_folds=min(4, max(2, n // 50)), val_fraction=0.25, mode="expanding")
        # evaluate_params ожидает X,y,folds в виде arrays; передаём Xmeta,ya
        import types
        # адаптируем: evaluate_params использует X.values; передаём DataFrame? сделаем ndarray через обёртку.
        # Проще: перебор вручную.
        from itertools import product
        grid = meta_spec.get("grid", {})
        keys = list(grid.keys())
        combos = [dict(zip(keys, c)) for c in product(*[grid[k] for k in keys])] if keys else [{}]
        cap = config.N_TUNING_ITER if len(combos) > config.N_TUNING_ITER else len(combos)
        best_p, best_s = {}, -np.inf
        for combo in combos[:cap]:
            # оценка по фолдам
            scs = []
            for tr, va in mfolds:
                est = factory({**p, **combo})
                try:
                    est.fit(Xmeta[tr], ya[tr])
                    proba = est.predict_proba(Xmeta[va])[:, 1]
                    from sklearn.metrics import log_loss
                    scs.append(-log_loss(ya[va], np.clip(proba, 1e-6, 1 - 1e-6)))
                except Exception:
                    scs.append(-np.inf)
            sc = float(np.mean(scs)) if scs else -np.inf
            if sc > best_s:
                best_s = sc
                best_p = combo
        p = {**p, **best_p}
    est = factory(p)
    est.fit(Xmeta, ya)
    return {"name": meta_name, "estimator": est, "base_names": names, "params": p}



def cv_predict_stacker(meta_spec: dict, params: dict, oof_dict: dict, y,
                      folds) -> np.ndarray:
    """Out-of-fold предсказания меты на OOF-матрице (для порога без утечки)."""
    Xmeta_full, names, mask = build_meta_matrix(oof_dict)
    ya_full = (y.values if hasattr(y, "values") else np.asarray(y))
    finite_idx = np.where(mask)[0]
    n_full = len(ya_full)
    oof_pred = np.full(n_full, np.nan, dtype=float)
    Xmeta = Xmeta_full
    ya = ya_full[mask]
    factory = meta_spec["factory"]
    pp = dict(params) if params else {}
    for tr, va in folds:
        est = factory(pp)
        try:
            est.fit(Xmeta[tr], ya[tr])
            if hasattr(est, "predict_proba"):
                proba = est.predict_proba(Xmeta[va])[:, 1]
            else:
                df_ = est.decision_function(Xmeta[va])
                proba = 1 / (1 + np.exp(-df_))
            oof_pred[finite_idx[va]] = np.clip(proba, 1e-6, 1 - 1e-6)
        except Exception:
            continue
    return oof_pred

def diversity_prune(oof_dict: dict, corr_threshold: float = 0.95) -> list[str]:
    """Удалить модели с корреляцией OOF > corr_threshold. Возвращает список имён-к-оставлению."""
    names = list(oof_dict.keys())
    if len(names) <= 1:
        return list(names)
    common = np.all(np.column_stack([np.isfinite(oof_dict[n]) for n in names]), axis=1)
    if int(common.sum()) < 10:
        return list(names)
    M_ = np.column_stack([oof_dict[n][common] for n in names])
    try:
        corr = np.corrcoef(M_, rowvar=False)
    except Exception:
        return list(names)
    if np.ndim(corr) == 0:
        return list(names)
    keep = []
    keep_idx = []
    for i, n in enumerate(names):
        ok = True
        for j in keep_idx:
            if abs(corr[i, j]) > corr_threshold:
                ok = False
                break
        if ok:
            keep.append(n)
            keep_idx.append(i)
    if not keep:
        keep = [names[0]]
    return keep


def predict_stacker(stacker: dict, probas_holdout: dict) -> np.ndarray:
    """Предсказать вероятности на hold-out по базовым proba."""
    names = stacker["base_names"]
    M_ = np.column_stack([np.asarray(probas_holdout[n], dtype=float) for n in names])
    est = stacker["estimator"]
    if hasattr(est, "predict_proba"):
        return est.predict_proba(M_)[:, 1]
    # decision_function fallback
    df = est.decision_function(M_)
    return 1 / (1 + np.exp(-df))