# -*- coding: utf-8 -*-
"""Генерация OOF-предсказаний по времени.

Один и тот же набор фолдов для всех базовых моделей. Строго out-of-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import calibrate as C


def _to_arrays(X, y):
    Xa = X.values if hasattr(X, "values") else np.asarray(X)
    ya = y.values if hasattr(y, "values") else np.asarray(y)
    return Xa, ya


def generate_oof(spec, params, X, y, folds, calibrate_method: str | None = None) -> np.ndarray:
    """Вернуть OOF-массив вероятностей (длина len(y)), NaN вне val-фолдов."""
    Xa, ya = _to_arrays(X, y)
    oof = np.full(len(ya), np.nan, dtype=float)
    factory = spec["factory"]
    for tr, va in folds:
        est_params = dict(params) if params else {}
        if calibrate_method is None:
            est = factory(est_params)
            est.fit(Xa[tr], ya[tr])
            proba = est.predict_proba(Xa[va])[:, 1]
        else:
            # Калибровка на хвосте train (временной порядок).
            cut = int(len(tr) * 0.8)
            tr_sub = tr[:cut]
            calib_sub = tr[cut:]
            if len(calib_sub) < 5:
                est = factory(est_params)
                est.fit(Xa[tr], ya[tr])
                proba = est.predict_proba(Xa[va])[:, 1]
            else:
                base = factory(est_params)
                cal = C.calibrate_on_split(base, Xa[tr_sub], ya[tr_sub],
                                           Xa[calib_sub], ya[calib_sub],
                                           method=calibrate_method)
                proba = cal.predict_proba(Xa[va])[:, 1]
        oof[va] = np.clip(proba, 1e-6, 1 - 1e-6)
    return oof


def generate_all_oof(zoo, tuned, X, y, folds, calibrate_method: str | None = "sigmoid") -> dict:
    """Сгенерировать OOF для всех моделей zoo. tuned: dict name->{'params':...}."""
    result = {}
    for name, spec in zoo.items():
        params = tuned.get(name, {}).get("params", {}) if tuned else {}
        try:
            oof = generate_oof(spec, params, X, y, folds, calibrate_method)
            result[name] = oof
            # покрытие
            cov = np.isfinite(oof).mean()
            print(f"[oof] {name}: coverage={cov:.3f}")
        except Exception as e:
            print(f"[oof] {name}: FAILED {e}")
    return result


def train_final(spec, params, X_train, y_train, calibrate_method: str | None = None):
    """Обучить финальную модель на train+val для предсказания hold-out."""
    Xa = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
    ya = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
    est_params = dict(params) if params else {}
    factory = spec["factory"]
    if calibrate_method is None:
        est = factory(est_params)
        est.fit(Xa, ya)
        return est
    # калибровка на хвосте
    cut = int(len(ya) * 0.85)
    base = factory(est_params)
    cal = C.calibrate_on_split(base, Xa[:cut], ya[:cut], Xa[cut:], ya[cut:],
                              method=calibrate_method)
    return cal