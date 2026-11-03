# -*- coding: utf-8 -*-
"""Сохранение/загрузка артефактов обученных моделей (joblib)."""

from __future__ import annotations

from pathlib import Path

import joblib

import config


def save_models(holdout_name: str, final_base=None, stackers=None, oof=None,
                thresholds=None, train_prices=None, feature_columns=None,
                tuned=None, extra=None) -> Path:
    d = config.ARTIFACTS_DIR / holdout_name
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "holdout": holdout_name,
        "final_base": final_base or {},
        "stackers": stackers or {},
        "oof": oof or {},
        "thresholds": thresholds or {},
        "train_prices": train_prices,
        "feature_columns": feature_columns,
        "tuned": tuned,
        "extra": extra or {},
    }
    path = d / "models.joblib"
    joblib.dump(payload, path)
    return path


def load_models(holdout_name: str):
    path = config.ARTIFACTS_DIR / holdout_name / "models.joblib"
    if not path.exists():
        return None
    return joblib.load(path)