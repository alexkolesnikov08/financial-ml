# -*- coding: utf-8 -*-
"""Калибровка вероятностей.

Ручная калибровка на отдельном калибровочном куске (строго позже по времени),
чтобы избежать утечки и не зависеть от версии sklearn.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class CalibratedWrapper:
    """Обёртка: base (уже обученный) + калибратор на calib-выборке."""

    def __init__(self, base, method: str = "sigmoid"):
        self.base = base
        self.method = method
        self.calibrator = None
        self.classes_ = np.array([0, 1])

    def _base_score(self, X):
        if hasattr(self.base, "predict_proba"):
            return self.base.predict_proba(X)[:, 1]
        if hasattr(self.base, "decision_function"):
            s = self.base.decision_function(X)
            # масштабируем в [0,1] через сигмоиду для изотоники
            return 1.0 / (1.0 + np.exp(-s))
        pred = self.base.predict(X)
        return pred.astype(float)

    def fit(self, X_calib, y_calib):
        scores = self._base_score(X_calib)
        y = np.asarray(y_calib).astype(int)
        # Guard: одноклассовая калибровочная выборка — калибровка невозможна.
        if len(np.unique(y)) < 2:
            self.calibrator = ("none", None)
            return self
        if self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(scores, y)
            self.calibrator = ("iso", iso)
        elif self.method == "temperature":
            from scipy.optimize import minimize_scalar
            from sklearn.metrics import log_loss as _ll
            def _nll(T):
                T = float(max(T, 1e-3))
                scaled = 1.0 / (1.0 + np.exp(-np.log(np.clip(scores, 1e-6, 1 - 1e-6)
                                                     / np.clip(1 - scores, 1e-6, 1 - 1e-6)) / T))
                try:
                    return float(_ll(y, np.clip(scaled, 1e-6, 1 - 1e-6), labels=[0, 1]))
                except Exception:
                    return 1e6
            res = minimize_scalar(_nll, bounds=(0.1, 10.0), method="bounded",
                                  options={"xatol": 1e-4})
            T = float(res.x) if np.isfinite(res.x) else 1.0
            self.calibrator = ("temp", T)
        elif self.method == "beta":
            eps = 1e-6
            logp = np.log(np.clip(scores, eps, 1 - eps))
            log1p = np.log(np.clip(1.0 - scores, eps, 1 - eps))
            Xf = np.column_stack([logp, log1p])
            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            lr.fit(Xf, y)
            self.calibrator = ("beta", lr)
        else:
            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            lr.fit(scores.reshape(-1, 1), y)
            self.calibrator = ("sig", lr)
        return self

    def predict_proba(self, X):
        if self.calibrator is None:
            raise RuntimeError("Калибратор не обучен: вызовите fit()")
        scores = self._base_score(X)
        kind = self.calibrator[0]
        if kind == "none":
            p = np.asarray(scores, dtype=float)
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return np.column_stack([1 - p, p])
        if kind == "iso":
            p = self.calibrator[1].transform(scores)
        elif kind == "temp":
            T = float(self.calibrator[1])
            logit = np.log(np.clip(scores, 1e-6, 1 - 1e-6) / np.clip(1 - scores, 1e-6, 1 - 1e-6))
            p = 1.0 / (1.0 + np.exp(-logit / T))
        elif kind == "beta":
            eps = 1e-6
            logp = np.log(np.clip(scores, eps, 1 - eps))
            log1p = np.log(np.clip(1.0 - scores, eps, 1 - eps))
            Xf = np.column_stack([logp, log1p])
            p = self.calibrator[1].predict_proba(Xf)[:, 1]
        else:
            p = self.calibrator[1].predict_proba(scores.reshape(-1, 1))[:, 1]
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_calibrated(estimator, method: str = "sigmoid", cv: int = 3):
    """Совместимость со старым API: возвращает CalibratedClassifierCV (cv int)."""
    from sklearn.calibration import CalibratedClassifierCV
    return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)


def calibrate_on_split(base_est, X_train, y_train, X_calib, y_calib, method="sigmoid"):
    """Обучить base на train, затем калибровать на calib (time-ordered)."""
    base_est.fit(X_train, y_train)
    wrapper = CalibratedWrapper(base_est, method=method)
    wrapper.fit(X_calib, y_calib)
    return wrapper


def platt_scale(proba, y):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(np.asarray(proba).reshape(-1, 1), y)
    return lr