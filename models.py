# -*- coding: utf-8 -*-
"""Реестр классических ML-моделей и сетки гиперпараметров.

Каждая модель - фабрика (callable -> estimator) + сетка для тюнинга.
Все модели поддерживают predict_proba (через калибровку при необходимости).
"""

from __future__ import annotations

import warnings

from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import (LogisticRegression, RidgeClassifier,
                                  SGDClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Binarizer

import config

warnings.filterwarnings("ignore")

# Опциональные бустинги.
try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# TabPack (Yandex): ensemble of MLPs.
try:
    import os as _os
    _os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch as _torch  # noqa: F401
    from tabpackage.sklearn import TabPackClassifier as _TabPackClassifier
    HAS_TABPACK = True
except Exception:
    HAS_TABPACK = False
    _TabPackClassifier = None
try:
    import catboost as cb
    HAS_CAT = True
except Exception:
    HAS_CAT = False


def _wrap_calibrated(est, method="sigmoid", cv=3):
    # Временной CV вместо стратифицированного (time-aware калибровка).
    ts = TimeSeriesSplit(n_splits=cv)
    return CalibratedClassifierCV(estimator=est, method=method, cv=ts)


class TabPackWrapper:
    """Обёртка над TabPackClassifier: внутренний train/val split для extra_set."""

    def __init__(self, params=None):
        self.params = dict(params) if params else {}
        self.params.setdefault('n_models', 2)
        self.params.setdefault('max_epochs', 2)
        self.params.setdefault('show_progress_bar', False)
        self.params.setdefault('print_progress', False)
        self.model = None

    def fit(self, X, y):
        import numpy as np
        Xa = np.asarray(X, dtype=np.float32)
        ya = np.asarray(y)
        cut = max(5, int(len(ya) * 0.8))
        if cut >= len(ya):
            cut = len(ya) - 5
        Xtr, ytr = Xa[:cut], ya[:cut]
        Xval, yval = Xa[cut:], ya[cut:]
        if len(Xval) < 2:
            Xval, yval = Xa[:2], ya[:2]
        from tabpackage.sklearn import TabPackClassifier
        self.params.setdefault('device', 'mps')
        self.model = TabPackClassifier(**self.params, random_state=42)
        self.model.fit(Xtr, ytr, extra_set=(Xval, yval))
        return self

    def predict_proba(self, X):
        import numpy as np
        return self.model.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X):
        import numpy as np
        return self.model.predict(np.asarray(X, dtype=np.float32))


class SimpleMLPWrapper:
    """Лёгкий MLP на PyTorch: 2 hidden слоя, dropout, early stopping."""

    def __init__(self, params=None):
        self.params = dict(params) if params else {}
        self.params.setdefault("device", "mps")
        self.hidden = int(self.params.get("hidden", 64))
        self.dropout = float(self.params.get("dropout", 0.2))
        self.lr = float(self.params.get("lr", 1e-3))
        self.epochs = int(self.params.get("epochs", 30))
        self.batch = int(self.params.get("batch", 64))
        self.patience = int(self.params.get("patience", 5))
        self.activation = self.params.get("activation", "relu")
        self.n_layers = int(self.params.get("n_layers", 2))
        self.use_bn = bool(self.params.get("use_bn", False))
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None

    def fit(self, X, y):
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        import torch
        import torch.nn as nn
        import numpy as np
        Xa = np.asarray(X, dtype=np.float32)
        ya = np.asarray(y).astype(np.int64)
        # standardize
        self.scaler_mean = Xa.mean(axis=0)
        self.scaler_std = Xa.std(axis=0) + 1e-6
        Xn = (Xa - self.scaler_mean) / self.scaler_std
        cut = max(5, int(len(ya) * 0.8))
        if cut >= len(ya):
            cut = len(ya) - 5
        Xtr, ytr = Xn[:cut], ya[:cut]
        Xva, yva = Xn[cut:], ya[cut:]
        if len(Xva) < 2:
            Xva, yva = Xtr[:2], ytr[:2]

        in_dim = Xn.shape[1]
        n_classes = 2
        def _act():
            a = self.activation
            if a == "relu":
                return nn.ReLU()
            if a == "gelu":
                return nn.GELU()
            if a == "silu":
                return nn.SiLU()
            if a == "tanh":
                return nn.Tanh()
            if a == "leaky_relu":
                return nn.LeakyReLU(0.1)
            if a == "elu":
                return nn.ELU()
            return nn.ReLU()

        class Net(nn.Module):
            def __init__(self, in_dim, hidden, dropout, n_layers, use_bn, act_fn):
                super().__init__()
                layers = []
                layers.append(nn.Linear(in_dim, hidden))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden))
                layers.append(act_fn())
                layers.append(nn.Dropout(dropout))
                for _ in range(n_layers - 1):
                    layers.append(nn.Linear(hidden, hidden))
                    if use_bn:
                        layers.append(nn.BatchNorm1d(hidden))
                    layers.append(act_fn())
                    layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(hidden, n_classes))
                self.net = nn.Sequential(*layers)
            def forward(self, x):
                return self.net(x)

        torch.manual_seed(42)
        net = Net(in_dim, self.hidden, self.dropout, self.n_layers, self.use_bn, _act)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()
        Xtr_t = torch.from_numpy(Xtr)
        ytr_t = torch.from_numpy(ytr)
        Xva_t = torch.from_numpy(Xva)
        yva_t = torch.from_numpy(yva)

        best_val = float("inf")
        best_state = None
        bad = 0
        n = len(Xtr_t)
        for epoch in range(self.epochs):
            net.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch):
                idx = perm[i:i+self.batch]
                xb = Xtr_t[idx]
                yb = ytr_t[idx]
                opt.zero_grad()
                out = net(xb)
                loss = crit(out, yb)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val_out = net(Xva_t)
                val_loss = crit(val_out, yva_t).item()
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.model = net
        self.model.eval()
        return self

    def predict_proba(self, X):
        import torch, numpy as np
        Xa = np.asarray(X, dtype=np.float32)
        Xn = (Xa - self.scaler_mean) / self.scaler_std
        t = torch.from_numpy(Xn)
        self.model.eval()
        with torch.no_grad():
            out = self.model(t)
            proba = torch.softmax(out, dim=1).numpy()
        return proba

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def make_zoo() -> dict:
    """Вернуть dict: name -> {'factory': callable(params)->est, 'grid': dict}."""
    zoo = {}

    # --- Logistic Regression L1 ---
    zoo["logreg_l1"] = {
        "factory": lambda p: LogisticRegression(
            penalty="l1", solver="saga", C=p.get("C", 1.0),
            max_iter=2000, random_state=config.RANDOM_STATE, n_jobs=1),
        "grid": {"C": [0.01, 0.1, 0.5, 1.0, 5.0]},
    }
    # --- Logistic Regression L2 ---
    zoo["logreg_l2"] = {
        "factory": lambda p: LogisticRegression(
            penalty="l2", solver="lbfgs", C=p.get("C", 1.0),
            max_iter=2000, random_state=config.RANDOM_STATE),
        "grid": {"C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]},
    }
    # --- ElasticNet ---
    zoo["logreg_elastic"] = {
        "factory": lambda p: LogisticRegression(
            penalty="elasticnet", solver="saga",
            C=p.get("C", 1.0), l1_ratio=p.get("l1_ratio", 0.5),
            max_iter=2000, random_state=config.RANDOM_STATE, n_jobs=1),
        "grid": {"C": [0.1, 0.5, 1.0, 5.0],
                 "l1_ratio": [0.2, 0.5, 0.8]},
    }
    # --- SGD log_loss ---
    zoo["sgd_logloss"] = {
        "factory": lambda p: SGDClassifier(
            loss="log_loss", penalty=p.get("penalty", "l2"),
            alpha=p.get("alpha", 1e-4), max_iter=2000, tol=1e-3,
            random_state=config.RANDOM_STATE, n_jobs=1),
        "grid": {"penalty": ["l2", "l1", "elasticnet"],
                 "alpha": [1e-5, 1e-4, 1e-3, 1e-2]},
    }
    # --- RidgeClassifier + calibration ---
    zoo["ridge_calibrated"] = {
        "factory": lambda p: _wrap_calibrated(
            RidgeClassifier(alpha=p.get("alpha", 1.0)),
            method=p.get("cal_method", "sigmoid"), cv=3),
        "grid": {"alpha": [0.1, 1.0, 10.0, 50.0],
                 "cal_method": ["sigmoid", "isotonic"]},
    }
    # --- HistGradientBoosting ---
    zoo["histgb"] = {
        "factory": lambda p: HistGradientBoostingClassifier(
            max_iter=p.get("max_iter", 200), learning_rate=p.get("learning_rate", 0.1),
            max_leaf_nodes=p.get("max_leaf_nodes", 31),
            min_samples_leaf=p.get("min_samples_leaf", 20),
            l2_regularization=p.get("l2_regularization", 0.0),
            random_state=config.RANDOM_STATE),
        "grid": {"max_iter": [100, 200, 400],
                 "learning_rate": [0.05, 0.1],
                 "max_leaf_nodes": [15, 31],
                 "min_samples_leaf": [20, 50]},
    }
    # --- RandomForest ---
    zoo["rf"] = {
        "factory": lambda p: RandomForestClassifier(
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", None),
            min_samples_leaf=p.get("min_samples_leaf", 5),
            max_features=p.get("max_features", "sqrt"),
            random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS),
        "grid": {"n_estimators": [200, 400],
                 "max_depth": [None, 8, 16],
                 "min_samples_leaf": [5, 20]},
    }
    # --- ExtraTrees ---
    zoo["et"] = {
        "factory": lambda p: ExtraTreesClassifier(
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", None),
            min_samples_leaf=p.get("min_samples_leaf", 5),
            random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS),
        "grid": {"n_estimators": [200, 400],
                 "max_depth": [None, 8, 16],
                 "min_samples_leaf": [5, 20]},
    }

    if HAS_LGBM:
        zoo["lgbm"] = {
            "factory": lambda p: lgb.LGBMClassifier(
                n_estimators=p.get("n_estimators", 300),
                learning_rate=p.get("learning_rate", 0.05),
                num_leaves=p.get("num_leaves", 31),
                min_child_samples=p.get("min_child_samples", 20),
                subsample=p.get("subsample", 0.8),
                colsample_bytree=p.get("colsample_bytree", 0.8),
                reg_alpha=p.get("reg_alpha", 0.0),
                reg_lambda=p.get("reg_lambda", 1.0),
                random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS, verbose=-1),
            "grid": {"n_estimators": [200, 400],
                     "learning_rate": [0.03, 0.05, 0.1],
                     "num_leaves": [15, 31, 63],
                     "min_child_samples": [20, 50]},
        }
    if HAS_XGB:
        zoo["xgb"] = {
            "factory": lambda p: xgb.XGBClassifier(
                n_estimators=p.get("n_estimators", 300),
                learning_rate=p.get("learning_rate", 0.05),
                max_depth=p.get("max_depth", 6),
                subsample=p.get("subsample", 0.8),
                colsample_bytree=p.get("colsample_bytree", 0.8),
                reg_alpha=p.get("reg_alpha", 0.0),
                reg_lambda=p.get("reg_lambda", 1.0),
                random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS,
                eval_metric="logloss", tree_method="hist"),
            "grid": {"n_estimators": [200, 400],
                     "learning_rate": [0.03, 0.05, 0.1],
                     "max_depth": [4, 6, 8],
                     "subsample": [0.8, 1.0]},
        }
    if HAS_CAT:
        zoo["catboost"] = {
            "factory": lambda p: cb.CatBoostClassifier(
                iterations=p.get("iterations", 300),
                learning_rate=p.get("learning_rate", 0.05),
                depth=p.get("depth", 6),
                l2_leaf_reg=p.get("l2_leaf_reg", 3.0),
                random_seed=config.RANDOM_STATE, verbose=0, thread_count=-1),
            "grid": {"iterations": [200, 400],
                     "learning_rate": [0.03, 0.05, 0.1],
                     "depth": [4, 6, 8],
                     "l2_leaf_reg": [1.0, 3.0, 10.0]},
        }

    # --- GaussianNB ---
    zoo["gaussian_nb"] = {
        "factory": lambda p: GaussianNB(var_smoothing=p.get("var_smoothing", 1e-9)),
        "grid": {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
    }
    # --- BernoulliNB (через Binarizer) ---
    zoo["bernoulli_nb"] = {
        "factory": lambda p: Pipeline([
            ("binarizer", Binarizer(threshold=p.get("bin_threshold", 0.0))),
            ("nb", BernoulliNB(alpha=p.get("alpha", 1.0))),
        ]),
        "grid": {"alpha": [0.1, 0.5, 1.0, 5.0, 10.0]},
    }
    # --- LinearDiscriminantAnalysis ---
    zoo["lda"] = {
        "factory": lambda p: LinearDiscriminantAnalysis(
            solver=p.get("solver", "svd"),
            shrinkage=p.get("shrinkage", None)),
        "grid": {"solver": ["svd", "lsqr"]},
    }
    # --- AdaBoost (base DecisionTree depth=2) ---
    zoo["ada_boost"] = {
        "factory": lambda p: AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2,
                                             random_state=config.RANDOM_STATE),
            n_estimators=p.get("n_estimators", 100),
            learning_rate=p.get("learning_rate", 1.0),
            random_state=config.RANDOM_STATE),
        "grid": {"n_estimators": [50, 100, 200],
                 "learning_rate": [0.5, 1.0]},
    }
    # --- KNN (через StandardScaler) ---
    zoo["knn"] = {
        "factory": lambda p: Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=p.get("n_neighbors", 5),
                weights=p.get("weights", "uniform"))),
        ]),
        "grid": {"n_neighbors": [3, 5, 7, 11],
                 "weights": ["uniform", "distance"]},
    }
    # --- Dummy baseline ---
    zoo["dummy"] = {
        "factory": lambda p: DummyClassifier(
            strategy=p.get("strategy", "stratified"),
            random_state=config.RANDOM_STATE),
        "grid": {"strategy": ["stratified", "most_frequent", "prior"]},
    }

    # MLP variants: activations
    for act in ["relu", "gelu", "silu", "tanh", "leaky_relu", "elu"]:
        zoo[f"mlp_{act}"] = {"factory": lambda p, a=act: SimpleMLPWrapper({**p, "activation": a}),
                             "grid": {"hidden": [64], "dropout": [0.2], "lr": [1e-3]}}

    # MLP variants: depth
    for nl in [1, 2, 3, 4]:
        zoo[f"mlp_depth{nl}"] = {"factory": lambda p, n=nl: SimpleMLPWrapper({**p, "n_layers": n}),
                                 "grid": {"hidden": [64], "dropout": [0.2], "lr": [1e-3]}}

    # MLP variants: width
    for hd in [32, 64, 128, 256]:
        zoo[f"mlp_w{hd}"] = {"factory": lambda p, h=hd: SimpleMLPWrapper({**p, "hidden": h}),
                             "grid": {"dropout": [0.2], "lr": [1e-3]}}

    # MLP variants: dropout
    for dr in [0.0, 0.1, 0.2, 0.3, 0.5]:
        zoo[f"mlp_do{int(dr*10)}"] = {"factory": lambda p, d=dr: SimpleMLPWrapper({**p, "dropout": d}),
                                      "grid": {"hidden": [64], "lr": [1e-3]}}

    # MLP variants: batch norm
    zoo["mlp_bn"] = {"factory": lambda p: SimpleMLPWrapper({**p, "use_bn": True}),
                     "grid": {"hidden": [64], "dropout": [0.2], "lr": [1e-3]}}

    return zoo


def meta_zoo() -> dict:
    """Мета-модели для стекинга (простые)."""
    mz = {}
    mz["meta_logreg_l2"] = {
        "factory": lambda p: LogisticRegression(
            penalty="l2", C=p.get("C", 1.0), max_iter=2000,
            random_state=config.RANDOM_STATE),
        "grid": {"C": [0.01, 0.1, 1.0, 10.0]},
    }
    mz["meta_logreg_l1"] = {
        "factory": lambda p: LogisticRegression(
            penalty="l1", solver="saga", C=p.get("C", 1.0), max_iter=2000,
            random_state=config.RANDOM_STATE, n_jobs=1),
        "grid": {"C": [0.01, 0.1, 1.0, 10.0]},
    }
    mz["meta_sgd"] = {
        "factory": lambda p: SGDClassifier(
            loss="log_loss", alpha=p.get("alpha", 1e-3), max_iter=2000,
            random_state=config.RANDOM_STATE, n_jobs=1),
        "grid": {"alpha": [1e-4, 1e-3, 1e-2]},
    }
    mz["meta_ridge_cal"] = {
        "factory": lambda p: _wrap_calibrated(
            RidgeClassifier(alpha=p.get("alpha", 1.0)),
            method="sigmoid", cv=3),
        "grid": {"alpha": [0.1, 1.0, 10.0]},
    }
    mz["meta_histgb"] = {
        "factory": lambda p: HistGradientBoostingClassifier(
            max_iter=p.get("max_iter", 100), learning_rate=0.05,
            max_leaf_nodes=15, random_state=config.RANDOM_STATE),
        "grid": {"max_iter": [50, 100]},
    }
    if HAS_LGBM:
        mz["meta_lgbm"] = {
            "factory": lambda p: lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.05, num_leaves=15,
                random_state=config.RANDOM_STATE, verbose=-1),
            "grid": {},
        }
    mz["meta_rf"] = {
            "factory": lambda p: RandomForestClassifier(
                n_estimators=200, max_depth=4, min_samples_leaf=20,
                random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS),
            "grid": {},
        }
    mz["meta_et"] = {
        "factory": lambda p: ExtraTreesClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=20,
            random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS),
        "grid": {},
    }
    mz["meta_gaussian_nb"] = {
        "factory": lambda p: GaussianNB(var_smoothing=p.get("var_smoothing", 1e-9)),
        "grid": {"var_smoothing": [1e-9, 1e-7]},
    }
    mz["meta_lda"] = {
        "factory": lambda p: LinearDiscriminantAnalysis(
            solver=p.get("solver", "svd")),
        "grid": {"solver": ["svd", "lsqr"]},
    }
    return mz