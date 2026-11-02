# -*- coding: utf-8 -*-
"""Оркестрация всего пайплайна: tuning -> OOF -> ансамбли -> hold-out симуляция.

Multi-holdout: каждый hold-out прогоняется по одному и тому же протоколу,
результаты усредняются.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import split as SP
import models as M
import tune as T
import cv_oof as CO
import ensemble as ENS
import backtest as BT
import eval as EV
from features import split_xy


def _optimize_threshold(prices, signals, grid=config.THRESHOLD_GRID):
    prices = np.asarray(prices, dtype=float)
    signals = np.asarray(signals, dtype=float)
    best_thr = 0.5
    best_eq = -np.inf
    for thr in grid:
        try:
            res = BT.simulate(prices, signals, float(thr))
        except Exception:
            continue
        if res["final_equity"] > best_eq:
            best_eq = res["final_equity"]
            best_thr = float(thr)
    return best_thr, float(best_eq)


def run_one_holdout(holdout: dict, df_feat: pd.DataFrame) -> dict:
    """Прогнать один hold-out период. Возвращает dict с метриками всех моделей/ансамблей."""
    train_mask = holdout["train_mask"]
    holdout_mask = holdout["holdout_mask"]
    train_df = df_feat.iloc[np.where(train_mask)[0]].copy()
    hold_df = df_feat.iloc[np.where(holdout_mask)[0]].copy()

    train_prices = train_df["close"].values
    hold_prices = hold_df["close"].values

    Xtr, ytr, cols = split_xy(train_df)
    Xho, yho, _ = split_xy(hold_df)

    folds = SP.time_folds(len(Xtr), n_folds=config.N_FOLDS,
                          val_fraction=config.VAL_FRACTION, mode=config.CV_MODE,
                          min_train_fraction=config.MIN_TRAIN_FRACTION)

    zoo = M.make_zoo()
    meta_zoo = M.meta_zoo()

    # 1. Тюнинг базовых моделей (prices передаём для bot_pnl-метрики, если включена).
    tuned = T.tune_all(Xtr, ytr, folds, zoo=zoo, n_iter=config.N_TUNING_ITER,
                       metric=config.TUNE_METRIC, prices=train_prices)

    # 2. OOF (калиброванные) для базовых.
    oof_dict = CO.generate_all_oof(zoo, tuned, Xtr, ytr, folds, calibrate_method="sigmoid")

    # 3. Пороги для базовых моделей на OOF (полные массивы, NaN -> нет сигнала).
    base_thresholds = {}
    for name, oof in oof_dict.items():
        thr, _ = _optimize_threshold(train_prices, oof)
        base_thresholds[name] = thr

    # 4. Финальные базовые модели на всём train.
    final_base = {}
    for name, spec in zoo.items():
        params = tuned.get(name, {}).get("params", {})
        model = CO.train_final(spec, params, Xtr, ytr, calibrate_method="sigmoid")
        final_base[name] = model

    # 5. Предсказания базовых на hold-out.
    Xho_a = Xho.values
    base_hold_proba = {}
    for name, model in final_base.items():
        try:
            base_hold_proba[name] = model.predict_proba(Xho_a)[:, 1]
        except Exception:
            df_ = model.decision_function(Xho_a)
            base_hold_proba[name] = 1 / (1 + np.exp(-df_))

    # 6. Веса для weighted soft из OOF.
    weights = ENS.make_weights_from_oof(oof_dict, ytr)

    # 7. Ансамбли на OOF -> пороги (на полных массивах).
    ens_oof_probas = {}
    all_finite_mask = (np.all(np.column_stack([np.isfinite(o) for o in oof_dict.values()]), axis=1)
                       if oof_dict else np.zeros(len(ytr), dtype=bool))

    if oof_dict and all_finite_mask.sum() >= 20:
        # voting variants на полных массивах (NaN сохраняются)
        ens_oof_probas["ens_soft_mean"] = ENS.soft_mean(oof_dict)
        ens_oof_probas["ens_weighted_soft"] = ENS.weighted_soft(oof_dict, weights)
        ens_oof_probas["ens_geom_mean"] = ENS.geom_mean(oof_dict)
        ens_oof_probas["ens_rank_avg"] = ENS.rank_avg(oof_dict)
        ens_oof_probas["ens_hard_vote"] = ENS.hard_vote(oof_dict)

        # stacking: мета обучается на finite-строках; OOF-предсказания меты - на полных.
        sub_oof_full = {n: oof_dict[n] for n in oof_dict}
        from split import time_folds as _tf
        meta_folds = _tf(int(all_finite_mask.sum()), n_folds=min(4, max(2, int(all_finite_mask.sum()) // 50)),
                         val_fraction=0.25, mode="expanding", min_train_fraction=0.3)
        # meta_folds индексы относительно finite-строк; cv_predict_stacker ожидает folds по finite-индексам.
        for mname, mspec in meta_zoo.items():
            try:
                stacker_tmp = ENS.train_stacker(mname, mspec, {k: v[all_finite_mask] for k, v in oof_dict.items()},
                                                pd.Series((ytr.values if hasattr(ytr, "values") else np.asarray(ytr))[all_finite_mask]),
                                                tune=False)
                finite_oof = {k: v[all_finite_mask] for k, v in oof_dict.items()}
                y_finite = pd.Series((ytr.values if hasattr(ytr, "values") else np.asarray(ytr))[all_finite_mask])
                meta_oof_finite = ENS.cv_predict_stacker(mspec, stacker_tmp["params"], finite_oof, y_finite, meta_folds)
                meta_oof = np.full(len(ytr), np.nan, dtype=float)
                finite_idx = np.where(all_finite_mask)[0]
                meta_oof[finite_idx] = np.asarray(meta_oof_finite, dtype=float).ravel()[:len(finite_idx)]
                ens_oof_probas[f"stack_{mname}"] = meta_oof
            except Exception as e:
                print(f"[stack] {mname}: FAILED {e}")

    ens_thresholds = {}
    for name, proba in ens_oof_probas.items():
        thr, _ = _optimize_threshold(train_prices, proba)
        ens_thresholds[name] = thr

    # 8. Ансамбли на hold-out.
    ens_hold_probas = {}
    ens_hold_probas["ens_soft_mean"] = ENS.soft_mean(base_hold_proba)
    ens_hold_probas["ens_weighted_soft"] = ENS.weighted_soft(base_hold_proba, weights)
    ens_hold_probas["ens_geom_mean"] = ENS.geom_mean(base_hold_proba)
    ens_hold_probas["ens_rank_avg"] = ENS.rank_avg(base_hold_proba)
    ens_hold_probas["ens_hard_vote"] = ENS.hard_vote(base_hold_proba)

    # stackers для hold-out (обучаем на finite OOF)
    stackers = {}
    if oof_dict and all_finite_mask.sum() >= 20:
        finite_oof = {k: v[all_finite_mask] for k, v in oof_dict.items()}
        y_finite = pd.Series((ytr.values if hasattr(ytr, "values") else np.asarray(ytr))[all_finite_mask])
        for mname, mspec in meta_zoo.items():
            try:
                stacker = ENS.train_stacker(mname, mspec, finite_oof, y_finite, tune=False)
                stackers[mname] = stacker
            except Exception as e:
                print(f"[stack] {mname}: FAILED {e}")

    for mname, stacker in stackers.items():
        try:
            ens_hold_probas[f"stack_{mname}"] = ENS.predict_stacker(stacker, base_hold_proba)
        except Exception as e:
            print(f"[stack hold] {mname}: FAILED {e}")

    # 9. Симуляция для базовых.
    results = {}
    base_rows = []
    for name in zoo.keys():
        proba = base_hold_proba[name]
        thr = base_thresholds.get(name, 0.5)
        sim = BT.simulate(hold_prices, proba, thr)
        results[name] = sim
        base_rows.append(EV.summarize_holdout(sim, name))

    # Ансамбли.
    ens_rows = []
    for name, proba in ens_hold_probas.items():
        thr = ens_thresholds.get(name, 0.5)
        sim = BT.simulate(hold_prices, proba, thr)
        results[name] = sim
        ens_rows.append(EV.summarize_holdout(sim, name))

    # Бенчмарки.
    bh = BT.buy_and_hold(hold_prices)
    cash = BT.cash_benchmark()
    benchmarks = {
        "buy_and_hold": bh["final_equity"],
        "cash": cash["final_equity"],
    }

    # Сводная таблица.
    all_rows = base_rows + ens_rows
    summary_df = EV.to_dataframe(all_rows)
    summary_df = pd.concat([summary_df, pd.DataFrame([
        {"name": "buy_and_hold", "final_equity": bh["final_equity"],
         "total_return": bh["total_return"], "num_trades": 1, "num_closed": 1,
         "winrate": 1.0, "profit_factor": bh.get("profit_factor", None),
         "max_drawdown": bh["max_drawdown"], "exposure": 1.0,
         "avg_trade": bh["final_equity"] - config.START_CAPITAL},
        {"name": "cash", "final_equity": cash["final_equity"],
         "total_return": 0.0, "num_trades": 0, "num_closed": 0,
         "winrate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0,
         "exposure": 0.0, "avg_trade": 0.0},
    ])], ignore_index=True)

    return {
        "holdout": holdout["name"],
        "holdout_start": str(holdout["holdout_start"]),
        "holdout_end": str(holdout["holdout_end"]),
        "summary": summary_df,
        "results": results,
        "benchmarks": benchmarks,
        "thresholds": {**base_thresholds, **ens_thresholds},
        "tuned": tuned,
        "oof": oof_dict,
        "train_prices": train_prices,
    }


def run_pipeline(df_feat: pd.DataFrame, n_holdouts: int = config.N_HOLDOUTS) -> dict:
    """Запустить multi-holdout и усреднить."""
    holdouts = SP.make_holdouts(df_feat, n_holdouts=n_holdouts,
                               holdout_days=config.HOLDOUT_DAYS)
    per_holdout = []
    for h in holdouts:
        print(f"\n===== {h['name']} =====")
        res = run_one_holdout(h, df_feat)
        per_holdout.append(res)
        print(res["summary"].to_string(index=False))

    all_names = list(per_holdout[0]["summary"]["name"])
    agg_rows = []
    for name in all_names:
        eqs = []
        for r in per_holdout:
            row = r["summary"][r["summary"]["name"] == name]
            if not row.empty:
                eqs.append(float(row["final_equity"].iloc[0]))
        if eqs:
            agg_rows.append({
                "name": name,
                "final_equity_mean": float(np.mean(eqs)),
                "final_equity_std": float(np.std(eqs)),
                "final_equity_min": float(np.min(eqs)),
                "n_holdouts": len(eqs),
            })
    agg_df = pd.DataFrame(agg_rows).sort_values("final_equity_mean", ascending=False)

    return {
        "per_holdout": per_holdout,
        "aggregate": agg_df,
    }