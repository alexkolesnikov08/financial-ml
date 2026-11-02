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
from features import split_xy, select_features
import persist as PERSIST


def save_models(*args, **kwargs):
    """Обёртка над persist.save_models (для совместимости)."""
    return PERSIST.save_models(*args, **kwargs)


def _optimize_threshold(prices, signals, grid=config.THRESHOLD_GRID, min_trades: int = 5):
    prices = np.asarray(prices, dtype=float)
    signals = np.asarray(signals, dtype=float)
    best_thr = 0.5
    best_eq = -np.inf
    best_key = None
    candidates = []
    for thr in grid:
        try:
            res = BT.simulate(prices, signals, float(thr), confidence_threshold=0.0)
        except Exception:
            continue
        nt = int(res.get("num_trades", 0) or 0)
        pf = res.get("profit_factor", None)
        pf = float(pf) if pf is not None and np.isfinite(pf) else 0.0
        candidates.append((float(thr), res, nt, pf))
        if nt < int(min_trades):
            continue
        # метрика с учётом числа сделок: final_equity, затем nt, затем profit_factor
        key = (float(res["final_equity"]), nt, pf)
        if best_key is None or key > best_key:
            best_key = key
            best_thr = float(thr)
            best_eq = float(res["final_equity"])
    # Fallback: если ни один порог не дал min_trades сделок — берём лучший по (nt, final_equity).
    if best_key is None and candidates:
        fb = max(candidates, key=lambda c: (c[2], c[1]["final_equity"]))
        best_thr = fb[0]
        best_eq = float(fb[1]["final_equity"])
    if not np.isfinite(best_eq):
        best_eq = float(config.START_CAPITAL)
    return best_thr, float(best_eq)


def run_one_holdout(holdout: dict, df_feat: pd.DataFrame) -> dict:
    """Прогнать один hold-out период. Возвращает dict с метриками всех моделей/ансамблей."""
    train_mask = holdout["train_mask"]
    holdout_mask = holdout["holdout_mask"]
    train_df = df_feat.iloc[np.where(train_mask)[0]].copy()
    hold_df = df_feat.iloc[np.where(holdout_mask)[0]].copy()

    if train_df.empty or hold_df.empty:
        raise ValueError(
            f"Пустой train/holdout: train={len(train_df)}, holdout={len(hold_df)}")

    train_prices = train_df["close"].values
    hold_prices = hold_df["close"].values

    # Стратегические маски/параметры для backtest.
    time_mask = None
    try:
        _tf = getattr(config, "TIME_FILTERS", None)
        if _tf is not None:
            if isinstance(_tf, dict):
                time_mask = BT.make_time_mask(hold_df, **_tf)
            elif _tf is True:
                time_mask = BT.make_time_mask(hold_df)
            else:
                time_mask = None
        if time_mask is not None:
            time_mask = np.asarray(time_mask, dtype=bool)
    except Exception as e:
        print(f"[time_mask] FAILED {e}")
        time_mask = None

    trend_mask = None
    try:
        _use_trend = bool(getattr(config, "USE_TREND_MASK", False)) or getattr(config, "TREND_MASK", None) is not None
        if _use_trend:
            if "ema_50" in hold_df.columns:
                _tm = hold_df["close"] > hold_df["ema_50"]
                if "ema_20" in hold_df.columns:
                    _tm = _tm | (hold_df["ema_20"] > hold_df["ema_50"])
                trend_mask = np.asarray(_tm.fillna(False), dtype=bool)
            elif "sma_50" in hold_df.columns:
                _tm = hold_df["close"] > hold_df["sma_50"]
                trend_mask = np.asarray(_tm.fillna(False), dtype=bool)
    except Exception:
        trend_mask = None

    atr_arr = None
    if "atr_14" in hold_df.columns:
        atr_arr = np.asarray(hold_df["atr_14"].values, dtype=float)

    thr_hi = getattr(config, "THR_HI", None)
    if thr_hi is None:
        thr_hi = getattr(config, "CONFIDENCE_THRESHOLD", None)
    thr_lo = getattr(config, "THR_LO", None)
    min_hold = getattr(config, "MIN_HOLD", 0)
    cooldown = getattr(config, "COOLDOWN", 0)
    trail_mult = getattr(config, "TRAIL_MULT", None)
    use_consensus = bool(getattr(config, "USE_CONSENSUS", False))

    Xtr, ytr, cols = split_xy(train_df)
    Xho, yho, _ = split_xy(hold_df)

    # Feature selection (v1.3): отбор признаков на train, применяем к hold-out.
    if bool(getattr(config, "FEATURE_SELECTION", False)):
        try:
            selected = select_features(
                Xtr, ytr, method="importance",
                top_k=getattr(config, "FEATURE_TOP_K", 40))
            if selected:
                Xtr = Xtr[selected]
                Xho = Xho[selected]
                cols = list(selected)
        except Exception as e:
            print(f"[feature_selection] FAILED {e}")

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
    stackers = {}

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

    # Edge-параметры на OOF: средние движения на выигрышных/убыточных сделках
    # базовой стратегии (soft mean по OOF). Long-only OOF-сигнал: для short
    # параметры оставляем как есть (long-only), edge-фильтр применяется в backtest
    # с учётом направления сделки.
    edge_mu_win = None
    edge_mu_loss = None
    try:
        if oof_dict:
            oof_arrays = [np.asarray(v, dtype=float) for v in oof_dict.values()]
            # soft mean с nanmean по столбцам
            stacked = np.column_stack(oof_arrays)
            oof_signal = np.nanmean(stacked, axis=1)
            oof_signal = np.where(np.isfinite(oof_signal), oof_signal, np.nan)
            _edge_thr = thr_hi if thr_hi is not None else 0.5
            edge_sim = BT.simulate(train_prices, oof_signal, float(_edge_thr))
            _trades = edge_sim.get("trades", []) or []
            _wins = []
            _losses = []
            for t in _trades:
                if t.get("exit_i") is None or t.get("pnl") is None:
                    continue
                ep = t.get("entry_price")
                xp = t.get("exit_price")
                if ep is None or xp is None:
                    continue
                try:
                    move = float(xp) / float(ep) - 1.0
                except Exception:
                    continue
                if t["pnl"] > 0:
                    _wins.append(move)
                elif t["pnl"] < 0:
                    _losses.append(abs(move))
            if _wins:
                edge_mu_win = float(np.mean(_wins))
            if _losses:
                edge_mu_loss = float(np.mean(_losses))
    except Exception as e:
        print(f"[edge] FAILED {e}")
        edge_mu_win = None
        edge_mu_loss = None

    # 8. Ансамбли на hold-out.
    ens_hold_probas = {}
    ens_hold_probas["ens_soft_mean"] = ENS.soft_mean(base_hold_proba)
    ens_hold_probas["ens_weighted_soft"] = ENS.weighted_soft(base_hold_proba, weights)
    ens_hold_probas["ens_geom_mean"] = ENS.geom_mean(base_hold_proba)
    ens_hold_probas["ens_rank_avg"] = ENS.rank_avg(base_hold_proba)
    ens_hold_probas["ens_hard_vote"] = ENS.hard_vote(base_hold_proba)

    # stackers для hold-out (обучаем на finite OOF)
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

    # Матрица консенсуса из base_hold_proba.
    consensus_matrix = None
    if use_consensus:
        try:
            consensus_matrix = np.column_stack(
                [np.asarray(base_hold_proba[n], dtype=float) for n in base_hold_proba]
            )
        except Exception as e:
            print(f"[consensus] FAILED {e}")
            consensus_matrix = None

    def _sim_kwargs():
        kw = {
            "thr_lo": thr_lo,
            "time_mask": time_mask,
            "trend_mask": trend_mask,
            "atr": atr_arr,
            "trail_mult": trail_mult,
            "min_hold": min_hold,
            "cooldown": cooldown,
            "edge_mu_win": edge_mu_win,
            "edge_mu_loss": edge_mu_loss,
            "edge_threshold": getattr(config, "EDGE_THRESHOLD", None),
            "confidence_threshold": getattr(config, "CONFIDENCE_THRESHOLD", None),
            "short": getattr(config, "USE_SHORT", False),
        }
        if use_consensus and consensus_matrix is not None:
            kw["consensus_signals"] = consensus_matrix
            kw["min_consensus"] = 2
        return kw

    # 9. Симуляция для базовых.
    results = {}
    base_rows = []
    for name in zoo.keys():
        proba = base_hold_proba[name]
        if thr_hi is None:
            thr = base_thresholds.get(name, 0.5)
        else:
            thr = thr_hi
        sim = BT.simulate(hold_prices, proba, thr, **_sim_kwargs())
        results[name] = sim
        y_pred = (np.asarray(proba) >= thr).astype(int)
        base_rows.append(EV.summarize_holdout(sim, name, y_true=yho, y_pred=y_pred))

    # Ансамбли.
    ens_rows = []
    for name, proba in ens_hold_probas.items():
        if thr_hi is None:
            thr = ens_thresholds.get(name, 0.5)
        else:
            thr = thr_hi
        sim = BT.simulate(hold_prices, proba, thr, **_sim_kwargs())
        results[name] = sim
        y_pred = (np.asarray(proba) >= thr).astype(int)
        ens_rows.append(EV.summarize_holdout(sim, name, y_true=yho, y_pred=y_pred))

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
         "avg_trade": bh["final_equity"] - config.START_CAPITAL,
         "accuracy": None, "precision": None, "recall": None, "f1": None},
        {"name": "cash", "final_equity": cash["final_equity"],
         "total_return": 0.0, "num_trades": 0, "num_closed": 0,
         "winrate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0,
         "exposure": 0.0, "avg_trade": 0.0,
         "accuracy": None, "precision": None, "recall": None, "f1": None},
    ])], ignore_index=True)

    # Сохранение артефактов (joblib).
    strategic = {
        "THR_HI": getattr(config, "THR_HI", None),
        "THR_LO": getattr(config, "THR_LO", None),
        "MIN_HOLD": getattr(config, "MIN_HOLD", None),
        "COOLDOWN": getattr(config, "COOLDOWN", None),
        "TRAIL_MULT": getattr(config, "TRAIL_MULT", None),
        "TIME_FILTERS": getattr(config, "TIME_FILTERS", None),
        "USE_CONSENSUS": getattr(config, "USE_CONSENSUS", None),
        "CONFIDENCE_THRESHOLD": getattr(config, "CONFIDENCE_THRESHOLD", None),
        "USE_SHORT": getattr(config, "USE_SHORT", None),
        "edge_mu_win": edge_mu_win,
        "edge_mu_loss": edge_mu_loss,
    }
    try:
        PERSIST.save_models(
            holdout["name"],
            final_base=final_base,
            stackers=stackers,
            oof=oof_dict,
            thresholds={**base_thresholds, **ens_thresholds},
            train_prices=train_prices,
            feature_columns=cols,
            tuned=tuned,
            extra={
                **strategic,
                "strategic_params": strategic,
                "strategic": strategic,
            },
        )
    except Exception as e:
        print(f"[persist] FAILED {e}")

    # Regime-анализ (если включён): используем split.regime_split и eval.regime_metrics.
    regime_analysis = None
    if bool(getattr(config, "REGIME_ANALYSIS", False)):
        try:
            regime_masks = SP.regime_split(df_feat, holdout)
            thr_map = {**base_thresholds, **ens_thresholds}
            all_probas = {}
            all_probas.update(base_hold_proba)
            all_probas.update(ens_hold_probas)
            regime_analysis = {}
            for name, proba in all_probas.items():
                thr = thr_hi if thr_hi is not None else thr_map.get(name, 0.5)
                y_pred = (np.asarray(proba) >= float(thr)).astype(int)
                try:
                    regime_analysis[name] = EV.regime_metrics(yho, y_pred, regime_masks)
                except Exception as e:
                    print(f"[regime_metrics] {name}: FAILED {e}")
        except Exception as e:
            print(f"[regime_analysis] FAILED {e}")
            regime_analysis = None

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
        "feature_columns": cols,
        "regime_analysis": regime_analysis,
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

    regime_analysis = None
    if bool(getattr(config, "REGIME_ANALYSIS", False)):
        regime_analysis = [r.get("regime_analysis") for r in per_holdout]

    return {
        "per_holdout": per_holdout,
        "aggregate": agg_df,
        "regime_analysis": regime_analysis,
    }