# -*- coding: utf-8 -*-
"""Визуализация результатов пайплайна.

Все функции принимают результат `run_pipeline` или отдельные артефакты
и сохраняют PNG в reports/. Не требуют запуска обучения.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

REPORTS = config.REPORTS_DIR


def _save(fig, name: str) -> Path:
    p = REPORTS / name
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_model_scatter(per_holdout: list[dict] | None = None,
                       aggregate: pd.DataFrame | None = None,
                       filename: str = "model_scatter.png") -> Path:
    """Точки моделей: X=winrate, Y=total_return, размер=num_trades, цвет=тип."""
    # Берём последний hold-out summary (или агрегат).
    if per_holdout:
        sdf = per_holdout[-1]["summary"].copy()
    elif aggregate is not None:
        sdf = aggregate.copy()
    else:
        raise ValueError("Нужен per_holdout или aggregate")

    def kind(name):
        if name.startswith("stack_"):
            return "stacking"
        if name.startswith("ens_"):
            return "ensemble"
        if name in ("buy_and_hold", "cash"):
            return "benchmark"
        return "base"

    sdf["kind"] = sdf["name"].map(kind)
    color_map = {"base": "#4C78A8", "ensemble": "#F58518",
                 "stacking": "#54A24B", "benchmark": "#B279A2"}

    fig, ax = plt.subplots(figsize=(9, 6))
    for k, g in sdf.groupby("kind"):
        sizes = np.clip(g.get("num_trades", pd.Series(1, index=g.index)).fillna(1) * 4 + 20, 20, 400)
        ax.scatter(g["winrate"], g["total_return"] * 100, s=sizes,
                   c=color_map[k], label=k, alpha=0.75, edgecolor="black", linewidth=0.5)
        for _, r in g.iterrows():
            ax.annotate(r["name"], (r["winrate"], r["total_return"] * 100),
                        fontsize=7, alpha=0.8, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Winrate")
    ax.set_ylabel("Total return, %")
    ax.set_title("Модели: return vs winrate (размер = число сделок)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, filename)


def plot_equity_curves(per_holdout: dict, top_n: int = 6,
                       filename: str = "equity_curves.png") -> Path:
    """Кривые капитала для топ-N моделей по final_equity."""
    sdf = per_holdout["summary"]
    top = sdf.sort_values("final_equity", ascending=False)["name"].head(top_n).tolist()
    res = per_holdout["results"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in top:
        if name in res:
            eq = res[name]["equity_curve"]
            ax.plot(eq, label=name, linewidth=1.2)
    ax.axhline(config.START_CAPITAL if False else 1000, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Бар")
    ax.set_ylabel("Equity, $")
    ax.set_title(f"Кривые капитала (топ-{top_n})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, filename)


def plot_holdout_stability(per_holdout: list[dict],
                           filename: str = "holdout_stability.png") -> Path:
    """Сgrouped bar: final_equity по hold-out для каждой модели."""
    names = list(per_holdout[0]["summary"]["name"])
    x = np.arange(len(names))
    width = 0.8 / len(per_holdout)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, r in enumerate(per_holdout):
        sdf = r["summary"].set_index("name")
        vals = [float(sdf.loc[n, "final_equity"]) if n in sdf.index else 0.0 for n in names]
        ax.bar(x + i * width, vals, width, label=r["holdout"], alpha=0.85)
    ax.axhline(1000, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(x + width * (len(per_holdout) - 1) / 2)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Final equity, $")
    ax.set_title("Стабильность по hold-out периодам")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, filename)


def plot_drawdown(per_holdout: dict, model_name: str | None = None,
                  filename: str = "drawdown.png") -> Path:
    """Underwater-график просадки для выбранной (или лучшей) модели."""
    sdf = per_holdout["summary"]
    if model_name is None:
        model_name = sdf.sort_values("final_equity", ascending=False)["name"].iloc[0]
    eq = per_holdout["results"][model_name]["equity_curve"]
    eq = np.asarray(eq, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-12) * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(range(len(dd)), dd, 0, color="#E45756", alpha=0.6)
    ax.plot(dd, color="#E45756", linewidth=0.8)
    ax.set_xlabel("Бар")
    ax.set_ylabel("Drawdown, %")
    ax.set_title(f"Просадка: {model_name} (max {dd.min():.2f}%)")
    ax.grid(True, alpha=0.3)
    return _save(fig, filename)


def plot_oof_correlation(oof_dict: dict, filename: str = "oof_corr.png") -> Path:
    """Тепловая карта корреляций OOF-предсказаний базовых моделей."""
    names = list(oof_dict.keys())
    M = np.column_stack([np.asarray(oof_dict[n], dtype=float) for n in names])
    mask = np.all(np.isfinite(M), axis=1)
    M = M[mask]
    corr = np.corrcoef(M, rowvar=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Корреляция OOF-предсказаний")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")
    return _save(fig, filename)


def plot_threshold_sensitivity(oof_dict: dict, prices,
                               filename: str = "threshold_sens.png") -> Path:
    """Final equity vs threshold для каждой базовой модели (на OOF)."""
    from backtest import simulate
    grid = config.THRESHOLD_GRID
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, oof in oof_dict.items():
        eqs = []
        for thr in grid:
            try:
                r = simulate(np.asarray(prices, float), np.asarray(oof, float), float(thr))
                eqs.append(r["final_equity"])
            except Exception:
                eqs.append(np.nan)
        ax.plot(grid, eqs, label=name, linewidth=1.0, alpha=0.8)
    ax.axhline(1000, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Порог")
    ax.set_ylabel("Final equity на OOF, $")
    ax.set_title("Чувствительность к порогу (OOF)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    return _save(fig, filename)


def generate_all_plots(result: dict, oof_dict: dict | None = None,
                       prices=None) -> list[Path]:
    """Сгенерировать весь набор графиков из результата run_pipeline."""
    out = []
    per = result["per_holdout"]
    agg = result["aggregate"]
    out.append(plot_model_scatter(per_holdout=per))
    out.append(plot_holdout_stability(per))
    if per:
        out.append(plot_equity_curves(per[0]))
        out.append(plot_drawdown(per[0]))
    if oof_dict is not None:
        try:
            out.append(plot_oof_correlation(oof_dict))
        except Exception as e:
            print(f"[viz] oof_corr failed: {e}")
        if prices is not None:
            try:
                out.append(plot_threshold_sensitivity(oof_dict, prices))
            except Exception as e:
                print(f"[viz] threshold_sens failed: {e}")
    return out