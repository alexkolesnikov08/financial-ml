#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Генерация графиков по результатам одного run (reports/)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tag", default=None)
    return p.parse_args()


def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def load_summaries(reports_dir: Path):
    files = sorted(reports_dir.glob("summary_holdout_*.csv"))
    out = []
    for f in files:
        try:
            out.append((f, pd.read_csv(f)))
        except Exception:
            pass
    return out


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    reports_dir = run_dir / "reports"
    figs_dir = run_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or run_dir.name

    aggregate = None
    agg_path = reports_dir / "aggregate.csv"
    if agg_path.exists():
        try:
            aggregate = pd.read_csv(agg_path)
        except Exception as e:
            print(f"[warn] aggregate read fail: {e}")

    results = None
    res_path = reports_dir / "results.json"
    if res_path.exists():
        try:
            with open(res_path) as fh:
                results = json.load(fh)
        except Exception as e:
            print(f"[warn] results.json read fail: {e}")

    summaries = load_summaries(reports_dir)

    # 1. equity_bar
    try:
        if aggregate is not None and "final_equity_mean" in aggregate.columns:
            df = aggregate.sort_values("final_equity_mean", ascending=False).copy()
            top = df.head(15).copy()
            if "buy_and_hold" in df["name"].values and "buy_and_hold" not in top["name"].values:
                top = pd.concat([top, df[df["name"] == "buy_and_hold"]], ignore_index=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=top, x="final_equity_mean", y="name", ax=ax, palette="viridis")
            ax.set_title(f"Final equity (mean) by model — {tag}")
            ax.set_xlabel("final_equity_mean")
            ax.set_ylabel("model")
            fig.text(0.99, 0.01, f"source: {tag}", ha="right", fontsize=8, alpha=0.6)
            save_fig(fig, figs_dir / "equity_bar.png")
            print("equity_bar.png")
    except Exception as e:
        print(f"[skip] equity_bar: {e}")

    # 2. holdout_stability
    try:
        if summaries:
            # top models
            if aggregate is not None:
                top_models = aggregate.sort_values("final_equity_mean", ascending=False).head(8)["name"].tolist()
            else:
                top_models = list(summaries[0][1]["name"].unique()[:8])
            data = {m: [] for m in top_models}
            for _, sdf in summaries:
                for m in top_models:
                    sub = sdf[sdf["name"] == m]
                    if len(sub):
                        data[m].append(float(sub["final_equity"].iloc[0]))
                    else:
                        data[m].append(float("nan"))
            fig, ax = plt.subplots(figsize=(10, 6))
            xs = list(range(len(summaries)))
            for m in top_models:
                ax.plot(xs, data[m], marker="o", label=m)
            ax.set_title(f"Holdout stability (final_equity) — {tag}")
            ax.set_xlabel("holdout index")
            ax.set_ylabel("final_equity")
            ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.text(0.99, 0.01, f"source: {tag}", ha="right", fontsize=8, alpha=0.6)
            save_fig(fig, figs_dir / "holdout_stability.png")
            print("holdout_stability.png")
    except Exception as e:
        print(f"[skip] holdout_stability: {e}")

    # 3. scatter_return_vs_accuracy
    try:
        if summaries and aggregate is not None:
            acc_means = []
            for _, sdf in summaries:
                acc_means.append(sdf[["name", "accuracy"]])
            acc_df = pd.concat(acc_means, ignore_index=True)
            acc_mean = acc_df.groupby("name", as_index=False)["accuracy"].mean()
            merged = aggregate.merge(acc_mean, on="name", how="left")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(merged["accuracy"], merged["final_equity_mean"], alpha=0.7)
            for _, r in merged.iterrows():
                ax.annotate(str(r["name"]), (r["accuracy"], r["final_equity_mean"]), fontsize=6, alpha=0.8)
            ax.set_title(f"Return vs accuracy — {tag}")
            ax.set_xlabel("accuracy (mean over holdouts)")
            ax.set_ylabel("final_equity_mean")
            fig.text(0.99, 0.01, f"source: {tag}", ha="right", fontsize=8, alpha=0.6)
            save_fig(fig, figs_dir / "scatter_return_vs_accuracy.png")
            print("scatter_return_vs_accuracy.png")
    except Exception as e:
        print(f"[skip] scatter: {e}")

    # 4. drawdown_bar
    try:
        if summaries:
            dd_list = []
            for _, sdf in summaries:
                dd_list.append(sdf[["name", "max_drawdown"]])
            dd_df = pd.concat(dd_list, ignore_index=True)
            dd_mean = dd_df.groupby("name", as_index=False)["max_drawdown"].mean()
            dd_mean = dd_mean.sort_values("max_drawdown", ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=dd_mean, x="max_drawdown", y="name", ax=ax, palette="rocket")
            ax.set_title(f"Max drawdown (mean) by model — {tag}")
            ax.set_xlabel("max_drawdown (mean)")
            ax.set_ylabel("model")
            fig.text(0.99, 0.01, f"source: {tag}", ha="right", fontsize=8, alpha=0.6)
            save_fig(fig, figs_dir / "drawdown_bar.png")
            print("drawdown_bar.png")
    except Exception as e:
        print(f"[skip] drawdown_bar: {e}")

    # 5. trades_bar
    try:
        if summaries:
            tr_list = []
            for _, sdf in summaries:
                tr_list.append(sdf[["name", "num_trades"]])
            tr_df = pd.concat(tr_list, ignore_index=True)
            tr_sum = tr_df.groupby("name", as_index=False)["num_trades"].sum()
            tr_sum = tr_sum.sort_values("num_trades", ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=tr_sum, x="num_trades", y="name", ax=ax, palette="mako")
            ax.set_title(f"Num trades (sum over holdouts) — {tag}")
            ax.set_xlabel("num_trades (sum)")
            ax.set_ylabel("model")
            fig.text(0.99, 0.01, f"source: {tag}", ha="right", fontsize=8, alpha=0.6)
            save_fig(fig, figs_dir / "trades_bar.png")
            print("trades_bar.png")
    except Exception as e:
        print(f"[skip] trades_bar: {e}")

    # 6. winrate_bar
    try:
        if summaries:
            wr_list = []
            for _, sdf in summaries:
                wr_list.append(sdf[["name", "winrate"]])
            wr_df = pd.concat(wr_list, ignore_index=True)
            wr_mean = wr_df.groupby("name", as_index=False)["winrate"].mean()
            wr_mean = wr_mean.sort_values("winrate", ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=wr_mean, x="winrate", y="name", ax=ax, palette="crest")
            ax.set_title(f"Winrate (mean) by model — {tag}")
            ax.set_xlabel("winrate (mean)")
            ax.set_ylabel("model")
            fig.text(0.99, 0.01, f"source: {tag}", ha="right", fontsize=8, alpha=0.6)
            save_fig(fig, figs_dir / "winrate_bar.png")
            print("winrate_bar.png")
    except Exception as e:
        print(f"[skip] winrate_bar: {e}")

    # 7. threshold_sens
    try:
        if isinstance(results, dict) and isinstance(results.get("thresholds"), dict) and results["thresholds"]:
            th = results["thresholds"]
            # normalize values to thr_hi
            rows = []
            for m, v in th.items():
                if isinstance(v, dict):
                    val = v.get("thr_hi", v.get("threshold", None))
                    if val is None:
                        # first numeric
                        for vv in v.values():
                            if isinstance(vv, (int, float)):
                                val = vv
                                break
                else:
                    val = v
                if val is not None:
                    rows.append((m, float(val)))
            if rows:
                th_df = pd.DataFrame(rows, columns=["name", "thr_hi"]).sort_values("thr_hi", ascending=False).head(20)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=th_df, x="thr_hi", y="name", ax=ax, palette="flare")
                ax.set_title(f"Threshold (thr_hi) by model — {tag}")
                ax.set_xlabel("thr_hi")
                ax.set_ylabel("model")
                fig.text(0.99, 0.01, f"source: {tag}", ha="right", fontsize=8, alpha=0.6)
                save_fig(fig, figs_dir / "threshold_sens.png")
                print("threshold_sens.png")
    except Exception as e:
        print(f"[skip] threshold_sens: {e}")

    # 8. regime_analysis
    try:
        if isinstance(results, dict) and isinstance(results.get("regime_analysis"), dict) and results["regime_analysis"]:
            ra = results["regime_analysis"]
            rows = []
            for model, regimes in ra.items():
                if not isinstance(regimes, dict):
                    continue
                for regime, metrics in regimes.items():
                    if not isinstance(metrics, dict):
                        continue
                    metric = metrics.get("accuracy", metrics.get("final_equity"))
                    if metric is None:
                        continue
                    rows.append({"model": model, "regime": regime, "metric": float(metric)})
            if rows:
                rdf = pd.DataFrame(rows)
                # filter top models
                if aggregate is not None:
                    top_models = aggregate.sort_values("final_equity_mean", ascending=False).head(8)["name"].tolist()
                    rdf = rdf[rdf["model"].isin(top_models)]
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=rdf, x="model", y="metric", hue="regime", ax=ax)
                ax.set_title(f"Regime analysis (accuracy) — {tag}")
                ax.set_xlabel("model")
                ax.set_ylabel("accuracy")
                ax.tick_params(axis="x", rotation=45)
                fig.text(0.99, 0.01, f"source: {tag}", ha="right", fontsize=8, alpha=0.6)
                save_fig(fig, figs_dir / "regime_analysis.png")
                print("regime_analysis.png")
    except Exception as e:
        print(f"[skip] regime_analysis: {e}")


if __name__ == "__main__":
    main()
