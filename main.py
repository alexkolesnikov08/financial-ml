# -*- coding: utf-8 -*-
"""Точка входа. Пока не запускать без готовности окружения.

Запуск: python main.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import config
import data as D
import features as F
from pipeline import run_pipeline
import eval as EV
import viz

def main():
    # 1. Данные
    df = D.load_or_download()
    df = D.clean(df)
    df = D.to_local(df)
    print(f"Данные: {df.shape}, {df.index.min()} .. {df.index.max()}")

    # 2. Фичи
    if getattr(config, "USE_CROSS_ASSETS", False):
        cross_assets = D.load_cross_assets()
        feat_df = F.build(df, cross_assets=cross_assets)
    else:
        feat_df = F.build(df)
    print(f"После фичей: {feat_df.shape}")

    # 3. Пайплайн
    result = run_pipeline(feat_df, n_holdouts=config.N_HOLDOUTS)

    # 4. Сохранение
    agg = result["aggregate"]
    print("\n===== AGGREGATE (среднее по hold-out) =====")
    print(agg.to_string(index=False))
    agg.to_csv(config.REPORTS_DIR / "aggregate.csv", index=False)

    # Сохраняем per-holdout summary
    for r in result["per_holdout"]:
        r["summary"].to_csv(config.REPORTS_DIR / f"summary_{r['holdout']}.csv", index=False)

    # Сохраняем артефакты
    EV.save_results({"aggregate": agg.to_dict(orient="records")}, config.REPORTS_DIR / "results.json")

    # Графики
    try:
        first = result["per_holdout"][0]
        paths = viz.generate_all_plots(result, oof_dict=first.get("oof"),
                                       prices=first.get("train_prices"))
        print("Графики сохранены:")
        for p in paths:
            print("  ", p)
    except Exception as e:
        print(f"[viz] ошибка генерации графиков: {e}")

    print(f"\nАртефакты сохранены в {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()