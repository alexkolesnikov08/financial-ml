# -*- coding: utf-8 -*-
"""Сбор данных: скачать OHLCV через yfinance и сохранить в data/.

Запуск:
    python collect_data.py
    python collect_data.py --ticker AAPL --interval 1h --period 730d
    python collect_data.py --cross   # скачать cross-assets (SPY, QQQ, VIX)
"""

from __future__ import annotations

import argparse
import os
import sys

import config
import data as D

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=config.TICKER)
    parser.add_argument("--interval", default=config.TIMEFRAME)
    parser.add_argument("--period", default=None)
    parser.add_argument("--cross", action="store_true", help="Скачать cross-assets")
    args = parser.parse_args()

    os.makedirs(config.DATA_DIR, exist_ok=True)

    # Основной тикер
    print(f"Скачиваем {args.ticker} {args.interval} ({args.period})...")
    df = D.download(args.ticker, interval=args.interval, period=args.period)
    df = D.clean(df)
    df = D.to_local(df)
    out = config.DATA_DIR / f"{args.ticker}_{args.interval}.parquet"
    df.to_parquet(out)
    print(f"Сохранено: {out} ({len(df)} баров)")

    # Cross-assets
    if args.cross or getattr(config, "USE_CROSS_ASSETS", False):
        print("Скачиваем cross-assets...")
        cross = D.load_cross_assets()
        for name, cdf in cross.items():
            cpath = config.DATA_DIR / f"{name}_{args.interval}.parquet"
            cdf.to_parquet(cpath)
            print(f"  {name}: {len(cdf)} баров -> {cpath}")

if __name__ == "__main__":
    main()
