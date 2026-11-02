# -*- coding: utf-8 -*-
"""Загрузка и чистка цен из yfinance.

Строгая дисциплина по времени: никакой интерполяции в будущее, только чистка
дубликатов, сортировка по времени, приведение таймзоны.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

import config

warnings.filterwarnings("ignore")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s


def download(ticker: str = config.TICKER,
             interval: str = config.PREFERRED_INTERVAL,
             period: str = config.DOWNLOAD_PERIOD) -> pd.DataFrame:
    """Скачать OHLCV через yfinance. Возвращает DataFrame с DatetimeIndex (UTC)."""
    df = yf.download(
        ticker, interval=interval, period=period,
        auto_adjust=True, progress=False, threads=False,
        session=_session(),
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance вернул пустой DataFrame для {ticker} {interval}")
    # yfinance иногда возвращает MultiIndex-колонки
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.lower)
    needed = ["open", "high", "low", "close", "volume"]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"Нет колонки {c} в данных yfinance")
    df = df[needed]
    # Индекс -> UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Чистка: сортировка, дедуп, удаление NaN, проверка монотонности."""
    df = df.copy()
    df = df.sort_index()
    # Дедуп по индексу (оставляем последнее наблюдение).
    df = df[~df.index.duplicated(keep="last")]
    # Удаляем строки с NaN в OHLCV.
    df = df.dropna(subset=["open", "high", "low", "close"])
    # volume NaN -> 0
    df["volume"] = df["volume"].fillna(0)
    # Проверка монотонности
    if not df.index.is_monotonic_increasing:
        raise RuntimeError("Индекс не монотонен после чистки")
    # Убираем нулевые/отрицательные цены
    df = df[(df["close"] > 0) & (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
    return df


def to_local(df: pd.DataFrame, tz: str = config.TZ) -> pd.DataFrame:
    """Добавить локальные время-фичи (без утечки: просто конвертация индекса)."""
    df = df.copy()
    local = df.index.tz_convert(tz)
    df["hour"] = local.hour
    df["minute"] = local.minute
    df["dayofweek"] = local.dayofweek
    df["date"] = local.date
    return df


def load_or_download(interval: str | None = None,
                     force: bool = False) -> pd.DataFrame:
    """Загрузить кэшированный parquet или скачать заново.

    Пробует preferred interval, при неудаче/малой истории - fallback.
    """
    if interval is None:
        interval = config.PREFERRED_INTERVAL
    cache = config.DATA_DIR / f"{config.TICKER}_{interval}.parquet"
    if cache.exists() and not force:
        df = pd.read_parquet(cache)
        if len(df) >= 200:
            return df
    # Скачиваем
    try:
        df = download(config.TICKER, interval, config.DOWNLOAD_PERIOD)
        df = clean(df)
    except Exception as e:
        if interval != config.FALLBACK_INTERVAL:
            print(f"[data] {interval} не удался ({e}), пробую {config.FALLBACK_INTERVAL}")
            return load_or_download(config.FALLBACK_INTERVAL, force=True)
        raise
    if len(df) < 200 and interval != config.FALLBACK_INTERVAL:
        print(f"[data] {interval}: мало строк ({len(df)}), fallback на {config.FALLBACK_INTERVAL}")
        return load_or_download(config.FALLBACK_INTERVAL, force=True)
    df.to_parquet(cache)
    return df


def ensure_min_history(df: pd.DataFrame, min_bars: int = 500) -> pd.DataFrame:
    if len(df) < min_bars:
        raise RuntimeError(f"Истории мало: {len(df)} баров < {min_bars}")
    return df


if __name__ == "__main__":
    df = load_or_download()
    df = clean(df)
    print(df.head())
    print("shape:", df.shape)
    print("range:", df.index.min(), df.index.max())