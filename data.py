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


def _min_bars_for(timeframe: str) -> int:
    """Минимальное число баров для валидации истории по таймфрейму."""
    tf = (timeframe or "").lower()
    return {
        "5m": 500,
        "15m": 300,
        "1h": 200,
        "1d": 200,
    }.get(tf, 200)


def period_for_timeframe(timeframe: str | None = None) -> str:
    """Подобрать период загрузки yfinance для таймфрейма."""
    tf = (timeframe if timeframe is not None else getattr(config, "TIMEFRAME", "5m")).lower()
    if tf == "1h":
        return "730d"
    if tf == "1d":
        return "5y"
    return getattr(config, "DOWNLOAD_PERIOD", "60d")


def list_timeframes() -> list[str]:
    """Поддерживаемые таймфреймы (v1.3)."""
    tfs = getattr(config, "TIMEFRAMES", None)
    if not tfs:
        tfs = ["5m", "15m", "1h", "1d"]
    return list(tfs)


def load_or_download(interval: str | None = None,
                      period: str | None = None,
                      force: bool = False,
                      min_bars: int | None = None) -> pd.DataFrame:
    """Загрузить кэшированный parquet или скачать заново."""
    if interval is None:
        interval = getattr(config, "TIMEFRAME", config.PREFERRED_INTERVAL)
    if period is None:
        period = period_for_timeframe(interval)
    if min_bars is None:
        min_bars = _min_bars_for(interval)
    cache = config.DATA_DIR / f"{config.TICKER}_{interval}.parquet"
    if cache.exists() and not force:
        try:
            df = pd.read_parquet(cache)
            if len(df) >= 200:
                try:
                    return ensure_min_history(df, min_bars=min_bars)
                except RuntimeError:
                    return df
        except Exception:
            pass
    try:
        df = download(config.TICKER, interval, period)
        df = clean(df)
    except Exception as e:
        if interval != config.FALLBACK_INTERVAL:
            print(f"[data] {interval} не удался ({e}), пробую {config.FALLBACK_INTERVAL}")
            return load_or_download(config.FALLBACK_INTERVAL, force=True)
        raise
    if len(df) < 200 and interval != config.FALLBACK_INTERVAL:
        print(f"[data] {interval}: мало строк ({len(df)}), fallback на {config.FALLBACK_INTERVAL}")
        return load_or_download(config.FALLBACK_INTERVAL, force=True)
    df = ensure_min_history(df, min_bars=min_bars)
    df.to_parquet(cache)
    return df


def ensure_min_history(df: pd.DataFrame, min_bars: int = 500) -> pd.DataFrame:
    if len(df) < min_bars:
        raise RuntimeError(f"Истории мало: {len(df)} баров < {min_bars}")
    return df


def _load_one_cross(ticker: str,
                    interval: str,
                    period: str) -> pd.DataFrame:
    cache = config.DATA_DIR / f"{ticker}_{interval}.parquet"
    if cache.exists():
        try:
            df = pd.read_parquet(cache)
            if len(df) >= 200:
                return df
        except Exception:
            pass
    try:
        df = download(ticker, interval, period)
        df = clean(df)
    except Exception as e:
        if interval != config.FALLBACK_INTERVAL:
            print(f"[data] {ticker} {interval} не удался ({e}), пробую {config.FALLBACK_INTERVAL}")
            return _load_one_cross(ticker, config.FALLBACK_INTERVAL, period_for_timeframe(config.FALLBACK_INTERVAL))
        raise
    if len(df) < 200 and interval != config.FALLBACK_INTERVAL:
        print(f"[data] {ticker} {interval}: мало строк ({len(df)}), fallback на {config.FALLBACK_INTERVAL}")
        return _load_one_cross(ticker, config.FALLBACK_INTERVAL, period_for_timeframe(config.FALLBACK_INTERVAL))
    df.to_parquet(cache)
    return df


def load_cross_assets(interval: str | None = None,
                      period: str | None = None) -> dict:
    """Загрузить кросс-активы (SPY, QQQ, ^VIX и др.) из config.CROSS_ASSETS.

    Возвращает dict ticker -> DataFrame (UTC DatetimeIndex, очищенный).
    Кэш parquet per ticker/interval. Скачивание через HTTP-session (yfinance).
    По умолчанию использует config.TIMEFRAME.
    """
    if interval is None:
        interval = getattr(config, "TIMEFRAME", config.PREFERRED_INTERVAL)
    if period is None:
        period = period_for_timeframe(interval)
    out = {}
    for ticker in getattr(config, "CROSS_ASSETS", []):
        try:
            df = _load_one_cross(ticker, interval, period)
            try:
                df = ensure_min_history(df, min_bars=_min_bars_for(interval))
            except RuntimeError:
                pass
            out[ticker] = df
        except Exception as e:
            print(f"[data] не удалось загрузить {ticker}: {e}")
    return out


if __name__ == "__main__":
    df = load_or_download()
    df = clean(df)
    print(df.head())
    print("shape:", df.shape)
    print("range:", df.index.min(), df.index.max())