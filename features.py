# -*- coding: utf-8 -*-
"""Инженерия признаков. СТРОГО только прошлое.

Каждая фича в момент t использует данные только до t включительно.
Цель y_t = 1 если close_{t+1} > close_t. Никаких shift(-1) в признаках.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window=window, min_periods=window).mean()


def _std(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window=window, min_periods=window).std(ddof=0)


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Построить фичи + цель. Возвращает DataFrame с фичами и колонкой y."""
    df = df.copy()

    # Локальные время-фичи (если ещё нет).
    if "hour" not in df.columns:
        local = df.index.tz_convert(config.TZ)
        df["hour"] = local.hour
        df["minute"] = local.minute
        df["dayofweek"] = local.dayofweek

    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # Доходности (прошлое).
    for lag in (1, 2, 3, 5, 10):
        df[f"ret_{lag}"] = close.pct_change(lag)
    df["logret_1"] = np.log(close / close.shift(1))

    # Лаги цены.
    for lag in (1, 2, 3, 5):
        df[f"close_lag_{lag}"] = close.shift(lag)
        df[f"vol_lag_{lag}"] = vol.shift(lag)

    # Скользящие средние и std.
    for w in (5, 10, 20, 50):
        df[f"sma_{w}"] = _sma(close, w)
        df[f"ema_{w}"] = _ema(close, w)
    for w in (5, 10, 20):
        df[f"std_{w}"] = _std(close, w)

    # Z-score цены относительно SMA.
    for w in (10, 20):
        df[f"zscore_{w}"] = (close - df[f"sma_{w}"]) / df[f"std_{w}"].replace(0, np.nan)

    # ATR.
    df["atr_14"] = _atr(df, 14)
    df["atr_pct"] = df["atr_14"] / close

    # Объём.
    df["vol_sma_20"] = _sma(vol, 20)
    df["rel_volume"] = vol / df["vol_sma_20"].replace(0, np.nan)

    # Price action (текущий бар известен на закрытии).
    rng = (high - low).replace(0, np.nan)
    df["body"] = (close - open_) / rng
    df["upper_shadow"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / rng
    df["lower_shadow"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / rng
    df["range_pct"] = (high - low) / close
    df["gap"] = (open_ - close.shift(1)) / close.shift(1)

    # Расстояние до MA в %.
    for w in (5, 10, 20, 50):
        df[f"dist_sma_{w}"] = (close - df[f"sma_{w}"]) / df[f"sma_{w}"]

    # Время как категориальные/циклические.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    # Цель: close_{t+1} > close_t. shift(-1) только для цели, не для фичей.
    future = close.shift(-config.HORIZON)
    tgt = (future > close).astype(float)
    tgt[future.isna()] = np.nan
    df[config.TARGET_COL] = tgt

    # Удаляем строки с NaN в фичах (разогрев) и хвост без цели.
    feature_cols = feature_columns(df)
    df = df.dropna(subset=feature_cols + [config.TARGET_COL])
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Список колонок-признаков (исключая OHLCV, служебные, цель)."""
    exclude = {"open", "high", "low", "close", "volume", "hour", "minute",
               "dayofweek", "date", config.TARGET_COL}
    cols = [c for c in df.columns if c not in exclude]
    return cols


def split_xy(df: pd.DataFrame):
    cols = feature_columns(df)
    X = df[cols].copy()
    y = df[config.TARGET_COL].astype(int).copy()
    return X, y, cols