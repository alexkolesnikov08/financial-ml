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


def build(df: pd.DataFrame, cross_assets: dict | None = None) -> pd.DataFrame:
    """Построить фичи + цель. Возвращает DataFrame с фичами и колонкой y."""
    df = df.copy()

    # Проверка сортировки: индекс должен быть монотонно возрастающим.
    if not df.index.is_monotonic_increasing:
        raise ValueError("df.index должен быть отсортирован по времени (monotonic increasing)")

    # Локальные время-фичи (если ещё нет).
    if "hour" not in df.columns:
        local = df.index.tz_convert(config.TZ)
        df["hour"] = local.hour
        df["minute"] = local.minute
        df["dayofweek"] = local.dayofweek
    if "date" not in df.columns:
        local = df.index.tz_convert(config.TZ)
        df["date"] = local.date

    # Cross-asset: выравнивание строго backward (timestamp актива <= t).
    if cross_assets:
        for ticker, adf in cross_assets.items():
            if adf is None or (hasattr(adf, "empty") and adf.empty):
                continue
            a = adf.copy()
            if not isinstance(a.index, pd.DatetimeIndex):
                try:
                    a.index = pd.to_datetime(a.index, utc=True)
                except Exception:
                    continue
            if a.index.tz is None:
                a.index = a.index.tz_localize("UTC")
            else:
                a.index = a.index.tz_convert("UTC")
            a = a.sort_index()
            if "close" not in a.columns:
                continue
            prefix = ticker.lower().replace("^", "").strip()
            right = a[["close"]].rename(columns={"close": f"{prefix}_close"})
            df = pd.merge_asof(df, right, left_index=True, right_index=True,
                               direction="backward", tolerance=pd.Timedelta(minutes=60))

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

    # ============ НОВЫЕ ФИЧИ v1.1 ============
    # Price action
    rng = (high - low).replace(0, np.nan)
    df["bar_ret_norm"] = (close - open_) / df["atr_14"].replace(0, np.nan)
    df["wick_ratio_upper"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / rng
    df["range_norm"] = (high - low) / df["atr_14"].replace(0, np.nan)

    # Session-running (сброс по дате)
    session_key = df["date"]
    session_high = high.groupby(session_key).cummax()
    session_open = df.groupby(session_key)["open"].transform("first")
    df["session_high_dist"] = (close - session_high) / session_high.replace(0, np.nan)
    df["session_open_ret"] = (close - session_open) / session_open.replace(0, np.nan)

    daily_close = df.groupby(session_key)["close"].last()
    prev_session_close_map = daily_close.shift(1)
    prev_session_close = session_key.map(prev_session_close_map)
    df["prev_session_close_dist"] = (close - prev_session_close) / prev_session_close.replace(0, np.nan)

    # Volume
    vol_mean_20 = vol.rolling(20, min_periods=20).mean()
    vol_std_20 = vol.rolling(20, min_periods=20).std(ddof=0)
    df["vol_z_20"] = (vol - vol_mean_20) / vol_std_20.replace(0, np.nan)

    typ = (high + low + close) / 3.0
    vol_cum = vol.groupby(session_key).cumsum()
    tp_vol_cum = (typ * vol).groupby(session_key).cumsum()
    vwap = tp_vol_cum / vol_cum.replace(0, np.nan)
    df["vwap_session_dist"] = (close - vwap) / vwap.replace(0, np.nan)

    up_vol = vol.where(close > open_, 0.0)
    down_vol = vol.where(close <= open_, 0.0)
    df["up_down_vol_ratio_20"] = (up_vol.rolling(20, min_periods=20).sum()
                                   / down_vol.rolling(20, min_periods=20).sum().replace(0, np.nan))
    df["vol_climax"] = df["rel_volume"] * (df["ret_1"].abs() * 100.0)

    # Volatility
    df["rv_20"] = df["ret_1"].rolling(20, min_periods=20).std(ddof=0)
    df["vol_ratio_5_20"] = (df["ret_1"].rolling(5, min_periods=5).std(ddof=0)
                            / df["ret_1"].rolling(20, min_periods=20).std(ddof=0).replace(0, np.nan))
    df["atr_change_5"] = df["atr_14"].pct_change(5)
    log_hl = np.log(high / low.replace(0, np.nan))
    df["parkinson_20"] = np.sqrt((log_hl ** 2).rolling(20, min_periods=20).mean() / (4.0 * np.log(2.0)))

    # Time/session (только прошлое/текущее время, без будущей длины сессии)
    bars_since_open = df.groupby(session_key).cumcount()
    df["bars_since_open"] = bars_since_open
    df["is_open_30min"] = (bars_since_open < 6).astype(int)
    # минуты до закрытия рынка (по часам, без утечки будущего)
    close_minutes = 16 * 60
    mins_of_day = df["hour"] * 60 + df["minute"]
    df["minutes_to_close"] = close_minutes - mins_of_day
    df["is_close_30min"] = ((df["minutes_to_close"] >= 0) & (df["minutes_to_close"] <= 30)).astype(int)
    df["is_lunch"] = ((df["hour"] == 12) | ((df["hour"] == 13) & (df["minute"] < 30))).astype(int)

    # Statistical
    ret_mean_20 = df["ret_1"].rolling(20, min_periods=20).mean()
    ret_std_20 = df["ret_1"].rolling(20, min_periods=20).std(ddof=0)
    df["ret_z_20"] = (df["ret_1"] - ret_mean_20) / ret_std_20.replace(0, np.nan)
    df["ret_percentile_20"] = df["ret_1"].rolling(20, min_periods=20).rank(pct=True)
    def _autocorr(x):
        s = pd.Series(x)
        if s.std(ddof=0) == 0:
            return np.nan
        return float(s.autocorr(lag=1))
    df["autocorr_ret_1_10"] = df["ret_1"].rolling(10, min_periods=10).apply(_autocorr, raw=False)
    df["skew_20"] = df["ret_1"].rolling(20, min_periods=20).skew()
    df["kurt_20"] = df["ret_1"].rolling(20, min_periods=20).kurt()

    # RSI-14
    _delta = close.diff()
    _gain = _delta.clip(lower=0)
    _loss = (-_delta).clip(lower=0)
    _avg_gain = _gain.rolling(14, min_periods=14).mean()
    _avg_loss = _loss.rolling(14, min_periods=14).mean()
    _rs = _avg_gain / _avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100.0 - 100.0 / (1.0 + _rs)

    # MACD
    _ema12 = _ema(close, 12)
    _ema26 = _ema(close, 26)
    df["macd"] = _ema12 - _ema26
    df["macd_signal"] = _ema(df["macd"], 9)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    _sma20 = _sma(close, 20)
    _std20 = _std(close, 20)
    df["bb_width_20"] = (4.0 * _std20) / _sma20.replace(0, np.nan)
    df["bb_position_20"] = (close - (_sma20 - 2.0 * _std20)) / (4.0 * _std20).replace(0, np.nan)

    df["atr_trend_5"] = df["atr_14"].diff(5)

    # Interaction
    df["ret_1_x_rel_volume"] = df["ret_1"] * df["rel_volume"]
    df["body_x_range"] = df["body"] * df["range_pct"]
    df["zscore_20_x_atr"] = df["zscore_20"] * df["atr_14"]

    # Время как категориальные/циклические.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    # Cross-asset фичи (если данные переданы) — обобщённо для любых *_close.
    cross_cols = [c for c in df.columns if c.endswith("_close")]
    if cross_cols:
        aapl_ret_1 = df["close"].pct_change()
        aapl_ret_5 = df["close"].pct_change(5)
        # Первый кросс-актив используем для rel_strength (как раньше spy).
        first = cross_cols[0]
        first_prefix = first[:-6]  # strip "_close"
        for c in cross_cols:
            prefix = c[:-6]
            df[f"{prefix}_ret_1"] = df[c].pct_change()
            # Для VIX используем diff уровня вместо pct_change? Оставляем pct_change унифицированно.
            df[f"{prefix}_corr_20"] = aapl_ret_1.rolling(20, min_periods=20).corr(df[c].pct_change())
        # rel_strength для первого актива
        df["rel_strength_5"] = aapl_ret_5 - df[first].pct_change(5)
        # Алиасы для обратной совместимости (spy/qqq/vix)
        if "spy_close" in cross_cols:
            df["spy_ret_1"] = df["spy_close"].pct_change() if "spy_ret_1" not in df.columns else df["spy_ret_1"]
        if "qqq_close" in cross_cols:
            df["qqq_ret_1"] = df["qqq_close"].pct_change() if "qqq_ret_1" not in df.columns else df["qqq_ret_1"]
        if "vix_close" in cross_cols:
            df["vix_change_1"] = df["vix_close"].diff()
            df["corr_aapl_spy_20"] = df.get("spy_corr_20", df.get("corr_aapl_spy_20"))
        # Убираем вспомогательные close кросс-активов.
        df = df.drop(columns=cross_cols)

    # Цель: close_{t+1} > close_t. shift(-1) только для цели, не для фичей.
    future = close.shift(-config.HORIZON)
    tgt = (future > close).astype(float)
    tgt[future.isna()] = np.nan
    df[config.TARGET_COL] = tgt

    # Target encoding (time-aware): expanding mean с байесовским сглаживанием (alpha).
    # Строго по прошлому: используем target.shift(1).
    _te_alpha = 10.0
    _past_y = df[config.TARGET_COL].shift(1)
    _global_mean = _past_y.expanding(min_periods=1).mean()

    def _te_expand(cat: pd.Series) -> pd.Series:
        grp = cat.astype(str)
        _sum = _past_y.groupby(grp).expanding(min_periods=1).sum().reset_index(level=0, drop=True)
        _cnt = _past_y.groupby(grp).expanding(min_periods=1).count().reset_index(level=0, drop=True)
        return (_sum + _te_alpha * _global_mean) / (_cnt + _te_alpha)

    _te_hour = _te_expand(df["hour"])
    df["te_hour"] = _te_hour
    df["hour_te"] = _te_hour

    _te_dow = _te_expand(df["dayofweek"])
    df["te_dayofweek"] = _te_dow
    df["dow_te"] = _te_dow

    _te_combo_cat = df["hour"].astype(str) + "_" + df["dayofweek"].astype(str)
    _te_combo = _te_expand(_te_combo_cat)
    df["te_hour_dow"] = _te_combo
    df["hour_dow_te"] = _te_combo

    # Удаляем мультиколлинеарные дубли (оставляем ema/zscore/rel_volume/циклические).
    drop_cols = [c for c in df.columns if c.startswith("sma_") or c.startswith("dist_sma_")]
    drop_cols += ["vol_sma_20"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

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