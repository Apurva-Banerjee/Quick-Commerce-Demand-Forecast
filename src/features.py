"""Feature engineering for demand forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour/day as cyclical features."""
    data = df.copy()
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)
    return data


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag and rolling statistics per SKU."""
    data = df.copy().sort_values(["product_id", "datetime"])
    grouped = data.groupby("product_id")

    for lag in [1, 24, 168]:
        data[f"lag_{lag}"] = grouped["sales"].shift(lag)

    data["rolling_mean_24"] = grouped["sales"].transform(lambda s: s.shift(1).rolling(24).mean())
    data["rolling_std_24"] = grouped["sales"].transform(lambda s: s.shift(1).rolling(24).std())
    data["rolling_mean_168"] = grouped["sales"].transform(lambda s: s.shift(1).rolling(168).mean())

    # Basic demand spike proxy.
    data["demand_spike_flag"] = (
        (data["festival_flag"] == 1) | (data["rolling_mean_24"] > data["rolling_mean_168"] * 1.20)
    ).astype(int)

    return data


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering and remove rows with undefined lags."""
    data = add_time_cyclic_features(df)
    data = add_lag_and_rolling_features(data)
    data = data.dropna().reset_index(drop=True)
    return data
