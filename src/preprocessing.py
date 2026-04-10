"""Preprocessing utilities for time-series demand data."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich basic time features."""
    data = df.copy()
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    data = data.dropna(subset=["datetime", "product_id", "sales"])
    data = data.sort_values(["product_id", "datetime"]).reset_index(drop=True)

    # Fill weather missingness with product-level then global medians.
    for col in ["temperature", "rainfall"]:
        data[col] = data.groupby("product_id")[col].transform(lambda s: s.fillna(s.median()))
        data[col] = data[col].fillna(data[col].median())

    data["hour"] = data["datetime"].dt.hour
    data["day_of_week"] = data["datetime"].dt.dayofweek
    data["weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["month"] = data["datetime"].dt.month

    return data


def time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-aware train/test split for each SKU to avoid leakage."""
    train_parts = []
    test_parts = []

    for sku, group in df.groupby("product_id", sort=False):
        split_idx = int(len(group) * train_ratio)
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def scale_columns(train_df: pd.DataFrame, test_df: pd.DataFrame, numeric_cols: list[str]):
    """Scale selected numeric columns and return fitted scaler."""
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    train_scaled[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    test_scaled[numeric_cols] = scaler.transform(test_df[numeric_cols])
    return train_scaled, test_scaled, scaler
