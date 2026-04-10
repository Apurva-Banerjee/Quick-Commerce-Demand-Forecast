"""XGBoost model training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError:
    XGBRegressor = None


@dataclass
class XGBArtifacts:
    model: Any
    feature_cols: list[str]
    category_map: dict[str, int]


def prepare_xgb_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, int]]:
    """Prepare tabular features for XGBoost."""
    train = train_df.copy()
    test = test_df.copy()

    all_skus = sorted(train["product_id"].unique())
    category_map = {sku: idx for idx, sku in enumerate(all_skus)}
    train["product_code"] = train["product_id"].map(category_map)
    test["product_code"] = test["product_id"].map(category_map)

    feature_cols = [
        "product_code",
        "hour",
        "day_of_week",
        "weekend",
        "month",
        "festival_flag",
        "temperature",
        "rainfall",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "lag_1",
        "lag_24",
        "lag_168",
        "rolling_mean_24",
        "rolling_std_24",
        "rolling_mean_168",
        "demand_spike_flag",
    ]
    return train, test, feature_cols, category_map


def train_xgb(train_df: pd.DataFrame, feature_cols: list[str], target_col: str = "sales") -> Any:
    """Train XGBoost regressor (or sklearn fallback if unavailable)."""
    if XGBRegressor is not None:
        model = XGBRegressor(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        )
    else:
        model = HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_depth=10,
            max_iter=400,
            random_state=42,
        )
    model.fit(train_df[feature_cols], train_df[target_col])
    return model


def save_xgb_artifacts(artifacts: XGBArtifacts, path: Path) -> None:
    """Save XGBoost model and metadata to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, path)
