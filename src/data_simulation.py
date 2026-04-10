"""Synthetic data generation for quick-commerce demand forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import RANDOM_STATE, SKUS


@dataclass
class SimulatorConfig:
    """Configuration for realistic hourly demand simulation."""

    random_state: int = RANDOM_STATE
    skus: List[str] = None

    def __post_init__(self) -> None:
        if self.skus is None:
            self.skus = SKUS


def _holiday_dates(index: pd.DatetimeIndex) -> pd.Series:
    """Create a simulated holiday/festival flag."""
    fixed_holidays = pd.to_datetime(
        ["2025-01-01", "2025-01-26", "2025-02-14", "2025-03-08", "2025-03-29"]
    )
    return index.normalize().isin(fixed_holidays).astype(int)


def _simulate_weather(index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate hourly weather features with mild seasonality."""
    day_of_year = index.dayofyear.to_numpy()
    hour = index.hour.to_numpy()

    temp = 22 + 8 * np.sin(2 * np.pi * day_of_year / 365) + 3 * np.sin(2 * np.pi * hour / 24)
    temp = temp + rng.normal(0, 1.2, len(index))
    rainfall = np.maximum(0, rng.gamma(shape=1.2, scale=2.0, size=len(index)) - 1.0)
    rainfall[rng.random(len(index)) < 0.65] = 0.0

    return pd.DataFrame({"temperature": temp, "rainfall": rainfall}, index=index)


def _sku_profile(sku: str) -> Dict[str, float]:
    """Per-SKU baseline behavior."""
    mapping = {
        "MILK_1L": {"base": 80, "price_sensitivity": 1.10},
        "BREAD_WHITE": {"base": 65, "price_sensitivity": 0.95},
        "BANANA_1KG": {"base": 45, "price_sensitivity": 0.85},
        "APPLE_1KG": {"base": 40, "price_sensitivity": 0.80},
        "EGGS_12": {"base": 55, "price_sensitivity": 0.90},
    }
    return mapping.get(sku, {"base": 50, "price_sensitivity": 1.0})


def generate_synthetic_data(start: str, end: str, config: SimulatorConfig | None = None) -> pd.DataFrame:
    """Generate realistic multi-SKU hourly demand data."""
    config = config or SimulatorConfig()
    rng = np.random.default_rng(config.random_state)
    dt_index = pd.date_range(start=start, end=end, freq="h")

    weather = _simulate_weather(dt_index, rng)
    holiday = _holiday_dates(dt_index)

    rows = []
    for sku in config.skus:
        profile = _sku_profile(sku)
        hours = dt_index.hour.to_numpy()
        day_of_week = dt_index.dayofweek.to_numpy()
        weekend = (day_of_week >= 5).astype(int)

        morning_peak = np.exp(-((hours - 9) ** 2) / 18)
        evening_peak = np.exp(-((hours - 20) ** 2) / 20)
        weekly_pattern = 1 + 0.10 * np.cos(2 * np.pi * day_of_week / 7)
        trend = np.linspace(0.95, 1.12, len(dt_index))
        weather_impact = 1 + 0.008 * weather["rainfall"].to_numpy() - 0.004 * (weather["temperature"].to_numpy() - 24)
        holiday_boost = 1 + 0.25 * holiday

        expected = (
            profile["base"]
            * (1 + 0.7 * morning_peak + 0.9 * evening_peak)
            * weekly_pattern
            * (1 + 0.14 * weekend)
            * trend
            * weather_impact
            * holiday_boost
        )

        # Add realistic random fluctuations and occasional stockout-like dips.
        noise = rng.normal(0, 7, len(dt_index))
        sales = np.maximum(0, expected + noise).round().astype(int)
        stockout_mask = rng.random(len(dt_index)) < 0.005
        sales[stockout_mask] = np.maximum(0, sales[stockout_mask] - rng.integers(20, 50, stockout_mask.sum()))

        sku_df = pd.DataFrame(
            {
                "product_id": sku,
                "datetime": dt_index,
                "sales": sales,
                "temperature": weather["temperature"].to_numpy(),
                "rainfall": weather["rainfall"].to_numpy(),
                "festival_flag": holiday,
            }
        )
        rows.append(sku_df)

    df = pd.concat(rows, ignore_index=True)

    # Inject small missingness to test preprocessing robustness.
    for col in ["temperature", "rainfall"]:
        missing_idx = rng.choice(df.index, size=int(0.01 * len(df)), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df
