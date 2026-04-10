"""Evaluation metrics and visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error with zero-safe denominator."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(y_true == 0, 1, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Common regression metrics for demand forecasting."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
    }


def run_eda_plots(df: pd.DataFrame, plot_dir: Path) -> None:
    """Generate required EDA charts."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Sales over time (aggregate).
    plt.figure(figsize=(14, 4))
    ts = df.groupby("datetime", as_index=False)["sales"].sum()
    sns.lineplot(data=ts, x="datetime", y="sales")
    plt.title("Total Sales Over Time")
    plt.tight_layout()
    plt.savefig(plot_dir / "sales_over_time.png")
    plt.close()

    # Hourly pattern.
    plt.figure(figsize=(10, 4))
    hourly = df.groupby("hour", as_index=False)["sales"].mean()
    sns.lineplot(data=hourly, x="hour", y="sales", marker="o")
    plt.title("Average Hourly Demand Pattern")
    plt.tight_layout()
    plt.savefig(plot_dir / "hourly_pattern.png")
    plt.close()

    # Weekly trends.
    plt.figure(figsize=(10, 4))
    weekly = df.groupby("day_of_week", as_index=False)["sales"].mean()
    sns.barplot(data=weekly, x="day_of_week", y="sales")
    plt.title("Average Demand by Day of Week")
    plt.tight_layout()
    plt.savefig(plot_dir / "weekly_trends.png")
    plt.close()

    # Product-wise distribution.
    plt.figure(figsize=(11, 4))
    sns.boxplot(data=df, x="product_id", y="sales")
    plt.title("Product-wise Demand Distribution")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(plot_dir / "product_distribution.png")
    plt.close()


def plot_actual_vs_predicted(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
    n_points: int = 400,
) -> None:
    """Plot actual and predicted demand for visual comparison."""
    plt.figure(figsize=(14, 4))
    if len(timestamps) > n_points:
        timestamps = timestamps.iloc[:n_points]
        y_true = y_true[:n_points]
        y_pred = y_pred[:n_points]
    plt.plot(timestamps, y_true, label="Actual", linewidth=1.5)
    plt.plot(timestamps, y_pred, label="Predicted", linewidth=1.5)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
