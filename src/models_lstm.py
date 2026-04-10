"""LSTM model training and inference for demand forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class LSTMArtifacts:
    scaler: MinMaxScaler
    seq_len: int
    feature_cols: list[str]


def build_lstm_model(input_shape: Tuple[int, int]) -> Any:
    """Create a compact LSTM network."""
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential(
        [
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def create_lstm_sequences(df: pd.DataFrame, feature_cols: list[str], seq_len: int = 24):
    """Build SKU-wise sequences with strict temporal ordering."""
    X_list = []
    y_list = []
    meta_list = []

    for sku, group in df.groupby("product_id", sort=False):
        values = group[feature_cols + ["sales"]].to_numpy()
        for i in range(seq_len, len(values)):
            X_list.append(values[i - seq_len : i, :-1])
            y_list.append(values[i, -1])
            meta_list.append((sku, group.iloc[i]["datetime"]))

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_list, columns=["product_id", "datetime"])
    return X, y, meta


def scale_lstm_features(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]):
    """Scale continuous variables for stable LSTM training."""
    scaler = MinMaxScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_scaled, test_scaled, scaler


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 8,
    batch_size: int = 64,
) -> Any:
    """Train LSTM with early stopping."""
    from tensorflow.keras.callbacks import EarlyStopping

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    callbacks = [EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return model


def save_lstm_model(model: Any, path: Path) -> None:
    """Persist trained LSTM model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
