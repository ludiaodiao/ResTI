"""utils.py

Minimal data utilities used by `train.py`.

This repo builds a *middle-gap imputation / reconstruction* task:

- Take a sliding window of length (right_len + impu_len + left_len).
- Input x:
    * bidirectional flow: concat(first right_len, last left_len)  -> length right_len + left_len
    * causal flow:        take first (right_len + left_len)       -> length right_len + left_len
- Target y: the middle segment of length impu_len.

Notes
-----
- Datasets are CSV files under `./dataset/`.
- Column 0 is typically a datetime string. Channels start from column 1.
  (So channel=1 corresponds to the first numeric variable.)

"""

from __future__ import annotations

import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """(x,y) dataset where x and y are 1D sequences."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # model expects [B, T, C], where C=1 for a single channel
        self.x = torch.tensor(x, dtype=torch.float32).unsqueeze(2)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(2)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def _construct_pairs(
    ts_1d: np.ndarray,
    right_len: int,
    left_len: int,
    impu_len: int,
    is_use_bidirectional_flow: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding-window samples.

    Parameters match the naming in the original code:
    - right_len (rl): the first chunk length taken from the window head
    - left_len  (ll): the last chunk length taken from the window tail
    """
    ts_1d = np.asarray(ts_1d).reshape(-1)

    if is_use_bidirectional_flow:
        # "left and right without causality"
        seq_len = right_len + impu_len + left_len
        x_list, y_list = [], []
        for i in range(len(ts_1d) - seq_len + 1):
            sub = ts_1d[i : i + seq_len]
            x = np.concatenate((sub[:right_len], sub[-left_len:]))
            y = sub[right_len : right_len + impu_len]
            x_list.append(x)
            y_list.append(y)
        return np.asarray(x_list), np.asarray(y_list)

    # "left" (causal)
    seq_len = right_len + left_len + impu_len
    x_list, y_list = [], []
    for i in range(len(ts_1d) - seq_len + 1):
        sub = ts_1d[i : i + seq_len]
        x = sub[: right_len + left_len]
        y = sub[right_len + left_len : right_len + left_len + impu_len]
        x_list.append(x)
        y_list.append(y)

    return np.asarray(x_list), np.asarray(y_list)


def _dataset_csv_path(dataset_name: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "dataset", f"{dataset_name}.csv")


def get_dataloader(
    DATASET: str = "ETTh1",
    channel: int = 1,
    left_len: int = 96,
    impu_len: int = 96,
    right_len: int = 96,
    is_use_bidirectional_flow: bool = True,
    data_use_rate: float = 1.0,
):
    """Return (train_dataset, test_dataset).

    - Train/test split is chronological: first 80% for train, last 20% for test.
    - Standardization uses ONLY train statistics to avoid leakage.
    """

    csv_path = _dataset_csv_path(DATASET)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}. Put the CSV under ./dataset/{DATASET}.csv"
        )

    # Keep the original behavior: treat the first line as header, and index columns by integer.
    # This makes `channel=1` match the first numeric column for ETTh1 / powerconsumption.
    series = pd.read_csv(csv_path, header=None, skiprows=1)[channel].values

    # Convert to numeric when possible; drop NaNs (e.g., if someone accidentally selects a datetime column)
    series = pd.to_numeric(series, errors="coerce")
    series = series[np.isfinite(series)].astype(np.float32)

    if not (0 < float(data_use_rate) <= 1.0):
        raise ValueError(f"data_use_rate must be in (0,1], got {data_use_rate}")
    if float(data_use_rate) < 1.0:
        use_len = int(len(series) * float(data_use_rate))
        series = series[:use_len]

    # chronological split
    split_idx = int(len(series) * 0.8)
    ts_train = series[:split_idx].reshape(-1, 1)
    ts_test = series[split_idx:].reshape(-1, 1)

    scaler = StandardScaler()
    ts_train = scaler.fit_transform(ts_train).reshape(-1)
    ts_test = scaler.transform(ts_test).reshape(-1)

    X_train, y_train = _construct_pairs(
        ts_train, right_len=right_len, left_len=left_len, impu_len=impu_len,
        is_use_bidirectional_flow=is_use_bidirectional_flow
    )
    X_test, y_test = _construct_pairs(
        ts_test, right_len=right_len, left_len=left_len, impu_len=impu_len,
        is_use_bidirectional_flow=is_use_bidirectional_flow
    )

    if X_train.size == 0 or X_test.size == 0:
        raise RuntimeError(
            f"No samples were constructed (train={X_train.shape}, test={X_test.shape}). "
            f"Try reducing left/right/impu lengths or using a larger data_use_rate."
        )

    return TimeSeriesDataset(X_train, y_train), TimeSeriesDataset(X_test, y_test)
