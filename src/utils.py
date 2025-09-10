#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py — shared helpers for data IO, feature/window building, model/meta loading, and formatting.

Centralizes functionality used by:
- aws_train_model.py
- backtest.py
- paper_trade.py
- inference.py
- live_trader.py (if needed)

This keeps the codebase consistent and avoids duplication errors.
"""

from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path
from typing import Dict, Generator, List, Optional

import numpy as np
import pandas as pd
import torch

# Optional scaler persistence
try:
    import joblib
except Exception:
    joblib = None

# Optional dotenv
try:
    from dotenv import load_dotenv as _load_dotenv
except Exception:
    def _load_dotenv(*_args, **_kwargs):
        return False


# ---------------------------
# Defaults shared across scripts
# ---------------------------

DEFAULT_FEATURE_COLS = [
    "open", "high", "low", "close",
    "body", "range", "upper_wick", "lower_wick",
    "return", "sma_ratio", "ema_20", "macd", "rsi_14",
    "vol_change", "atr", "price_vs_hourly_trend", "bb_width",
]

PRICE_CANDIDATES = ["close", "adj_close", "adj close", "close_price", "price", "last", "mid", "c"]

DEFAULT_COLS_6 = ["timestamp", "open", "high", "low", "close", "volume"]
DEFAULT_COLS_7 = ["timestamp", "open", "high", "low", "close", "volume", "trades"]


# ---------------------------
# Env / seed utilities
# ---------------------------

def load_dotenv() -> None:
    """Load environment variables from a local .env if present."""
    try:
        _load_dotenv()
    except Exception:
        pass


def set_seed(seed: int) -> None:
    """Reproducible seeding for numpy/torch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------
# CSV header repair & loading
# ---------------------------

def _columns_look_headerless(cols: List[str]) -> bool:
    lowers = [str(c).strip().lower() for c in cols]
    if any(k in lowers for k in ["open", "high", "low", "close", "volume", "timestamp", "time", "c", "o", "h", "l", "v"]):
        return False
    numeric_like = 0
    for c in cols:
        s = str(c).strip().replace(".", "", 1).replace("-", "", 1)
        if s.isdigit():
            numeric_like += 1
    return numeric_like >= max(3, len(cols) // 2)


def _apply_default_headers(df: pd.DataFrame) -> pd.DataFrame:
    n = df.shape[1]
    if n == 6:
        df.columns = DEFAULT_COLS_6
    elif n == 7:
        df.columns = DEFAULT_COLS_7
    else:
        df.columns = [f"col{i}" for i in range(n)]
    return df


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Fix headerless CSVs by assigning default OHLCV columns when needed."""
    if _columns_look_headerless(list(df.columns)):
        df = _apply_default_headers(df)
    return df


def list_csvs_sorted(path: str) -> List[str]:
    """Return sorted list of CSVs if directory, else a single CSV path."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        return files
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if os.path.splitext(path)[1].lower() != ".csv":
        raise ValueError(f"Only .csv supported; got {path}")
    return [path]


def iter_csv_chunks_with_fix(path: str, chunksize: int) -> Generator[pd.DataFrame, None, None]:
    """Yield normalized CSV chunks for very large files."""
    for chunk in pd.read_csv(path, chunksize=chunksize):
        yield normalize_headers(chunk)


def read_csv_concat_sorted(data_dir: str) -> pd.DataFrame:
    """Read all CSVs (or one CSV) and return a single concatenated, normalized DataFrame."""
    p = Path(data_dir)
    files: List[str]
    if p.is_dir():
        files = sorted(glob.glob(str(p / "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {data_dir}")
    else:
        if p.suffix.lower() != ".csv":
            raise ValueError(f"Expected a .csv file or directory, got: {data_dir}")
        files = [str(p)]
    parts = []
    for f in files:
        parts.append(normalize_headers(pd.read_csv(f)))
    return pd.concat(parts, ignore_index=True)


# ---------------------------
# Feature / price helpers
# ---------------------------

def resolve_price_col(columns: List[str], preferred: Optional[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in columns}
    if preferred:
        if preferred in columns:
            return preferred
        if preferred.lower() in lower_map:
            return lower_map[preferred.lower()]
    for cand in PRICE_CANDIDATES:
        if cand in lower_map:
            return lower_map[cand]
    return None


def infer_feature_cols(sample_df: pd.DataFrame, feature_cols: Optional[List[str]], label_col: Optional[str], price_col: Optional[str]) -> List[str]:
    """Infer numeric feature columns if not explicitly provided."""
    if feature_cols:
        return list(feature_cols)
    drop = {c for c in [label_col, price_col, "timestamp", "time"] if c is not None}
    numeric = sample_df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric if c not in drop]
    if not feats:
        raise ValueError("Could not infer numeric feature columns from the sample.")
    return feats


# ---------------------------
# Windows
# ---------------------------

def build_windows_from_flat(features: np.ndarray, seq_len: int) -> np.ndarray:
    """Create overlapping [N-seq_len+1, seq_len, F] windows from [N, F] flat features."""
    N, F = features.shape
    if N < seq_len:
        return np.empty((0, seq_len, F), dtype=np.float32)
    stride0, stride1 = features.strides
    shape = (N - seq_len + 1, seq_len, F)
    strides = (stride0, stride0, stride1)
    return np.lib.stride_tricks.as_strided(features, shape=shape, strides=strides).copy()


def build_windows(features: np.ndarray, seq_len: int) -> np.ndarray:
    """Alias used by some scripts."""
    return build_windows_from_flat(features, seq_len)


# ---------------------------
# Formatting helpers
# ---------------------------

def fmt_money(x, currency: str = "$") -> str:
    """Format a number as money."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    try:
        x = float(x)
    except Exception:
        return str(x)
    if abs(x) >= 1e12:
        return f"{currency}{x:.3e}"
    return f"{currency}{x:,.2f}"


def fmt_pct(x) -> str:
    """Format a fraction as percent with two decimals."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return str(x)


# ---------------------------
# Meta / model loading
# ---------------------------

def load_meta(model_dir_or_path: str) -> Dict:
    """
    Load model_meta.json from either a directory (e.g. 'model')
    or a direct file path (e.g. 'model/model_meta.json').
    """
    p = Path(model_dir_or_path)
    meta_path = p if p.is_file() else (p / "model_meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"model_meta.json not found at {meta_path}")
    with open(meta_path, "r") as f:
        return json.load(f)


def load_model_bundle(model_dir: str):
    """
    Returns (model.eval(), scaler_or_None, meta_dict).
    Requires a models.py with LSTMClassifier/LSTMRegressor in the Python path.
    """
    try:
        from models import build_model_from_meta  # late import
    except Exception as e:
        raise SystemExit(
            "Could not import build_model_from_meta from models.py. Ensure models.py is on PYTHONPATH.\n"
            f"Underlying error: {e}"
        )

    meta = load_meta(model_dir)

    model = build_model_from_meta(meta)

    weights_path = Path(model_dir) / meta.get("model_state_path", "model.pt")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)

    scaler = None
    spath = meta.get("scaler_path", "scaler.joblib")
    if spath and joblib is not None:
        sp = Path(model_dir) / spath
        if sp.exists():
            scaler = joblib.load(sp)

    return model.eval(), scaler, meta


# ---------------------------
# Simple binary label helper (optional)
# ---------------------------

def binary_label_next_bar_up(closes: np.ndarray) -> np.ndarray:
    """
    y[i] = 1 if close[i] > close[i-1], else 0. First element is 0 by construction.
    """
    y = np.zeros_like(closes, dtype=np.int64)
    y[1:] = (closes[1:] > closes[:-1]).astype(np.int64)
    return y


__all__ = [
    # defaults
    "DEFAULT_FEATURE_COLS", "PRICE_CANDIDATES",
    # env / device / seed
    "load_dotenv", "set_seed", "get_device_str",
    # csv utils
    "normalize_headers", "list_csvs_sorted", "iter_csv_chunks_with_fix", "read_csv_concat_sorted",
    # feature / price
    "resolve_price_col", "infer_feature_cols",
    # windows
    "build_windows_from_flat", "build_windows",
    # formatting
    "fmt_money", "fmt_pct",
    # meta/model
    "load_meta", "load_model_bundle",
    # labels
    "binary_label_next_bar_up",
]
