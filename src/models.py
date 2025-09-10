"""
Unified LSTMClassifier definition and helpers used across training, backtesting, and inference.

Import this everywhere instead of redefining the model:
    from models import LSTMClassifier, build_model_from_meta, load_scaler

The design is intentionally minimal and framework-agnostic, but robust enough for
your training/backtest/inference use cases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Union


import torch
import torch.nn as nn

try:
    # Prefer joblib (faster, more common for sklearn objects)
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # Fallback handled in load_scaler()


# ----------------------------
# Model definition
# ----------------------------

class LSTMClassifier(nn.Module):
    """
    Generic LSTM classifier that outputs logits for N classes.
    Expects input of shape: [batch, seq_len, input_size] (batch_first=True).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        num_classes: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "num_classes": num_classes,
        }

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        direction_factor = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * direction_factor),
            nn.Linear(hidden_size * direction_factor, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        # We take the last hidden state for classification (common approach)
        output, (hn, cn) = self.lstm(x)  # output: [B, T, H*D]
        last = output[:, -1, :]          # [B, H*D]
        logits = self.head(last)         # [B, C]
        return logits


class LSTMRegressor(nn.Module):
    """
    Generic LSTM regressor that outputs predicted value (e.g., price change).
    Expects input of shape: [batch, seq_len, input_size] (batch_first=True).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        output_size: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "output_size": output_size,
        }

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        direction_factor = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * direction_factor),
            nn.Linear(hidden_size * direction_factor, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        # We take the last hidden state for regression
        output, (hn, cn) = self.lstm(x)  # output: [B, T, H*D]
        last = output[:, -1, :]          # [B, H*D]
        pred = self.head(last)           # [B, output_size]
        return pred


# ----------------------------
# Metadata utilities
# ----------------------------

@dataclass
class ModelMeta:
    # Architecture
    input_size: int
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False
    num_classes: int = 3
    output_size: int = 1  # For regressor

    # Artifacts (relative to model dir unless absolute)
    model_state_path: str = "model.pt"
    scaler_path: Optional[str] = "scaler.joblib"

    # Optional flags
    feature_scaling: bool = True  # If false, scaler_path may be None
    framework: str = "pytorch"
    model_type: str = "lstm_classifier"

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ModelMeta":
        # Accept flexible keys and sensible fallbacks
        return ModelMeta(
            input_size=int(d.get("input_size") or d.get("n_features") or d["features"]),
            hidden_size=int(d.get("hidden_size", 128)),
            num_layers=int(d.get("num_layers", 2)),
            dropout=float(d.get("dropout", 0.1)),
            bidirectional=bool(d.get("bidirectional", False)),
            num_classes=int(d.get("num_classes", d.get("classes", 3))),
            output_size=int(d.get("output_size", 1)),
            model_state_path=str(d.get("model_state_path", d.get("weights", "model.pt"))),
            scaler_path=d.get("scaler_path", d.get("scaler", "scaler.joblib")),
            feature_scaling=bool(d.get("feature_scaling", d.get("scale_features", True))),
            framework=str(d.get("framework", "pytorch")),
            model_type=str(d.get("model_type", "lstm_classifier")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_model_from_meta(meta: Union[ModelMeta, Dict[str, Any]]) -> Union[LSTMClassifier, LSTMRegressor]:
    """
    Build an LSTM model instance from metadata dict or ModelMeta object.
    """
    if not isinstance(meta, ModelMeta):
        meta = ModelMeta.from_dict(meta)

    if meta.model_type == "lstm_regressor":
        model = LSTMRegressor(
            input_size=meta.input_size,
            hidden_size=meta.hidden_size,
            num_layers=meta.num_layers,
            dropout=meta.dropout,
            bidirectional=meta.bidirectional,
            output_size=meta.output_size,
        )
    else:  # default to classifier
        model = LSTMClassifier(
            input_size=meta.input_size,
            hidden_size=meta.hidden_size,
            num_layers=meta.num_layers,
            dropout=meta.dropout,
            bidirectional=meta.bidirectional,
            num_classes=meta.num_classes,
        )
    return model


# ----------------------------
# Artifact loading helpers
# ----------------------------

def resolve_path(base_dir: str, path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    import os
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def load_meta(meta_path: str) -> ModelMeta:
    with open(meta_path, "r") as f:
        data = json.load(f)
    return ModelMeta.from_dict(data)


def load_model_state(model: nn.Module, state_path: str, strict: bool = False) -> None:
    state = torch.load(state_path, map_location="cpu")
    # Allow non-strict to tolerate minor key mismatches or buffer names
    model.load_state_dict(state, strict=strict)


def load_scaler(scaler_path: Optional[str]):
    """
    Loads an sklearn-like scaler (StandardScaler/MinMax/etc) saved via joblib or pickle.
    Returns None if path is None. Users should check for None before applying.
    """
    if scaler_path is None:
        return None

    import os
    import pickle

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at: {scaler_path}")

    # Try joblib first
    if joblib is not None:
        try:
            return joblib.load(scaler_path)
        except Exception:
            pass

    # Fallback to pickle
    with open(scaler_path, "rb") as f:
        return pickle.load(f)
