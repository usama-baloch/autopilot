#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference.py — SageMaker-compatible inference script.

Loads model_meta.json, model.pt, scaler.joblib from model_dir,
accepts JSON or CSV, builds windows, returns probabilities for class=1 (buy).
"""

from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

# Optional deps
try:
    import pandas as pd
except Exception:
    pd = None

try:
    import joblib
except Exception:
    joblib = None

# Project utils/models
from utils import load_meta, build_windows, DEFAULT_FEATURE_COLS
from models import LSTMClassifier


# ---------- SageMaker entrypoints ----------

def model_fn(model_dir: str):
    """
    Load artifacts from model_dir.
    Returns a tuple (model.eval(), scaler_or_None, meta_dict).
    """
    meta = load_meta(model_dir)
    feature_cols: List[str] = list(meta.get("feature_cols", DEFAULT_FEATURE_COLS))
    hidden_size = int(meta.get("hidden_size", 512))
    num_layers = int(meta.get("num_layers", 3))
    dropout = float(meta.get("dropout", 0.3))
    bidirectional = bool(meta.get("bidirectional", True))
    num_classes = int(meta.get("num_classes", 2))

    model = LSTMClassifier(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        num_classes=num_classes,
    )

    weights_path = os.path.join(model_dir, meta.get("model_state_path", "model.pt"))
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    scaler = None
    spath = meta.get("scaler_path", "scaler.joblib")
    if spath and joblib is not None:
        sfile = os.path.join(model_dir, spath)
        if os.path.exists(sfile):
            scaler = joblib.load(sfile)

    return (model, scaler, meta)


def _json_to_matrix(payload: Dict[str, Any], feature_cols: List[str]) -> np.ndarray:
    """
    Accept either:
      {"rows":[{feature:value,...}, ...]}
      {"instances":[[f1,f2,...], ...]}
    Returns np.ndarray [N, F] ordered by feature_cols.
    """
    if "rows" in payload and isinstance(payload["rows"], list):
        rows = payload["rows"]
        if not rows:
            return np.zeros((0, len(feature_cols)), dtype=np.float32)
        out = []
        for r in rows:
            out.append([float(r[c]) for c in feature_cols])
        return np.array(out, dtype=np.float32)

    if "instances" in payload and isinstance(payload["instances"], list):
        arr = np.array(payload["instances"], dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("'instances' must be 2D: [N, F]")
        if arr.shape[1] != len(feature_cols):
            raise ValueError(f"instances columns ({arr.shape[1]}) != required features ({len(feature_cols)})")
        return arr

    raise ValueError("JSON must contain 'rows' (list of dicts) or 'instances' (2D list).")


def _csv_to_matrix(body: str, feature_cols: List[str]) -> np.ndarray:
    """
    CSV with header including feature columns; extra columns ignored but must contain at least all needed features.
    """
    if pd is None:
        raise RuntimeError("pandas is required for CSV input. Install pandas or send JSON instead.")
    df = pd.read_csv(io.StringIO(body))
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required features: {missing}")
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    return X


def input_fn(request_body: str, content_type: str):
    """
    Parse incoming request to a flat features matrix [N, F].
    """
    if not request_body:
        raise ValueError("Empty request body.")

    content_type = (content_type or "application/json").lower().strip()
    input_obj: Dict[str, Any] = {"flat": None, "content_type": content_type}

    if content_type == "application/json":
        payload = json.loads(request_body)
        input_obj["payload"] = payload
        return input_obj

    if content_type in ("text/csv", "application/csv"):
        input_obj["csv"] = request_body
        return input_obj

    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_object: Dict[str, Any], model_bundle):
    """
    Build windows and predict class-1 probabilities.
    Output dict includes probs and hard labels using buy_threshold from meta.
    """
    model, scaler, meta = model_bundle
    feature_cols: List[str] = list(meta.get("feature_cols", DEFAULT_FEATURE_COLS))
    window_size = int(meta.get("window_size", 150))
    threshold = float(meta.get("buy_threshold", 0.60))

    # Parse input into flat matrix [N,F]
    if "payload" in input_object:
        X_flat = _json_to_matrix(input_object["payload"], feature_cols)
    elif "csv" in input_object:
        X_flat = _csv_to_matrix(input_object["csv"], feature_cols)
    else:
        raise ValueError("Missing payload.")

    if X_flat.shape[0] < window_size:
        # Not enough rows to form a single window
        return {"probs": [], "labels": [], "threshold": threshold}

    # Build windows and scale
    X = build_windows(X_flat, window_size)  # [W, T, F]
    if scaler is not None:
        W, T, F = X.shape
        X = scaler.transform(X.reshape(W * T, F)).reshape(W, T, F)

    # Predict
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    with torch.no_grad():
        xb = torch.from_numpy(X).to(device)
        output = model(xb)
        if meta.get("model_type") == "lstm_regressor":
            preds = output.detach().cpu().numpy().flatten()
            labels = (preds >= threshold).astype(int).tolist()
            return {"preds": preds.tolist(), "labels": labels, "threshold": threshold}
        else:
            logits = output  # [W, 2]
            probs = F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            labels = (probs >= threshold).astype(int).tolist()
            return {"probs": probs.tolist(), "labels": labels, "threshold": threshold}


def output_fn(prediction: Dict[str, Any], accept: str = "application/json"):
    """
    Format the response.
    """
    accept = (accept or "application/json").lower().strip()
    body = json.dumps(prediction)
    if accept in ("application/json", "text/json"):
        return body, accept
    # Fallback to JSON
    return body, "application/json"
