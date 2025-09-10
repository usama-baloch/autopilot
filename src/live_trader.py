#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_trade.py — live loop with optional REAL Binance orders (spot via ccxt).

Defaults:
  • Data directory: ./eth_1m_data (growing CSVs)
  • Model directory: ./  (model_meta.json, model.pt, scaler.joblib)
  • One-position-at-a-time; TP/SL checked intra-bar each iteration.
  • Paper mode by default. Pass --real true to place real market orders on Binance.

Env (.env):
  BINANCE_API_KEY=...
  BINANCE_API_SECRET=...
  # optional
  BINANCE_TESTNET=true   # use Binance sandbox if supported by ccxt (safer for testing)

Examples
--------
# Paper (safe) mode:
python live_trade.py --threshold 0.62

# Real mode (actual orders):
python live_trade.py --real true --symbol ETH/USDT --alloc-pct 0.10 --threshold 0.62
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# project utils/models
from utils import (
    read_csv_concat_sorted, resolve_price_col, build_windows,
    load_model_bundle, fmt_money, load_dotenv
)

# ---- Optional real-trading deps (only required if --real true) ----
_EXCHANGE_OK = True
try:
    import ccxt  # for Binance real orders
except Exception:
    _EXCHANGE_OK = False


# ---------------------------
# Data helpers
# ---------------------------

def _load_latest_df(data_dir: str) -> pd.DataFrame:
    df = read_csv_concat_sorted(data_dir)
    if "timestamp" in df.columns:
        try:
            df = df.sort_values("timestamp")
        except Exception:
            pass
    return df.reset_index(drop=True)

def _align_ohlc(df: pd.DataFrame, price_col: str, window_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    opens = df["open"].to_numpy(dtype=float)[window_size - 1:]
    highs = df["high"].to_numpy(dtype=float)[window_size - 1:]
    lows  = df["low"].to_numpy(dtype=float)[window_size - 1:]
    closes= df[price_col].to_numpy(dtype=float)[window_size - 1:]
    return opens, highs, lows, closes

def _predict_probs(model, scaler, X: np.ndarray, device: str, meta: dict) -> np.ndarray:
    with torch.no_grad():
        xb = torch.from_numpy(X).to(device)
        output = model(xb)
        if meta.get("model_type") == "lstm_regressor":
            # For regressor, output is the predicted return
            return output.detach().cpu().numpy().flatten()
        else:
            # For classifier, output is logits
            p = F.softmax(output, dim=-1)[:, 1].detach().cpu().numpy()
            return p


def _forecast_multi_step(model, scaler, X_last: np.ndarray, device: str, meta: dict, steps: int = 3) -> np.ndarray:
    """
    Forecast multiple steps ahead by iteratively predicting and updating the window.
    Assumes regressor model.
    """
    if meta.get("model_type") != "lstm_regressor":
        raise ValueError("Multi-step forecasting requires regressor model")

    window_size = int(meta.get("window_size", 150))
    feature_cols = meta.get("feature_cols", [])
    price_col_idx = feature_cols.index(meta.get("price_col", "close")) if meta.get("price_col") in feature_cols else -1

    forecasts = []
    current_window = X_last[-1].copy()  # Last window [T, F]

    for step in range(steps):
        with torch.no_grad():
            xb = torch.from_numpy(current_window[np.newaxis, :, :]).to(device)  # [1, T, F]
            pred_return = model(xb).item()

        forecasts.append(pred_return)

        # Update window: shift left, append new features
        # For simplicity, assume features are updated with predicted price
        if price_col_idx >= 0:
            current_price = current_window[-1, price_col_idx]
            new_price = current_price * (1 + pred_return)
            new_window = np.roll(current_window, -1, axis=0)
            new_window[-1] = current_window[-1].copy()  # Copy last features
            new_window[-1, price_col_idx] = new_price  # Update price
            # Update return feature if present
            return_idx = feature_cols.index("return") if "return" in feature_cols else -1
            if return_idx >= 0:
                new_window[-1, return_idx] = pred_return
            current_window = new_window

    return np.array(forecasts)


# ---------------------------
# Binance helpers (ccxt)
# ---------------------------

def _init_binance_from_env(testnet_env: Optional[str] = None):
    """
    Initialize ccxt.binance using .env credentials.
    If BINANCE_TESTNET=true, enable sandbox mode (where supported).
    """
    if not _EXCHANGE_OK:
        raise SystemExit("ccxt is not installed. Install it with: pip install ccxt")

    api_key = os.getenv("BINANCE_API_KEY", "").strip()
    api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise SystemExit("Missing BINANCE_API_KEY / BINANCE_API_SECRET in environment (.env)")

    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",  # spot trading
        },
    })

    testnet_flag = (testnet_env or os.getenv("BINANCE_TESTNET", "false")).lower() in ("1", "true", "yes")
    if testnet_flag:
        # Enable sandbox mode (ccxt maps to testnet endpoints for supported exchanges)
        exchange.set_sandbox_mode(True)

    # Load markets for precision/limits
    exchange.load_markets()
    return exchange, testnet_flag

def _get_symbol_info(exchange, symbol: str) -> Dict:
    if symbol not in exchange.markets:
        # Try upper normalization
        sym = symbol.upper()
        if sym in exchange.markets:
            symbol = sym
        else:
            raise SystemExit(f"Symbol {symbol} not found on Binance.")
    return exchange.markets[symbol]

def _quantize_amount(exchange, symbol: str, amount: float) -> float:
    """
    Clip/round amount to symbol precision/limits to avoid INVALID_QUANTITY.
    """
    market = _get_symbol_info(exchange, symbol)
    precision = market.get("precision", {}).get("amount", None)
    step = None
    limits = market.get("limits", {})
    amount_min = limits.get("amount", {}).get("min", None)
    amount_max = limits.get("amount", {}).get("max", None)

    if precision is not None:
        amt = float(round(amount, int(precision)))
    else:
        amt = float(amount)

    if amount_min is not None:
        amt = max(amt, float(amount_min))
    if amount_max is not None:
        amt = min(amt, float(amount_max))

    # Additional safety: Binance often has step-size filters. ccxt abstracts many checks,
    # but if you need exact step rounding you can inspect market['info']['filters'].

    return float(amt)

def _fetch_balances(exchange) -> Dict[str, float]:
    bal = exchange.fetch_balance()
    # Spot balances: free balances under 'free'
    # We return a lightweight view
    free = bal.get("free", {})
    return {k: float(v) for k, v in free.items() if v is not None}

def _fetch_ticker_price(exchange, symbol: str) -> float:
    t = exchange.fetch_ticker(symbol)
    # Prefer 'last' then 'close' then 'ask' as fallback
    for k in ("last", "close", "ask", "bid"):
        v = t.get(k, None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    raise RuntimeError("Could not determine price from ticker.")


# ---------------------------
# Main
# ---------------------------

def main():
    load_dotenv()

    ap = argparse.ArgumentParser(description="Live trading loop (paper by default; real via Binance with --real true).")
    ap.add_argument("--data-dir", type=str, default="eth_1m_data", help="Dir of rolling CSVs or a single CSV")
    ap.add_argument("--model-dir", type=str, default=".", help="Where model_meta.json & model.pt live")
    ap.add_argument("--capital", type=float, default=10_000.0, help="Starting equity for paper mode")
    ap.add_argument("--threshold", type=float, default=None, help="Override buy threshold; default from model_meta.json")
    ap.add_argument("--tp-pct", type=float, default=None, help="Take-profit fraction (e.g. 0.005 = 0.5%)")
    ap.add_argument("--sl-pct", type=float, default=None, help="Stop-loss fraction (e.g. 0.0025 = 0.25%)")
    ap.add_argument("--fee-pct", type=float, default=None, help="Per-side fee (paper mode), e.g. 0.0008 = 0.08%")
    ap.add_argument("--interval", type=float, default=5.0, help="Seconds between checks")
    ap.add_argument("--min-new-bars", type=int, default=1, help="Trigger when at least this many new bars appear")
    ap.add_argument("--quiet", action="store_true", help="Less verbose prints")

    # Real trading args
    ap.add_argument("--real", type=str, default="false", help="Set to 'true' to place REAL Binance market orders")
    ap.add_argument("--symbol", type=str, default="ETH/USDT", help="Trading pair on Binance")
    ap.add_argument("--alloc-pct", type=float, default=0.10, help="Fraction of available QUOTE balance to allocate per entry (REAL mode)")
    args = ap.parse_args()

    # Load model + meta
    model, scaler, meta = load_model_bundle(args.model_dir)
    is_regressor = meta.get("model_type") == "lstm_regressor"
    feature_cols = list(meta.get("feature_cols", []))
    window_size = int(meta.get("window_size", 150))
    if is_regressor:
        threshold = float(meta.get("buy_threshold", 0.001)) if args.threshold is None else float(args.threshold)  # return threshold
    else:
        threshold = float(meta.get("buy_threshold", 0.60)) if args.threshold is None else float(args.threshold)
    fee_pct = float(meta.get("tx_cost", 0.0008)) if args.fee_pct is None else float(args.fee_pct)
    tp_pct = 0.005 if args.tp_pct is None else float(args.tp_pct)
    sl_pct = 0.0025 if args.sl_pct is None else float(args.sl_pct)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Determine mode
    real_mode = str(args.real).lower() in ("1", "true", "yes")

    # Live state
    cash = float(args.capital)  # used in paper mode
    in_trade = False
    entry_price = None
    tp_price = None
    sl_price = None
    last_n = 0

    # REAL TRADING STATE
    exchange = None
    base_ccy = None
    quote_ccy = None
    held_amount = 0.0  # qty of BASE we hold (REAL mode)
    if real_mode:
        exchange, testnet_flag = _init_binance_from_env(os.getenv("BINANCE_TESTNET"))
        # Parse symbol to base/quote (ccxt provides market info)
        m = _get_symbol_info(exchange, args.symbol)
        base_ccy = m["base"]
        quote_ccy = m["quote"]
        if not args.quiet:
            print(f"[REAL] Binance connected. symbol={args.symbol} base={base_ccy} quote={quote_ccy} testnet={testnet_flag}")
            print(f"[REAL] Entry sizing: alloc-pct={args.alloc_pct:.2%} of available {quote_ccy}")

    metric = "return" if is_regressor else "prob"
    print(f"[LIVE] Mode: {'REAL' if real_mode else 'PAPER'} | threshold={threshold:.4f} ({metric}) tp={tp_pct:.4%} sl={sl_pct:.4%} "
          f"{'' if real_mode else f'fee={fee_pct:.4%}'}")

    while True:
        try:
            # ---- Load latest data and prepare window ----
            df = _load_latest_df(args.data_dir)
            price_col = resolve_price_col(df.columns.tolist(), meta.get("price_col", "close"))
            if price_col is None:
                if not args.quiet:
                    print("[LIVE] price column not found in data; sleeping.")
                time.sleep(args.interval)
                continue

            drop_cols = {price_col, "timestamp", "time"}
            feat_cols = [c for c in feature_cols if c in df.columns and c not in drop_cols]
            if len(feat_cols) < len(feature_cols):
                missing = [c for c in feature_cols if c not in feat_cols]
                if not args.quiet:
                    print(f"[LIVE] Missing features (skipping): {missing}")
            if len(df) < window_size + 1 or len(feat_cols) == 0:
                if not args.quiet:
                    print("[LIVE] Waiting for enough rows/features...")
                time.sleep(args.interval)
                continue

            X_flat = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
            X = build_windows(X_flat, window_size)
            if scaler is not None:
                W, T, F = X.shape
                X = scaler.transform(X.reshape(W * T, F)).reshape(W, T, F)
            if X.shape[0] < args.min_new_bars:
                time.sleep(args.interval)
                continue

            opens, highs, lows, closes = _align_ohlc(df, price_col, window_size)
            new_W = X.shape[0]
            if new_W <= last_n + (args.min_new_bars - 1):
                time.sleep(args.interval)
                continue

            probs = _predict_probs(model, scaler, X, device, meta)
            prob_last = float(probs[-1])

            # For regressor, forecast 1-3 minutes
            if is_regressor:
                forecasts = _forecast_multi_step(model, scaler, X, device, meta, steps=3)
                forecast_1m = forecasts[0]
                forecast_3m = np.prod(1 + forecasts) - 1  # Cumulative return over 3 minutes
            else:
                forecasts = None
                forecast_1m = None
                forecast_3m = None

            # ---- Trading logic ----
            if not in_trade:
                if (is_regressor and prob_last > threshold) or (not is_regressor and prob_last >= threshold):
                    # ENTRY
                    if real_mode:
                        # Determine notional using alloc-pct of available quote balance
                        price_now = _fetch_ticker_price(exchange, args.symbol)
                        bals = _fetch_balances(exchange)
                        quote_free = float(bals.get(quote_ccy, 0.0))
                        if quote_free <= 0:
                            if not args.quiet:
                                print(f"[REAL] No available {quote_ccy} balance to buy.")
                        else:
                            notional = max(0.0, quote_free * float(args.alloc_pct))
                            qty_raw = notional / price_now if price_now > 0 else 0.0
                            qty = _quantize_amount(exchange, args.symbol, qty_raw)
                            if qty <= 0:
                                if not args.quiet:
                                    print(f"[REAL] Computed qty too small after precision/limits. notional={notional}, price={price_now}")
                            else:
                                # MARKET BUY
                                order = exchange.create_market_buy_order(args.symbol, qty)
                                entry_price = float(order.get("price") or price_now)
                                held_amount = float(order.get("amount") or qty)
                                tp_price = entry_price * (1.0 + tp_pct)
                                sl_price = entry_price * (1.0 - sl_pct)
                                in_trade = True
                                if not args.quiet:
                                    oid = order.get("id", "—")
                                    if is_regressor:
                                        val_str = f"return={prob_last:.4f} 1m={forecast_1m:.4f} 3m={forecast_3m:.4f}"
                                    else:
                                        val_str = f"prob={prob_last:.3f}"
                                    print(f"[REAL] BUY  {args.symbol} qty={held_amount} @ {fmt_money(entry_price)}  ({val_str}) id={oid}")
                    else:
                        # PAPER: enter at next bar open (conservative)
                        entry_price = float(opens[-1])
                        tp_price = entry_price * (1.0 + tp_pct)
                        sl_price = entry_price * (1.0 - sl_pct)
                        cash *= (1.0 - float(os.getenv("PAPER_FEE_PCT", fee_pct)))  # pay entry fee
                        in_trade = True
                        if not args.quiet:
                            if is_regressor:
                                val_str = f"return={prob_last:.4f} 1m={forecast_1m:.4f} 3m={forecast_3m:.4f}"
                            else:
                                val_str = f"prob={prob_last:.3f}"
                            print(f"[PAPER] BUY  @ {fmt_money(entry_price)}  ({val_str})  equity={fmt_money(cash)}")

            else:
                # EXIT checks using current bar hi/lo
                h = float(highs[-1])
                l = float(lows[-1])
                exit_px = None
                win = None

                # conservative: SL first if both crossed
                if l <= sl_price <= h:
                    exit_px = sl_price
                    win = False
                elif h >= tp_price:
                    exit_px = tp_price
                    win = True
                elif l <= sl_price:
                    exit_px = sl_price
                    win = False

                if exit_px is not None:
                    if real_mode:
                        # MARKET SELL
                        if held_amount > 0:
                            # Quantize again in case filters changed
                            qty_out = _quantize_amount(exchange, args.symbol, held_amount)
                            order = exchange.create_market_sell_order(args.symbol, qty_out)
                            px = float(order.get("price") or exit_px)
                            if not args.quiet:
                                oid = order.get("id", "—")
                                print(f"[REAL] SELL {args.symbol} qty={qty_out} @ {fmt_money(px)}  {'WIN' if win else 'LOSS'} id={oid}")
                        else:
                            if not args.quiet:
                                print("[REAL] Warning: held amount was zero at exit.")
                        in_trade = False
                        entry_price = tp_price = sl_price = None
                        held_amount = 0.0
                    else:
                        # PAPER: realize PnL and pay exit fee
                        gross_ret = (exit_px / entry_price) - 1.0
                        cash *= (1.0 + gross_ret)
                        cash *= (1.0 - float(os.getenv("PAPER_FEE_PCT", fee_pct)))
                        if not args.quiet:
                            print(f"[PAPER] SELL @ {fmt_money(exit_px)}  {'WIN' if win else 'LOSS'}  equity={fmt_money(cash)}")
                        in_trade = False
                        entry_price = tp_price = sl_price = None

            last_n = new_W
            time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\n[LIVE] Stopped by user.")
            break
        except Exception as e:
            print(f"[LIVE] Error: {e}")
            time.sleep(max(1.0, args.interval))

    if not real_mode:
        print(f"[PAPER] Final equity: {fmt_money(cash)}")
    else:
        print("[REAL] Loop ended.")

if __name__ == "__main__":
    main()