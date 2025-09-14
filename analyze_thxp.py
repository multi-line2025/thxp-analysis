#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THxP per-ticker analysis (CLI)
- Reads CSV files (glob supported) with timestamp, ticker/symbol, amount columns.
- Applies exponential decay with configurable half-life (default: 30 minutes).
- Computes per-ticker time series, outputs value/derivative/integral.
- Saves summary CSV, per-ticker CSVs, and plots (value/derivative/integral).
"""
# （ここに前回書いた完全版の analyze_thxp.py の中身を入れてください）

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THxP per-ticker analysis (CLI)
- Reads 1+ CSV files (glob supported) that include timestamp, ticker/symbol, amount columns.
- Applies exponential decay with a configurable half-life (default: 30 minutes).
- Builds per-ticker time series, and outputs for each ticker:
    - value (decayed)
    - derivative_per_sec
    - integral_value
- Saves summary CSV across all tickers and per-ticker detailed CSVs.
- Saves plots (value / derivative / integral) for top-N tickers by current value.

Usage examples:
    python analyze_thxp.py --input data/*.csv --half-life-min 30 --freq 5T --top 3 --outdir outputs

Notes:
- Assumes timezone Asia/Tokyo for timestamps unless already tz-aware; then converts to Tokyo.
- Matplotlib: one plot per figure; no explicit colors or styles.
"""
import argparse
import glob
import os
import re
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytz

TOKYO = pytz.timezone("Asia/Tokyo")

# ---------- Column detection ----------
def pick_col(cols, candidates):
    cols_lower = {c.lower(): c for c in cols}
    for cand_list in candidates:
        for cand in cand_list:
            if cand in cols_lower:
                return cols_lower[cand]
    return None

def ensure_datetime(s, tz=TOKYO):
    dt = pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
    if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
        dt = dt.dt.tz_convert(tz)
    else:
        dt = dt.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    return dt

def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))
    return s[:80] if len(s) > 80 else s

def load_and_normalize(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try_enc = [None, "cp932", "utf-8-sig"]
        last_err = None
        for enc in try_enc:
            try:
                df = pd.read_csv(p, encoding=enc) if enc else pd.read_csv(p)
                frames.append(df)
                break
            except Exception as e:
                last_err = e
        if len(frames) == 0 and last_err is not None:
            raise last_err

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    timestamp_candidates = [["timestamp"],["time"],["datetime"],["block_time"],["created_at"],["date"],["tx_time"]]
    ticker_candidates = [["symbol"],["ticker"],["token"],["asset"],["name"],["銘柄"],["brand"]]
    amount_candidates = [["amount"],["qty"],["quantity"],["value"],["thxp"],["points"],["数"],["数量"],["金額"]]
    from_candidates = [["from"],["sender"],["from_address"],["source"],["送信者"]]
    to_candidates = [["to"],["receiver"],["to_address"],["destination"],["受信者"]]

    ts_col = pick_col(df.columns, timestamp_candidates) or next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)
    ticker_col = pick_col(df.columns, ticker_candidates)
    if ticker_col is None:
        non_numeric_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        non_time_cols = [c for c in non_numeric_cols if c != ts_col]
        ticker_col = non_time_cols[0] if non_time_cols else None
    amt_col = pick_col(df.columns, amount_candidates)
    if amt_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            sums = df[numeric_cols].apply(lambda s: s[s>0].sum())
            amt_col = sums.sort_values(ascending=False).index[0] if not sums.empty else numeric_cols[0]
    from_col = pick_col(df.columns, from_candidates)
    to_col = pick_col(df.columns, to_candidates)

    # Build normalized columns
    df["_timestamp"] = ensure_datetime(df[ts_col]) if ts_col is not None else pd.NaT
    df["_ticker"] = df[ticker_col].astype(str).str.strip() if ticker_col is not None else "UNKNOWN"
    df["_amount"] = pd.to_numeric(df[amt_col], errors="coerce") if amt_col is not None else np.nan
    df["_from"] = df[from_col].astype(str) if from_col is not None else None
    df["_to"] = df[to_col].astype(str) if to_col is not None else None

    df = df.dropna(subset=["_timestamp", "_amount"]).copy()
    df = df.sort_values(by=["_ticker", "_timestamp"]).reset_index(drop=True)
    return df

def decayed_value_at(series_times_ns, series_amounts, t_ns, half_life_sec: float) -> float:
    age_sec = (t_ns - series_times_ns) / 1e9
    mask = age_sec >= 0
    if mask.any():
        return float(np.sum(series_amounts[mask] * np.power(0.5, age_sec[mask] / half_life_sec)))
    return 0.0

def compute_series(sub_df: pd.DataFrame, half_life_min: float, freq: str) -> pd.DataFrame:
    if sub_df.empty:
        return pd.DataFrame()
    half_life_sec = half_life_min * 60.0
    now_tokyo = pd.Timestamp.now(TOKYO)
    idx_start = sub_df["_timestamp"].min().floor(freq)
    idx_end = max(sub_df["_timestamp"].max().ceil(freq), now_tokyo.floor(freq))
    idx = pd.date_range(start=idx_start, end=idx_end, freq=freq, tz=TOKYO)
    tx_ns = sub_df["_timestamp"].astype("int64").values
    tx_amt = sub_df["_amount"].astype(float).values

    values = []
    for t in idx:
        values.append(decayed_value_at(tx_ns, tx_amt, t.value, half_life_sec))
    series = pd.Series(values, index=idx, name="value")
    dt_sec = (series.index[1] - series.index[0]).total_seconds() if len(series.index) >= 2 else 1.0
    derivative = series.diff() / dt_sec
    integral = (series * dt_sec).cumsum()
    out = pd.DataFrame({"value": series.values, "derivative_per_sec": derivative.values, "integral_value": integral.values}, index=series.index)
    return out

def build_summary(df: pd.DataFrame, half_life_min: float) -> pd.DataFrame:
    now_tokyo = pd.Timestamp.now(TOKYO)
    age_seconds = (now_tokyo - df["_timestamp"]).dt.total_seconds()
    decay_factor = 0.5 ** (age_seconds / (half_life_min*60.0))
    df["_decayed_amount_now"] = df["_amount"] * decay_factor
    g = df.groupby("_ticker", dropna=False)
    summary = g.agg(
        first_time=("_timestamp", "min"),
        last_time=("_timestamp", "max"),
        trades=("_timestamp", "count"),
        raw_sum=("_amount", "sum"),
        decayed_sum_now=("_decayed_amount_now", "sum"),
        avg_trade_size=("_amount", "mean"),
    ).reset_index()
    summary["span_minutes"] = (summary["last_time"] - summary["first_time"]).dt.total_seconds() / 60.0
    summary["trades_per_hour"] = summary["trades"] / np.maximum(summary["span_minutes"]/60.0, 1e-9)
    return summary.sort_values("decayed_sum_now", ascending=False).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="Input CSVs (glob patterns allowed)")
    ap.add_argument("--half-life-min", type=float, default=30.0, help="Half-life in minutes (default: 30)" )
    ap.add_argument("--freq", default="5T", help="Resample frequency (e.g., 1T, 5T, 15T)" )
    ap.add_argument("--top", type=int, default=3, help="How many top tickers to plot (by current value)" )
    ap.add_argument("--outdir", default="outputs", help="Output directory" )
    args = ap.parse_args()

    # Expand globs
    paths = []
    for pat in args.input:
        paths.extend(glob.glob(pat))
    if not paths:
        raise SystemExit("No input files found for given --input patterns.")

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "plots"), exist_ok=True)

    df = load_and_normalize(paths)
    summary_df = build_summary(df, args.half_life_min)
    summary_path = os.path.join(args.outdir, "summary_tickers.csv")
    summary_df.to_csv(summary_path, index=False)

    # Compute per-ticker time series, save CSVs, gather integral/derivative summary
    rows = []
    for ticker in summary_df["_ticker"].tolist():
        sub = df[df["_ticker"] == ticker].copy()
        ts = compute_series(sub, args.half_life_min, args.freq)
        if ts.empty:
            continue
        per_ticker_path = os.path.join(args.outdir, f"analysis_ticker_{sanitize_filename(ticker)}.csv")
        ts.reset_index().rename(columns={"index": "time_tokyo"}).to_csv(per_ticker_path, index=False)

        latest_val = ts["value"].iloc[-1]
        peak_idx = ts["value"].idxmax()
        peak_val = ts.loc[peak_idx, "value"] if hasattr(peak_idx, "tzinfo") else float("nan")
        rows.append({
            "ticker": ticker,
            "latest_value_now": latest_val,
            "max_integral_value": float(ts["integral_value"].max()),
            "max_derivative_per_sec": float(ts["derivative_per_sec"].max()),
            "min_derivative_per_sec": float(ts["derivative_per_sec"].min()),
            "peak_value": float(peak_val),
            "peak_value_time_tokyo": peak_idx if hasattr(peak_idx, "tzinfo") else "",
            "per_ticker_csv": per_ticker_path,
        })

    integ_deriv_summary = pd.DataFrame(rows).sort_values("latest_value_now", ascending=False).reset_index(drop=True)
    integ_deriv_summary_path = os.path.join(args.outdir, "ticker_integral_derivative_summary.csv")
    integ_deriv_summary.to_csv(integ_deriv_summary_path, index=False)

    # Plot top-N
    top_n = min(args.top, len(summary_df))
    top_tickers = summary_df.head(top_n)["_ticker"].tolist()
    for ticker in top_tickers:
        sub = df[df["_ticker"] == ticker].copy()
        ts = compute_series(sub, args.half_life_min, args.freq)
        if ts.empty:
            continue
        # Value
        plt.figure()
        plt.plot(ts.index, ts["value"].values)
        plt.title(f"{ticker} - Decayed Value (half-life {args.half_life_min} min)")
        plt.xlabel("Time (Asia/Tokyo)")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "plots", f"{sanitize_filename(ticker)}_value.png"))
        plt.close()

        # Derivative
        plt.figure()
        plt.plot(ts.index, ts["derivative_per_sec"].values)
        plt.title(f"{ticker} - Derivative of Value (per second)")
        plt.xlabel("Time (Asia/Tokyo)")
        plt.ylabel("dValue/dt (per sec)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "plots", f"{sanitize_filename(ticker)}_derivative.png"))
        plt.close()

        # Integral
        plt.figure()
        plt.plot(ts.index, ts["integral_value"].values)
        plt.title(f"{ticker} - Integral of Value (cumulative area)")
        plt.xlabel("Time (Asia/Tokyo)")
        plt.ylabel("Integral (value·sec)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "plots", f"{sanitize_filename(ticker)}_integral.png"))
        plt.close()

    print("\n=== DONE ===")
    print("Summary (tickers):", summary_path)
    print("Integral/Derivative summary:", integ_deriv_summary_path)
    print("Per-ticker CSVs and plots saved under:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()