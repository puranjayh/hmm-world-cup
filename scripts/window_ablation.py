"""
window_ablation.py — Sweep the state-inference lookback window and tabulate its
effect on held-out log-loss.

Runs the full benchmark at several WINDOW values and writes
results/window_ablation.csv (one row per window: sample-weighted mean log-loss
and accuracy across all four tournaments, all-matches and W/L-only). The point
is to show the window is NOT a tuned knob — the spread across windows is within
run-to-run noise.

Usage:
    python scripts/window_ablation.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

METRICS_JSON = PROJECT_ROOT / "model" / "artifacts" / "gaussian" / "metrics_global_ghmm.json"
OUT_CSV      = PROJECT_ROOT / "results" / "window_ablation.csv"

WINDOWS = [1, 2, 3, 4, 5, 7, 10, 15, 20]
MODEL   = "GlobalGHMM+Draw"


def _weighted(metrics: dict, subset: str, field: str) -> float:
    """Sample-size-weighted mean of `field` across tournaments for one subset."""
    num = den = 0.0
    for run in metrics.values():
        block = run["models"] if subset == "all" else run["nodraw"]
        m = block.get(MODEL)
        if m:
            num += m[field] * m["n"]
            den += m["n"]
    return num / den if den else float("nan")


def main() -> None:
    rows = []
    for w in WINDOWS:
        print(f"  window = {w:>2} …", flush=True)
        env = {**os.environ, "WINDOW": str(w), "REPRODUCIBLE": "1"}
        subprocess.run(
            [sys.executable, "-m", "model.gaussian_hmm.evaluate_global"],
            cwd=PROJECT_ROOT, env=env, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        metrics = json.loads(METRICS_JSON.read_text())
        rows.append({
            "window":          w,
            "ll_all":          round(_weighted(metrics, "all", "log_loss"), 4),
            "acc_all":         round(_weighted(metrics, "all", "accuracy"), 4),
            "ll_wl":           round(_weighted(metrics, "wl", "log_loss"), 4),
            "acc_wl":          round(_weighted(metrics, "wl", "accuracy"), 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV.relative_to(PROJECT_ROOT)}")
    print(df.to_string(index=False))
    print(f"\nll_all spread across windows : {df['ll_all'].max() - df['ll_all'].min():.4f} nats")
    print(f"ll_wl  spread across windows : {df['ll_wl'].max() - df['ll_wl'].min():.4f} nats")
    # NOTE: re-run `python -m model.gaussian_hmm.evaluate_global` afterwards to
    # restore the metrics JSON to the canonical WINDOW=3 before make_results.py.


if __name__ == "__main__":
    main()
