"""
make_results.py — Regenerate every CSV in results/ from a single evaluation run.

Previously these tables were assembled by hand from terminal output across
separate runs, which let inconsistencies creep in (e.g. accuracy columns taken
from GlobalGHMM while the gated column came from GlobalGHMM+Draw, in the same
row). Everything here is derived from one metrics_global_ghmm.json, so the rows
are guaranteed mutually consistent and tied to one commit.

Usage:
    python -m model.gaussian_hmm.evaluate_global    # writes the metrics JSON
    python scripts/make_results.py                  # writes results/*.csv
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))   # allow running as `python scripts/make_results.py`
METRICS_JSON = PROJECT_ROOT / "model" / "artifacts" / "gaussian" / "metrics_global_ghmm.json"
RESULTS_DIR  = PROJECT_ROOT / "results"

# Keep in sync with evaluate_global.WINDOW
from model.gaussian_hmm.evaluate_global import WINDOW  # noqa: E402

MODEL_ORDER = ["GlobalGHMM", "GlobalGHMM+Draw", "XGBoost", "RF", "Elo", "Uniform"]
DISPLAY_NAME = {"RF": "Random Forest"}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    if not METRICS_JSON.exists():
        raise SystemExit(
            f"{METRICS_JSON} not found — run `python -m model.gaussian_hmm.evaluate_global` first."
        )

    metrics = json.loads(METRICS_JSON.read_text())
    RESULTS_DIR.mkdir(exist_ok=True)
    commit = _git_commit()

    main_rows, gate_rows = [], []

    for run in metrics.values():
        label = run["label"]
        # subset -> {model: metric dict}
        for subset, block in [("All Matches", run["models"]), ("W/L Only", run["nodraw"])]:
            for model in MODEL_ORDER:
                m = block.get(model)
                if not m:
                    continue
                main_rows.append({
                    "tournament": label,
                    "subset":     subset,
                    "model":      DISPLAY_NAME.get(model, model),
                    "n":          m["n"],
                    "log_loss":   m["log_loss"],
                    "brier":      m["brier"],
                    "accuracy":   m["accuracy"],
                    "rps":        m["rps"],
                    "window":     WINDOW,
                    "commit":     commit,
                })

        # Confidence gating is reported for GlobalGHMM+Draw only — label it as such
        # so the source model is never ambiguous.
        for key, cm in run["conf_gated"].items():
            gate_rows.append({
                "tournament": label,
                "model":      "GlobalGHMM+Draw",
                "threshold":  int(key.replace("thresh_", "")) / 100,
                "n":          cm["n"],
                "coverage":   cm["coverage"],
                "accuracy":   cm["accuracy"],
                "window":     WINDOW,
                "commit":     commit,
            })

    main_df = pd.DataFrame(main_rows)
    gate_df = pd.DataFrame(gate_rows)

    main_path = RESULTS_DIR / "main_results.csv"
    gate_path = RESULTS_DIR / "confidence_gating.csv"
    main_df.to_csv(main_path, index=False)
    gate_df.to_csv(gate_path, index=False)

    print(f"commit {commit}, WINDOW={WINDOW}")
    print(f"  wrote {main_path.relative_to(PROJECT_ROOT)}  ({len(main_df)} rows)")
    print(f"  wrote {gate_path.relative_to(PROJECT_ROOT)}  ({len(gate_df)} rows)")


if __name__ == "__main__":
    main()
