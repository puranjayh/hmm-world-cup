"""
make_results.py — Derive every results CSV from the single metrics JSON.

The results tables used to be transcribed by hand across separate runs, which is
how an accuracy column could come from one model while the gated column in the
same row came from another, and how the committed CSVs ended up describing a
dataset vintage the pipeline no longer produced.

Nothing here recomputes a metric. Every number is read straight out of
model/artifacts/gaussian/metrics_global_ghmm.json, and every row is stamped with
the window, the primary Elo regime, and the commit that produced it — so a table
can always be traced back to the run behind it.

Usage
-----
    python scripts/make_results.py                 # writes results/*.csv
    python scripts/make_results.py --check         # verify CSVs match the JSON
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

METRICS_JSON = _PKG_ROOT / "model" / "artifacts" / "gaussian" / "metrics_global_ghmm.json"
RESULTS_DIR  = _PKG_ROOT / "results"

# Display names for the paper tables. Keys are the model keys used in the JSON.
MODEL_LABELS = {
    "GlobalGHMM":      "GlobalGHMM",
    "GlobalGHMM+Draw": "GlobalGHMM+Draw",
    "XGBoost":         "XGBoost",
    "RF":              "Random Forest",
    "Elo":             "Elo",
    "Uniform":         "Uniform",
}

SUBSETS = {"models": "All Matches", "nodraw": "W/L Only"}


def _load() -> tuple[dict, dict]:
    if not METRICS_JSON.exists():
        raise SystemExit(
            f"Metrics not found at {METRICS_JSON}\n"
            "Run:  python -m model.gaussian_hmm.evaluate_global"
        )
    with open(METRICS_JSON, encoding="utf-8") as f:
        data = json.load(f)
    meta = data.pop("_meta", {})
    if not meta:
        print("  WARNING: metrics JSON has no _meta block — it predates run "
              "stamping. Re-run evaluate_global.py for traceable output.")
    return data, meta


def _stamp(meta: dict) -> dict:
    """Provenance columns attached to every emitted row."""
    return {
        "window":       meta.get("window"),
        "elo_mode":     meta.get("primary_elo_mode"),
        "reproducible": meta.get("reproducible"),
        "commit":       meta.get("commit"),
        "generated_utc": meta.get("generated_utc"),
    }


def build_main_results(data: dict, meta: dict) -> pd.DataFrame:
    """One row per (window, subset, model). Metrics never cross model boundaries."""
    stamp = _stamp(meta)
    rows = []
    for tag, run in data.items():
        for subset_key, subset_label in SUBSETS.items():
            for model_key, m in run.get(subset_key, {}).items():
                if not m:                      # W/L-only can be empty
                    continue
                rows.append({
                    "tag":        tag,
                    "tournament": run["label"],
                    "subset":     subset_label,
                    "model":      MODEL_LABELS.get(model_key, model_key),
                    "n":          m.get("n"),
                    "log_loss":   m.get("log_loss"),
                    "brier":      m.get("brier"),
                    "accuracy":   m.get("accuracy"),
                    "rps":        m.get("rps"),
                    **stamp,
                })
    return pd.DataFrame(rows)


def build_confidence_gating(data: dict, meta: dict) -> pd.DataFrame:
    """Gated accuracy. Explicitly records which model the gating describes."""
    stamp = _stamp(meta)
    rows = []
    for tag, run in data.items():
        for thresh_key, cm in run.get("conf_gated", {}).items():
            rows.append({
                "tag":           tag,
                "tournament":    run["label"],
                "gated_model":   "GlobalGHMM+Draw",   # the model conf_gated is computed on
                "threshold_pct": int(thresh_key.replace("thresh_", "")),
                "n_matches":     cm.get("n"),
                "coverage":      cm.get("coverage"),
                "accuracy":      cm.get("accuracy"),
                **stamp,
            })
    return pd.DataFrame(rows)


def build_elo_modes(data: dict, meta: dict) -> pd.DataFrame:
    """Published vs dynamic Elo at test time — the cost of the production path."""
    stamp = _stamp(meta)
    rows = []
    for tag, run in data.items():
        for mode, models in run.get("elo_modes", {}).items():
            for model_key, m in models.items():
                rows.append({
                    "tag":        tag,
                    "tournament": run["label"],
                    "elo_regime": mode,
                    "is_primary": mode == run.get("primary_elo_mode"),
                    "model":      MODEL_LABELS.get(model_key, model_key),
                    "n":          m.get("n"),
                    "log_loss":   m.get("log_loss"),
                    "brier":      m.get("brier"),
                    "accuracy":   m.get("accuracy"),
                    "rps":        m.get("rps"),
                    **{k: v for k, v in stamp.items() if k != "elo_mode"},
                })
    return pd.DataFrame(rows)


def build_significance(data: dict, meta: dict) -> pd.DataFrame:
    """
    Accuracy with a bootstrap interval, plus a paired test against the reference
    model. Emitted so no table can quote a bare accuracy without the interval
    sitting next to it — at 32-64 matches per tournament a single flipped result
    moves accuracy by more than the gap between most of these models.
    """
    stamp = _stamp(meta)
    ref   = meta.get("sig_reference", "GlobalGHMM+Draw")
    rows  = []
    for tag, run in data.items():
        for model_key, s in run.get("significance", {}).items():
            for subset_key, subset_label in (("wdl", "W/D/L"), ("wl", "W/L Only")):
                m = s.get(subset_key, {})
                if m.get("accuracy") is None:
                    continue
                vr = (s.get("vs_reference") or {}).get(subset_key, {})
                rows.append({
                    "tag":            tag,
                    "tournament":     run["label"],
                    "subset":         subset_label,
                    "model":          MODEL_LABELS.get(model_key, model_key),
                    "n":              m.get("n"),
                    "accuracy":       m.get("accuracy"),
                    "ci_lo":          m.get("ci_lo"),
                    "ci_hi":          m.get("ci_hi"),
                    "vs_model":       ref if model_key != ref else None,
                    "ref_only_correct":   vr.get("ref_only_correct"),
                    "other_only_correct": vr.get("other_only_correct"),
                    "n_discordant":   vr.get("n_discordant"),
                    "mcnemar_p":      vr.get("p_value"),
                    "n_bootstrap":    meta.get("n_bootstrap"),
                    **stamp,
                })
    return pd.DataFrame(rows)


TABLES = {
    "main_results.csv":      build_main_results,
    "confidence_gating.csv": build_confidence_gating,
    "elo_modes.csv":         build_elo_modes,
    "significance.csv":      build_significance,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Verify committed CSVs match the metrics JSON; exit 1 if stale.")
    args = ap.parse_args()

    data, meta = _load()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stale = []
    for fname, builder in TABLES.items():
        df   = builder(data, meta)
        path = RESULTS_DIR / fname

        if args.check:
            if not path.exists():
                stale.append(f"{fname} (missing)")
                continue
            existing = pd.read_csv(path)
            fresh    = pd.read_csv(io_buf(df))
            if not existing.equals(fresh):
                stale.append(fname)
            continue

        df.to_csv(path, index=False)
        print(f"  wrote {path.relative_to(_PKG_ROOT)}  ({len(df)} rows)")

    if args.check:
        if stale:
            print("STALE (re-run without --check): " + ", ".join(stale))
            raise SystemExit(1)
        print("All results CSVs match the metrics JSON.")
        return

    commit = meta.get("commit") or "unknown"
    print(f"\n  window={meta.get('window')}  elo_mode={meta.get('primary_elo_mode')}  "
          f"reproducible={meta.get('reproducible')}  commit={commit}")


def io_buf(df: pd.DataFrame):
    """Round-trip a frame through CSV text so dtypes match a file read."""
    import io
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


if __name__ == "__main__":
    main()
