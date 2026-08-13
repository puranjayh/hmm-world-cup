"""
make_figures.py — Regenerate the paper figures from the committed artifacts.

Reads model/artifacts/gaussian/global_hmm.pkl, data/raw/filtered_matches.csv, and
results/predictions_*.csv; writes PNGs to paper_figures/.

    python scripts/make_figures.py
"""
from __future__ import annotations

import csv
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from model.gaussian_hmm.hmm_global import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures"
OUT.mkdir(exist_ok=True)
WINDOW = 3
RUNG = ["Very Poor", "Poor", "Below Avg", "Average", "Above Avg", "Strong", "Elite"]
FEAT_LABEL = {
    'ewa_win_rate': 'Win rate', 'ewa_goal_diff': 'Goal diff',
    'rolling_win_vs_strong_5': 'Win vs strong', 'rolling_goal_diff_std_5': 'GD volatility',
    'rolling_win_rate_std_5': 'Win volatility', 'ewa_win_rate_momentum': 'Win momentum',
    'ewa_goal_diff_momentum': 'GD momentum',
}

hmm = pickle.load(open(ROOT / "model/artifacts/gaussian/global_hmm.pkl", "rb"))
means = np.asarray(hmm.model.means_)
si = [FEATURE_NAMES.index(f) for f in ('ewa_win_rate', 'ewa_goal_diff', 'rolling_win_vs_strong_5')]
order = np.argsort(means[:, si].sum(axis=1))          # Poor -> Elite
palette = plt.cm.RdYlGn(np.linspace(0.12, 0.90, 7))


def fig_form_ladder():
    M = means[order]
    cols = [FEAT_LABEL[f] for f in FEATURE_NAMES]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    norm = TwoSlopeNorm(vmin=min(-2.2, M.min()), vcenter=0, vmax=max(2.2, M.max()))
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(7)); ax.set_yticklabels(RUNG, fontsize=10)
    for i in range(7):
        for j in range(len(cols)):
            v = M[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    color="white" if abs(v) > 1.3 else "black", fontsize=8.5)
    ax.set_title("The form ladder: latent-state means (standardized)", fontsize=13, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("std. dev. from mean")
    fig.tight_layout(); fig.savefig(OUT / "fig1_form_ladder.png", dpi=170); plt.close(fig)


def fig_trajectory(team, tstart, tend, title, fname):
    df = pd.read_csv(ROOT / "data/raw/filtered_matches.csv")
    df["date"] = pd.to_datetime(df["date"])
    g = df[df["team"] == team].sort_values("date")
    feats = g[FEATURE_NAMES].fillna(0).to_numpy(float)
    dates = g["date"].to_numpy()
    tm = g[(g["date"] >= tstart) & (g["date"] <= tend)]
    posts, labels = [], []
    for _, row in tm.iterrows():
        idx = np.searchsorted(dates, np.datetime64(pd.Timestamp(row["date"])), side="left")
        pf = hmm.posterior_features(feats[max(0, idx - WINDOW):idx])
        posts.append(np.asarray(pf[:7])[order]); labels.append(str(row["opponent"]))
    if not posts:
        return
    P = np.array(posts); x = np.arange(len(P))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(x, P.T, colors=palette, labels=RUNG, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([f"vs {l}" for l in labels], rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1); ax.set_xlim(0, len(P) - 1); ax.set_ylabel("Posterior over form states")
    ax.set_title(title, fontsize=13, pad=10)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9, title="Form rung")
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=170); plt.close(fig)


def fig_calibration():
    pred, occ = [], []
    for f in glob.glob(str(ROOT / "results/predictions_*.csv")):
        for r in csv.DictReader(open(f)):
            team = (r.get("Team") or "").strip(); res = (r.get("Result") or "").strip()
            if not team or not res:
                continue
            try:
                pw = float(r["Predicted Win (v2)"]) / 100
                pdw = float(r["Predicted Draw (v2)"]) / 100
                pl = float(r["Predicted Loss (v2)"]) / 100
            except (KeyError, ValueError):
                continue
            win = int(res == team); draw = int(res == "Draw"); loss = int(not win and not draw)
            pred += [pw, pdw, pl]; occ += [win, draw, loss]
    pred = np.array(pred); occ = np.array(occ)
    idx = np.clip(np.digitize(pred, np.linspace(0, 1, 11)) - 1, 0, 9)
    xs, ys, ns = [], [], []
    for b in range(10):
        m = idx == b
        if m.sum() >= 5:
            xs.append(pred[m].mean()); ys.append(occ[m].mean()); ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.plot([0, 1], [0, 1], "--", color="#888", label="perfect calibration")
    ax.plot(xs, ys, "o-", color="#1f4e8c", lw=2, ms=7, label="Global Gaussian HMM")
    for x, y, nn in zip(xs, ys, ns):
        ax.annotate(f"n={nn}", (x, y), textcoords="offset points", xytext=(6, -10), fontsize=7.5, color="#555")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration of predicted W/D/L probabilities\n(pooled over 4 held-out tournaments)", fontsize=12, pad=10)
    ax.legend(loc="upper left", fontsize=10); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "fig3_calibration.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    fig_form_ladder()
    fig_trajectory("Argentina", "2022-11-20", "2022-12-18",
                   "Form trajectory: Argentina, 2022 World Cup (champions)",
                   "fig2_trajectory_argentina2022.png")
    fig_trajectory("Morocco", "2022-11-20", "2022-12-18",
                   "Form trajectory: Morocco, 2022 World Cup (run to the semi-final)",
                   "fig2_trajectory_morocco2022.png")
    fig_calibration()
    print(f"figures written to {OUT}")
