"""
generate_series.py

Reference time-series generator for the change-point detection benchmark.
Use this module to produce the exact same dataset across teams/models.

Generative model
----------------
    X_t = L_t + a * N_beta(t)

    L_t        : piecewise-constant signal, 3-state random walk with
                 reflecting bounds on states {-1, 0, +1}.
                 Levels = base_level + state * level_step = {-pi, 0, +pi}.
                 Segment lengths ~ Geom(1/200), clipped to >= 20.
    N_beta(t)  : noise process. Six families:
                   - white          (Gaussian iid)
                   - pink   (beta=1)  via 1/f^beta PSD
                   - red    (beta=2)
                   - blue   (beta=-1)
                   - violet (beta=-2)
                   - el_nino-like   (AR(1), phi=0.995, + slow sinusoid)
    a          : amplitude in {1, 2, pi}.

Factorial design
----------------
    6 noise types  x  3 amplitudes  x  10 seeds  =  180 series.
    Series length = 10_000.
    Labels y_t = state in {-1, 0, +1}.

Dependencies
------------
    numpy, pandas, colorednoise, BTP.py (this repo).

Quick usage
-----------
    # 1. produce all 180 series in memory
    from generate_series import iter_dataset
    for name, seed, df in iter_dataset():
        X = df["x"].to_numpy()
        y = df["state"].to_numpy()
        # ... apply your CPD model to X, evaluate against y ...

    # 2. save once, share the directory with colleagues
    from generate_series import save_dataset
    save_dataset("dataset")          # writes 180 CSVs to ./dataset/

    # 3. evaluate predicted change-points with tolerance k
    from generate_series import match_change_points, precision_recall_f1
    tp, fp, fn = match_change_points(true_cps, pred_cps, tolerance=5)
    f1, prec, rec = precision_recall_f1(tp, fp, fn)
"""

from __future__ import annotations

import math
import os
from typing import Iterator, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import colorednoise as cn

from BTP import Binary_Telegraph_Process as BTP


# --------------------------------------------------------------------------- #
# Experimental protocol — DO NOT change without coordinating with all teams.
# --------------------------------------------------------------------------- #

SAMPLE_LEN: int = 10_000
EXPECTED_SEG_LEN: int = 200
MIN_SEG_LEN: int = 20

LEVEL_STEP: float = math.pi
BASE_LEVEL: float = 0.0

MIN_STATE: int = -1
MAX_STATE: int = 1
INITIAL_STATE: int = 0

RANDOM_SEEDS: List[int] = list(range(10))   # 10 repetitions

NOISE_CONFIGS: List[Tuple[str, Dict[str, Any]]] = [
    ("white_a1",   dict(noise_fn=np.random.normal,        p1=0,  p2=1)),
    ("white_a2",   dict(noise_fn=np.random.normal,        p1=0,  p2=2)),
    ("white_api",  dict(noise_fn=np.random.normal,        p1=0,  p2=math.pi)),
    ("pink_a1",    dict(noise_fn=cn.powerlaw_psd_gaussian, p1=1,  alpha=1)),
    ("pink_a2",    dict(noise_fn=cn.powerlaw_psd_gaussian, p1=1,  alpha=2)),
    ("pink_api",   dict(noise_fn=cn.powerlaw_psd_gaussian, p1=1,  alpha=math.pi)),
    ("red_a1",     dict(noise_fn=cn.powerlaw_psd_gaussian, p1=2,  alpha=1)),
    ("red_a2",     dict(noise_fn=cn.powerlaw_psd_gaussian, p1=2,  alpha=2)),
    ("red_api",    dict(noise_fn=cn.powerlaw_psd_gaussian, p1=2,  alpha=math.pi)),
    ("blue_a1",    dict(noise_fn=cn.powerlaw_psd_gaussian, p1=-1, alpha=1)),
    ("blue_a2",    dict(noise_fn=cn.powerlaw_psd_gaussian, p1=-1, alpha=2)),
    ("blue_api",   dict(noise_fn=cn.powerlaw_psd_gaussian, p1=-1, alpha=math.pi)),
    ("violet_a1",  dict(noise_fn=cn.powerlaw_psd_gaussian, p1=-2, alpha=1)),
    ("violet_a2",  dict(noise_fn=cn.powerlaw_psd_gaussian, p1=-2, alpha=2)),
    ("violet_api", dict(noise_fn=cn.powerlaw_psd_gaussian, p1=-2, alpha=math.pi)),
    ("el_nino_a1", dict(noise_fn=None, p1="el_nino", alpha=1)),
    ("el_nino_a2", dict(noise_fn=None, p1="el_nino", alpha=2)),
    ("el_nino_api",dict(noise_fn=None, p1="el_nino", alpha=math.pi)),
]

NOISE_CONFIG_BY_NAME: Dict[str, Dict[str, Any]] = dict(NOISE_CONFIGS)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

def make_series(noise_name: str, seed: int) -> pd.DataFrame:
    """
    Generate one series for the given (noise_name, seed) pair.

    Returns a DataFrame with columns:
        x      : observed signal X_t = L_t + a * N_beta(t)
        level  : underlying piecewise-constant signal L_t in {-pi, 0, +pi}
        state  : integer state in {-1, 0, +1} (this is also the ground-truth label)
    """
    if noise_name not in NOISE_CONFIG_BY_NAME:
        raise KeyError(
            f"Unknown noise name: {noise_name!r}. "
            f"Allowed: {list(NOISE_CONFIG_BY_NAME)}"
        )
    cfg = NOISE_CONFIG_BY_NAME[noise_name]

    btp = BTP(
        SAMPLE_LEN,
        noise_fn=cfg.get("noise_fn"),
        p1=cfg.get("p1"),
        p2=cfg.get("p2"),
        p3=cfg.get("p3"),
        alpha=cfg.get("alpha"),
        RANDOM_SEED=seed,
        level_step=LEVEL_STEP,
        base_level=BASE_LEVEL,
        expected_seg_len=EXPECTED_SEG_LEN,
        min_seg_len=MIN_SEG_LEN,
        allow_random_walk=True,
        label_mode="state",
        min_state=MIN_STATE,
        max_state=MAX_STATE,
        initial_state=INITIAL_STATE,
    )
    return btp.labels()[["x", "level", "state"]].copy()


def iter_dataset() -> Iterator[Tuple[str, int, pd.DataFrame]]:
    """Yield (noise_name, seed, dataframe) for every series in the protocol."""
    for name, _cfg in NOISE_CONFIGS:
        for seed in RANDOM_SEEDS:
            yield name, seed, make_series(name, seed)


def save_dataset(out_dir: str = "dataset") -> None:
    """
    Save all 180 series as CSVs.

    File layout: <out_dir>/<noise_name>__seed<k>.csv
    Plus a manifest <out_dir>/manifest.csv listing all (noise, seed, file).
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name, seed, df in iter_dataset():
        fname = f"{name}__seed{seed}.csv"
        df.to_csv(os.path.join(out_dir, fname), index=False)
        rows.append({"noise": name, "seed": seed, "file": fname})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "manifest.csv"), index=False)


def load_series(out_dir: str, noise_name: str, seed: int) -> pd.DataFrame:
    """Load one series saved by save_dataset()."""
    return pd.read_csv(os.path.join(out_dir, f"{noise_name}__seed{seed}.csv"))


# --------------------------------------------------------------------------- #
# Evaluation helpers (use these so all teams report the same metrics).
# --------------------------------------------------------------------------- #

def true_change_points(state: np.ndarray) -> np.ndarray:
    """Indices where the state changes (ground-truth change points)."""
    state = np.asarray(state)
    return np.where(np.diff(state) != 0)[0] + 1


def match_change_points(
    true_cps: np.ndarray,
    pred_cps: np.ndarray,
    tolerance: int,
) -> Tuple[int, int, int]:
    """
    Greedy 1-to-1 matching of predicted to true change points within `tolerance`.
    Returns (tp, fp, fn).
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")

    true_cps = np.asarray(true_cps, dtype=int).reshape(-1)
    pred_cps = np.asarray(pred_cps, dtype=int).reshape(-1)

    if true_cps.size == 0 and pred_cps.size == 0:
        return 0, 0, 0
    if true_cps.size == 0:
        return 0, int(pred_cps.size), 0
    if pred_cps.size == 0:
        return 0, 0, int(true_cps.size)

    true_cps = np.sort(true_cps)
    pred_cps = np.sort(pred_cps)
    matched = np.zeros(true_cps.size, dtype=bool)
    tp = 0
    for pred in pred_cps:
        diffs = np.abs(true_cps - pred)
        candidates = np.where((diffs <= tolerance) & (~matched))[0]
        if candidates.size:
            best = candidates[np.argmin(diffs[candidates])]
            matched[best] = True
            tp += 1
    fp = int(pred_cps.size - tp)
    fn = int(true_cps.size - matched.sum())
    return int(tp), fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Standard precision / recall / F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


# --------------------------------------------------------------------------- #
# CLI: `python generate_series.py [output_dir]` writes the full dataset.
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    save_dataset(out)
    n = len(NOISE_CONFIGS) * len(RANDOM_SEEDS)
    print(f"Saved {n} series to {out}/")
