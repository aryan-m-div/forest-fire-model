# Fixed-theta Drossel-Schwabl-style simulator with simple initial layouts.
# Writes one results folder per run with config.json and fires.csv.
#
# NOTE: this is the ORIGINAL simulator. It seeds a single RNG and draws from it
# for BOTH the initial-lattice construction and the event loop. Because the four
# initial layouts consume different numbers of setup draws, runs that differ only
# in layout can share (or offset) the event stream. Combined with the model's
# memory-erasure timescale this makes three of the four layouts converge onto a
# single trajectory at high theta. The corrected simulator (ff_fast.py) draws the
# initialization and event streams independently and keys the event stream on the
# layout; the results reported in the paper come from ff_fast.py via runner.py.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict
import os, json, time
import numpy as np
import pandas as pd

# ------------------ States ------------------
EMPTY, TREE, BURNING, BURNT = 0, 1, 2, 3

# ------------------ Layout controls (clustered / stripes / gradient) ------------------
NUM_BLOBS = 6          # for clustered
BLOB_SIGMA_FRAC = 0.08 # sigma as fraction of N
STRIPE_PERIOD = 12
DENSE_IN_STRIPE = 0.75
SPARSE_IN_STRIPE = 0.35
GRAD_LEFT = 0.40
GRAD_RIGHT = 0.70

SAVE_EVERY = 5000  # CSV flush interval

# ------------------ Config ------------------
@dataclass
class RunConfig:
    N: int
    theta: int
    initial_density_t: float
    distribution: str          # "uniform" | "clustered" | "stripes" | "gradient"
    total_lightnings: int
    transient_fraction: float  # e.g., 0.2
    seed: int

# ------------------ Utilities ------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def short_id(rng: np.random.Generator) -> str:
    # 8-hex id for folder uniqueness
    return ("%08x" % int(rng.integers(0, 16**8)))

def make_run_folder(cfg: RunConfig, rng: np.random.Generator) -> str:
    tag = f"{cfg.distribution}_t{cfg.initial_density_t:.2f}_theta{cfg.theta}_N{cfg.N}_{now_tag()}_{short_id(rng)}"
    out_dir = os.path.join("results", tag)
    ensure_dir(out_dir)
    return out_dir

def write_metadata(out_dir: str, cfg: RunConfig) -> None:
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

# ------------------ Initial forests ------------------
def generate_forest_uniform(N: int, t: float, rng: np.random.Generator) -> np.ndarray:
    grid = (rng.random((N, N)) < t).astype(np.uint8)
    grid[grid == 1] = TREE
    return grid

def generate_forest_clustered(N: int, t: float, rng: np.random.Generator) -> np.ndarray:
    # Sum of Gaussian blobs -> normalize -> threshold to hit global t
    sigma = BLOB_SIGMA_FRAC * N
    yy, xx = np.mgrid[0:N, 0:N]
    field = np.zeros((N, N), dtype=np.float64)
    for _ in range(NUM_BLOBS):
        cy = rng.uniform(0, N)
        cx = rng.uniform(0, N)
        field += np.exp(-(((yy - cy) ** 2) + ((xx - cx) ** 2)) / (2.0 * sigma * sigma))
    # normalize 0..1 robustly (NumPy 2.x safe)
    mn, mx = field.min(), field.max()
    if mx - mn < 1e-12:
        norm = np.zeros_like(field)
    else:
        norm = (field - mn) / (np.ptp(field) + 1e-12)
    thresh = np.quantile(norm, 1.0 - t)  # keep top t fraction
    grid = (norm >= thresh).astype(np.uint8)
    grid[grid == 1] = TREE
    return grid

def generate_forest_stripes(N: int, t: float, rng: np.random.Generator) -> np.ndarray:
    # Vertical stripes alternating between dense and sparse probabilities, then adjust to hit t via quantile
    j = np.arange(N)
    stripe_id = (j // STRIPE_PERIOD) % 2  # 0,1,0,1,...
    p_col = np.where(stripe_id == 0, DENSE_IN_STRIPE, SPARSE_IN_STRIPE).astype(np.float64)
    # sample then adjust to match t using a simple bias
    base = rng.random((N, N))
    # turn base into occupancy by comparing to column probabilities
    occ = (base < p_col[None, :]).astype(np.float64)
    # adjust to match exact t by quantile re-thresholding on a score = base - p_col
    score = base - p_col[None, :]
    thresh = np.quantile(score, t)
    adj = (score <= thresh).astype(np.uint8)
    grid = adj
    grid[grid == 1] = TREE
    return grid

def generate_forest_gradient(N: int, t: float, rng: np.random.Generator) -> np.ndarray:
    # Linear probability across columns; then quantile-threshold to hit t exactly
    j = np.linspace(0.0, 1.0, N)
    p_col = GRAD_LEFT + (GRAD_RIGHT - GRAD_LEFT) * j
    base = rng.random((N, N))
    score = base - p_col[None, :]
    thresh = np.quantile(score, t)
    grid = (score <= thresh).astype(np.uint8)
    grid[grid == 1] = TREE
    return grid

def generate_forest(N: int, t: float, rng: np.random.Generator, dist: str) -> np.ndarray:
    if dist == "uniform":
        return generate_forest_uniform(N, t, rng)
    if dist == "clustered":
        return generate_forest_clustered(N, t, rng)
    if dist == "stripes":
        return generate_forest_stripes(N, t, rng)
    if dist == "gradient":
        return generate_forest_gradient(N, t, rng)
    raise ValueError(f"Unknown distribution '{dist}'")

# ------------------ Local measures ------------------
def local_density_9x9(grid: np.ndarray, i0: int, j0: int) -> float:
    N = grid.shape[0]
    i1, i2 = max(0, i0 - 4), min(N, i0 + 5)
    j1, j2 = max(0, j0 - 4), min(N, j0 + 5)
    window = grid[i1:i2, j1:j2]
    trees = np.count_nonzero(window == TREE)
    center_tree = 1 if grid[i0, j0] == TREE else 0
    denom = window.size - 1
    return (trees - center_tree) / max(1, denom)

# ------------------ Burn cluster (BFS) and metrics ------------------
def burn_cluster_instant(grid: np.ndarray, i0: int, j0: int) -> Dict:
    """Burn the 4-neighbour connected TREE cluster containing (i0,j0).
       Returns dict of metrics and mutates grid (TREE->BURNT).
    """
    N = grid.shape[0]
    if grid[i0, j0] != TREE:
        return {
            "fire_size": 0,
            "fire_radius": 0.0,
            "perimeter": 0,
            "gyration_radius": 0.0,
            "bbox_aspect": 1.0,
            "spans_vertical": False,
            "spans_horizontal": False,
            "cells": [],  # not written to CSV; used internally
        }

    q: List[Tuple[int, int]] = [(i0, j0)]
    grid[i0, j0] = BURNT

    cells: List[Tuple[int, int]] = []
    sum_dist = 0.0

    min_i = max_i = i0
    min_j = max_j = j0

    while q:
        i, j = q.pop()
        cells.append((i, j))
        di = i - i0
        dj = j - j0
        sum_dist += np.sqrt(di * di + dj * dj)

        if i < min_i: min_i = i
        if i > max_i: max_i = i
        if j < min_j: min_j = j
        if j > max_j: max_j = j

        # 4-neighbour expansion
        if i > 0 and grid[i - 1, j] == TREE:
            grid[i - 1, j] = BURNT
            q.append((i - 1, j))
        if i < N - 1 and grid[i + 1, j] == TREE:
            grid[i + 1, j] = BURNT
            q.append((i + 1, j))
        if j > 0 and grid[i, j - 1] == TREE:
            grid[i, j - 1] = BURNT
            q.append((i, j - 1))
        if j < N - 1 and grid[i, j + 1] == TREE:
            grid[i, j + 1] = BURNT
            q.append((i, j + 1))

    S = len(cells)
    fire_radius = (sum_dist / S) if S > 0 else 0.0

    # Perimeter (4-neighbour)
    perim = 0
    burned_mask = np.zeros((N, N), dtype=np.uint8)
    for (i, j) in cells:
        burned_mask[i, j] = 1
    for (i, j) in cells:
        if i == 0 or burned_mask[i - 1, j] == 0: perim += 1
        if i == N - 1 or burned_mask[i + 1, j] == 0: perim += 1
        if j == 0 or burned_mask[i, j - 1] == 0: perim += 1
        if j == N - 1 or burned_mask[i, j + 1] == 0: perim += 1

    # Radius of gyration about cluster centroid
    pts = np.asarray(cells, dtype=np.float64)
    c = pts.mean(axis=0) if S > 0 else np.array([i0, j0], dtype=np.float64)
    Rg = float(np.sqrt(((pts - c) ** 2).sum(axis=1).mean())) if S > 0 else 0.0

    # Bounding box aspect
    w = (max_j - min_j + 1)
    h = (max_i - min_i + 1)
    bbox_aspect = (w / h) if h > 0 else 1.0

    # Span flags (open boundaries)
    spans_vertical = (min_i == 0 and max_i == N - 1)
    spans_horizontal = (min_j == 0 and max_j == N - 1)

    return {
        "fire_size": int(S),
        "fire_radius": float(fire_radius),
        "perimeter": int(perim),
        "gyration_radius": float(Rg),
        "bbox_aspect": float(bbox_aspect),
        "spans_vertical": bool(spans_vertical),
        "spans_horizontal": bool(spans_horizontal),
        "cells": cells,  # not persisted
    }

# ------------------ One lightning event ------------------
def ds_step(theta: int, grid: np.ndarray, rng: np.random.Generator) -> Dict:
    N = grid.shape[0]

    # theta growth attempts: flip EMPTY or BURNT to TREE
    if theta > 0:
        ii = rng.integers(0, N, size=theta)
        jj = rng.integers(0, N, size=theta)
        sel = (grid[ii, jj] == EMPTY) | (grid[ii, jj] == BURNT)
        grid[ii[sel], jj[sel]] = TREE

    rho_before = np.count_nonzero(grid == TREE) / (N * N)

    # Lightning
    i0 = int(rng.integers(0, N))
    j0 = int(rng.integers(0, N))
    loc = local_density_9x9(grid, i0, j0)

    if grid[i0, j0] == TREE:
        burn = burn_cluster_instant(grid, i0, j0)
    else:
        burn = {
            "fire_size": 0,
            "fire_radius": 0.0,
            "perimeter": 0,
            "gyration_radius": 0.0,
            "bbox_aspect": 1.0,
            "spans_vertical": False,
            "spans_horizontal": False,
            "cells": [],
        }

    rho_after = np.count_nonzero(grid == TREE) / (N * N)

    rec = {
        "rho_before": float(rho_before),
        "rho_after": float(rho_after),
        "fire_size": int(burn["fire_size"]),
        "fire_radius": float(burn["fire_radius"]),
        "perimeter": int(burn["perimeter"]),
        "gyration_radius": float(burn["gyration_radius"]),
        "bbox_aspect": float(burn["bbox_aspect"]),
        "spans_vertical": bool(burn["spans_vertical"]),
        "spans_horizontal": bool(burn["spans_horizontal"]),
        "ignition_x": i0,
        "ignition_y": j0,
        "local_density_9x9": float(loc),
    }
    return rec

# ------------------ Main run loop ------------------
def run_simulation(cfg: RunConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    out_dir = make_run_folder(cfg, rng)
    write_metadata(out_dir, cfg)

    grid = generate_forest(cfg.N, cfg.initial_density_t, rng, cfg.distribution)

    csv_path = os.path.join(out_dir, "fires.csv")
    cols = [
        "lightning_id",
        "rho_before","rho_after","fire_size","fire_radius",
        "local_density_9x9","ignition_x","ignition_y",
        "perimeter","gyration_radius","bbox_aspect",
        "spans_vertical","spans_horizontal",
        "N","theta","initial_density_t","distribution","seed",
        "run_dir","is_post_transient"
    ]
    buffer = []
    transient_cut = int(cfg.total_lightnings * cfg.transient_fraction)

    for L_idx in range(cfg.total_lightnings):
        rec = ds_step(cfg.theta, grid, rng)

        row = {
            "lightning_id": L_idx,
            **rec,
            "N": cfg.N,
            "theta": cfg.theta,
            "initial_density_t": cfg.initial_density_t,
            "distribution": cfg.distribution,
            "seed": cfg.seed,
            "run_dir": out_dir,
            "is_post_transient": (L_idx >= transient_cut),
        }
        buffer.append(row)

        if (L_idx + 1) % SAVE_EVERY == 0:
            pd.DataFrame(buffer, columns=cols).to_csv(
                csv_path, mode="a", index=False, header=not os.path.exists(csv_path)
            )
            buffer = []

    if buffer:
        pd.DataFrame(buffer, columns=cols).to_csv(
            csv_path, mode="a", index=False, header=not os.path.exists(csv_path)
        )

    print(f"[DONE] {cfg.distribution} t={cfg.initial_density_t:.2f} theta={cfg.theta} N={cfg.N} -> {csv_path}")
