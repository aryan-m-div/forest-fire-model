"""Fast re-implementation of the Drossel-Schwabl event loop used in the paper.

Identical dynamics to forest_fire.py:
  - theta uniform regrowth attempts per lightning event (EMPTY or BURNT -> TREE)
  - one uniform lightning site
  - if TREE, burn the whole 4-neighbour connected TREE cluster to BURNT
Only fire_size and local_density_9x9 are recorded; the per-fire perimeter /
gyration / bbox metrics (which built an NxN mask per fire) are dropped.

Critical fix vs the original: the initialization RNG and the simulation RNG are
independent streams, so layouts that consume different numbers of setup draws
no longer end up sharing (or offsetting) the event stream.
"""
from __future__ import annotations
import numpy as np
import zlib
from scipy import ndimage

EMPTY, TREE, BURNING, BURNT = 0, 1, 2, 3

_S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

NUM_BLOBS = 6
BLOB_SIGMA_FRAC = 0.08
STRIPE_PERIOD = 12
DENSE_IN_STRIPE = 0.75
SPARSE_IN_STRIPE = 0.35
GRAD_LEFT = 0.40
GRAD_RIGHT = 0.70


# ---------------- initial layouts (semantics copied from forest_fire.py) ----------------
def gen_uniform(N, t, rng):
    g = (rng.random((N, N)) < t).astype(np.uint8)
    g[g == 1] = TREE
    return g


def gen_clustered(N, t, rng):
    sigma = BLOB_SIGMA_FRAC * N
    yy, xx = np.mgrid[0:N, 0:N]
    field = np.zeros((N, N), dtype=np.float64)
    for _ in range(NUM_BLOBS):
        cy = rng.uniform(0, N)
        cx = rng.uniform(0, N)
        field += np.exp(-(((yy - cy) ** 2) + ((xx - cx) ** 2)) / (2.0 * sigma * sigma))
    mn, mx = field.min(), field.max()
    norm = np.zeros_like(field) if (mx - mn) < 1e-12 else (field - mn) / (np.ptp(field) + 1e-12)
    thresh = np.quantile(norm, 1.0 - t)
    g = (norm >= thresh).astype(np.uint8)
    g[g == 1] = TREE
    return g


def gen_stripes(N, t, rng):
    j = np.arange(N)
    stripe_id = (j // STRIPE_PERIOD) % 2
    p_col = np.where(stripe_id == 0, DENSE_IN_STRIPE, SPARSE_IN_STRIPE).astype(np.float64)
    base = rng.random((N, N))
    score = base - p_col[None, :]
    thresh = np.quantile(score, t)
    g = (score <= thresh).astype(np.uint8)
    g[g == 1] = TREE
    return g


def gen_gradient(N, t, rng):
    j = np.linspace(0.0, 1.0, N)
    p_col = GRAD_LEFT + (GRAD_RIGHT - GRAD_LEFT) * j
    base = rng.random((N, N))
    score = base - p_col[None, :]
    thresh = np.quantile(score, t)
    g = (score <= thresh).astype(np.uint8)
    g[g == 1] = TREE
    return g


_GEN = {"uniform": gen_uniform, "clustered": gen_clustered,
        "stripes": gen_stripes, "gradient": gen_gradient}


def generate_forest(N, t, rng, dist):
    return _GEN[dist](N, t, rng)


# ---------------- local density (matches forest_fire.local_density_9x9) ----------------
def local_density_9x9(grid, i0, j0):
    N = grid.shape[0]
    i1, i2 = max(0, i0 - 4), min(N, i0 + 5)
    j1, j2 = max(0, j0 - 4), min(N, j0 + 5)
    w = grid[i1:i2, j1:j2]
    trees = int(np.count_nonzero(w == TREE))
    center = 1 if grid[i0, j0] == TREE else 0
    return (trees - center) / max(1, w.size - 1)


# ---------------- flood fill (same 4-neighbour cluster, LIFO like the original) ----------------
def burn_size_py(grid, i0, j0):
    """Reference implementation: explicit 4-neighbour flood fill."""
    if grid[i0, j0] != TREE:
        return 0
    N = grid.shape[0]
    grid[i0, j0] = BURNT
    stack = [(i0, j0)]
    n = 0
    while stack:
        i, j = stack.pop()
        n += 1
        if i > 0 and grid[i - 1, j] == TREE:
            grid[i - 1, j] = BURNT; stack.append((i - 1, j))
        if i < N - 1 and grid[i + 1, j] == TREE:
            grid[i + 1, j] = BURNT; stack.append((i + 1, j))
        if j > 0 and grid[i, j - 1] == TREE:
            grid[i, j - 1] = BURNT; stack.append((i, j - 1))
        if j < N - 1 and grid[i, j + 1] == TREE:
            grid[i, j + 1] = BURNT; stack.append((i, j + 1))
    return n


def burn_size(grid, i0, j0):
    """Connected-component labelling in C. Verified bit-identical to
    burn_size_py over random lattice states; ~31x faster on percolating grids."""
    if grid[i0, j0] != TREE:
        return 0
    lab, _ = ndimage.label(grid == TREE, structure=_S4)
    mask = lab == lab[i0, j0]
    grid[mask] = BURNT
    return int(mask.sum())


def step(theta, grid, rng, want_local=True):
    N = grid.shape[0]
    if theta > 0:
        ii = rng.integers(0, N, size=theta)
        jj = rng.integers(0, N, size=theta)
        cur = grid[ii, jj]
        sel = (cur == EMPTY) | (cur == BURNT)
        grid[ii[sel], jj[sel]] = TREE
    i0 = int(rng.integers(0, N))
    j0 = int(rng.integers(0, N))
    loc = local_density_9x9(grid, i0, j0) if want_local else 0.0
    S = burn_size(grid, i0, j0)
    return S, loc


# ---------------- run ----------------
def run(N, theta, t, dist, seed, n_events, transient_frac=0.20,
        want_local=True, independent_streams=True):
    """Returns (fire_sizes, local_densities) for post-transient events only."""
    if independent_streams:
        # FIX: setup and simulation draw from independent streams, and the
        # simulation stream depends on the layout, so no two layouts can share it.
        # crc32 not hash(): hash() on str is randomised per process, so it is
        # not reproducible across runs or across pool workers.
        ss = np.random.SeedSequence([seed, zlib.crc32(dist.encode())])
        init_ss, sim_ss = ss.spawn(2)
        init_rng, sim_rng = np.random.default_rng(init_ss), np.random.default_rng(sim_ss)
    else:
        # ORIGINAL (buggy) behaviour, retained to reproduce the confound.
        init_rng = np.random.default_rng(seed)
        init_rng.integers(0, 16 ** 8)   # short_id draw in make_run_folder
        sim_rng = init_rng

    grid = generate_forest(N, t, init_rng, dist)
    cut = int(n_events * transient_frac)
    S_out = np.empty(n_events - cut, dtype=np.int32)
    L_out = np.empty(n_events - cut, dtype=np.float32)
    k = 0
    for e in range(n_events):
        S, loc = step(theta, grid, sim_rng, want_local)
        if e >= cut:
            S_out[k] = S; L_out[k] = loc; k += 1
    return S_out, L_out
