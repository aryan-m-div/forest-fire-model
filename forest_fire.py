"""Resumable chunked runner. Each invocation does as much pending work as fits
in a time budget, appends results, and exits. Call repeatedly until DONE."""
import os, sys, time, itertools, zlib
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ff_fast as fx

OUT = os.path.join(HERE, "rerun_out")
os.makedirs(OUT, exist_ok=True)
BUDGET = float(os.environ.get("BUDGET", "32"))
WORKERS = 4

# ----------------------------- experiment A -----------------------------
A_N, A_T, A_EVENTS, A_TRANS, A_BINS = 128, 0.55, 40_000, 0.20, 60
A_THETAS = [250, 1000, 3000]
A_LAYOUTS = ["uniform", "clustered", "stripes", "gradient"]
A_SEEDS = [101, 202, 303, 404, 505, 606]


def binned_curve(S, L, bins=A_BINS):
    """Binned median fire size vs local density. Matches analyze_final.compute_binned_median."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(L, edges, right=False) - 1, 0, bins - 1)
    med = np.full(bins, np.nan)
    cnt = np.zeros(bins, dtype=np.int64)
    for b in range(bins):
        v = S[idx == b]
        cnt[b] = v.size
        if v.size:
            med[b] = np.median(v)
    return centres, med, cnt


def knee_simple(centers, medians, counts, min_bin_count=200, delta_min=50.0):
    """EXACT reimplementation of analyze_final.knee_simple (the estimator actually
    used to produce the published numbers): first bin whose median exceeds a
    low-density baseline by delta = max(50, 3*IQR)."""
    mask = (~np.isnan(medians)) & (counts >= min_bin_count)
    valid_idx = np.where(mask)[0]
    if valid_idx.size < 6:
        return np.nan, np.nan, int(valid_idx.size), False
    c, m = centers[valid_idx], medians[valid_idx]
    base_mask = c <= 0.4
    if base_mask.sum() < 3:
        base_mask = np.arange(len(c)) < 3
    base_vals = m[base_mask]
    if len(base_vals) < 3:
        return np.nan, np.nan, int(valid_idx.size), False
    baseline = float(np.median(base_vals))
    iqr = float(np.percentile(base_vals, 75) - np.percentile(base_vals, 25)) if len(base_vals) >= 4 else 0.0
    delta = max(50.0, 3 * iqr, delta_min)
    knee_local = None
    for i, val in enumerate(m):
        if val >= baseline + delta:
            if i < 2 or i > len(m) - 3:
                continue
            knee_local = i
            break
    if knee_local is None:
        return np.nan, np.nan, int(valid_idx.size), False
    ig = valid_idx[knee_local]
    left, right = valid_idx[knee_local - 2], valid_idx[knee_local + 2]
    strength = np.nan
    if not np.isnan(medians[left]) and not np.isnan(medians[right]) and centers[right] != centers[left]:
        strength = (medians[right] - medians[left]) / (centers[right] - centers[left])
    return float(centers[ig]), float(strength), int(valid_idx.size), True


def a_key(t, l, s):
    return f"{t}|{l}|{s}"


def a_work(theta, layout, seed):
    S, L = fx.run(A_N, theta, A_T, layout, seed, A_EVENTS, A_TRANS,
                  want_local=True, independent_streams=True)
    cen, med, cnt = binned_curve(S, L)
    kx, ks, nb, ok = knee_simple(cen, med, cnt)
    np.savez_compressed(os.path.join(OUT, f"curve_{theta}_{layout}_{seed}.npz"),
                        centres=cen, medians=med, counts=cnt)
    N2 = A_N * A_N
    return dict(key=a_key(theta, layout, seed), N=A_N, theta=theta, layout=layout,
                seed=seed, n_post=int(S.size), n_fire=int((S > 0).sum()),
                p_fire=float((S > 0).mean()),
                p_exc_005=float((S >= 0.05 * N2).mean()),
                p_exc_010=float((S >= 0.10 * N2).mean()),
                p_exc_010_cond=float((S[S > 0] >= 0.10 * N2).mean()),
                mean_S=float(S.mean()), knee_x=float(kx),
                knee_strength=float(ks), bins_used=int(nb), valid_knee=bool(ok))


# ----------------------------- experiment B -----------------------------
B_T, B_CAP = 0.55, 20_000
B_PAIRS = [("uniform", "stripes"), ("uniform", "gradient"), ("uniform", "clustered")]
B_SEEDS = [11, 22, 33, 44, 55]
B_CONFIGS = [(128, th) for th in [250, 500, 750, 1000, 1500, 2000, 3000]] + \
            [(256, th) for th in [1000, 2000, 3000, 4000, 6000, 9000, 12000]]


def b_key(N, th, pair, s):
    return f"{N}|{th}|{pair[0]}-{pair[1]}|{s}"


def b_work(N, theta, pair, seed):
    a, b = pair
    ga = fx.generate_forest(N, B_T, np.random.default_rng(np.random.SeedSequence([seed, 1])), a)
    gb = fx.generate_forest(N, B_T, np.random.default_rng(np.random.SeedSequence([seed, 2])), b)
    ss = np.random.SeedSequence([seed, 99])
    ra, rb = np.random.default_rng(ss), np.random.default_rng(ss)
    conv = None
    for e in range(1, B_CAP + 1):
        fx.step(theta, ga, ra, want_local=False)
        fx.step(theta, gb, rb, want_local=False)
        if e <= 30 or e % 10 == 0:
            if np.array_equal(ga, gb):
                conv = e
                break
    return dict(key=b_key(N, theta, pair, seed), N=N, theta=theta,
                phi=theta / (N * N), pair=f"{a}|{b}", seed=seed,
                converged_at=(conv if conv else np.nan), censored=bool(conv is None),
                final_diff=int(np.count_nonzero(ga != gb)))


# ----------------------------- driver -----------------------------
def main(which):
    path = os.path.join(OUT, f"exp{which}.csv")
    done = set()
    if os.path.exists(path):
        done = set(pd.read_csv(path)["key"].astype(str))

    if which == "A":
        allw = [(t, l, s) for t in A_THETAS for l in A_LAYOUTS for s in A_SEEDS]
        pending = [w for w in allw if a_key(*w) not in done]
        fn = a_work
    else:
        allw = [(N, th, p, s) for (N, th) in B_CONFIGS for p in B_PAIRS for s in B_SEEDS]
        pending = [w for w in allw if b_key(*w) not in done]
        fn = b_work

    if not pending:
        print(f"DONE exp{which}: {len(done)}/{len(allw)}")
        return

    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs, it = {}, iter(pending)
        for _ in range(WORKERS):
            try:
                w = next(it); futs[ex.submit(fn, *w)] = w
            except StopIteration:
                break
        while futs:
            for f in as_completed(list(futs), timeout=None):
                rows.append(f.result()); futs.pop(f)
                if time.time() - t0 < BUDGET:
                    try:
                        w = next(it); futs[ex.submit(fn, *w)] = w
                    except StopIteration:
                        pass
                break
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    n = len(done) + len(rows)
    print(f"exp{which}: {n}/{len(allw)} complete (+{len(rows)} this call, {time.time()-t0:.0f}s)")
    if n >= len(allw):
        print(f"DONE exp{which}")


if __name__ == "__main__":
    main(sys.argv[1].upper())
