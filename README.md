# Forest-Fire Model: Local Connectivity and Fire-Size Statistics

Code and data for the paper *A Study of the Drossel–Schwabl Forest-Fire Model: Local Connectivity and Fire-Size Statistics* (Aryan Mago), submitted to the Oxford Journal of Student Scholarship.

The model is a two-dimensional stochastic forest-fire model with tree regrowth and lightning ignition. Each lightning event is preceded by θ random regrowth attempts, then a random strike; if the strike lands on a tree, the whole four-neighbour connected tree cluster burns. The study measures a local-density threshold at ignition, how regrowth rate raises large-fire risk, how long the model retains memory of its initial condition, and how results collapse across system sizes on φ = θ/N².

## Requirements

Python 3.9+ with `numpy`, `pandas`, `scipy`, and `matplotlib`:

```
pip install numpy pandas scipy matplotlib
```

## What's here

### `code/`
- **`ff_fast.py`** — the simulator used for all results in the paper. Draws the initial-lattice random stream and the event-loop random stream independently, and keys the event stream on the layout, so runs that differ only in initial layout are genuinely independent. Burns are found by connected-component labelling (verified bit-identical to an explicit flood fill).
- **`runner.py`** — resumable driver that produces the two result files below. Run `python runner.py A` for the layout/threshold experiment and `python runner.py B` for the memory-erasure experiment. Each call does a chunk of work and can be re-run until it prints `DONE`. Also contains the exact threshold ("knee") estimator used in the paper.
- **`forest_fire.py`** — the original simulator, kept for provenance. It shares one random stream between lattice setup and the event loop; see the note at the top of the file for why that matters and why the paper's results use `ff_fast.py` instead.
- **`run_matrix.py`** — the original parameter-sweep driver for `forest_fire.py`.

### `data/`
- **`layout_and_threshold_runs.csv`** — 72 independently seeded runs (N=128, three θ, four layouts, six seeds). Underlies the threshold result and the layout-effect null (Figure 4, Figure 7a).
- **`memory_erasure_runs.csv`** — 210 runs measuring how many events two differently-initialized lattices, driven by a common random stream, take to converge (Figure 7b).
- **`mainsweep_exceedance_A.csv`, `mainsweep_exceedance_B.csv`** — fire-size exceedance probabilities from the main parameter sweep. Underlie Tables 1 and 2 and Figures 5, 6, 8.
- **`mainsweep_knees_A.csv`, `mainsweep_knees_B.csv`** — threshold extractions from the main sweep (Figure 9).

### `figures/`
`fig1.png`–`fig9.png`, matching the figure numbers in the paper.

## Reproducing the results

```
cd code
python runner.py A     # repeat until it prints "DONE expA"
python runner.py B     # repeat until it prints "DONE expB"
```

Output CSVs are written to `code/rerun_out/`. They match the files in `data/`.

## License

Released for review and reuse. If you use this code or data, please cite the paper.
