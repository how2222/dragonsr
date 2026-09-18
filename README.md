# DragonSR — Directed Acyclic Graph Search for Equation Discovery

**Symbolic regression reframed as the optimisation of a directed acyclic graph.**
Instead of evolving expression trees, DragonSR searches over DAGs whose edges carry
several symbolic *channels* in parallel, keeps constants out of the search (they are
recovered afterwards by linear least squares), and uses the deterministic
Mutant-UCB optimiser of the [DRAGON](https://github.com/JulieKeisler/DRAGON) framework.

This repository is the self-contained, runnable companion of the paper:

> **DragonSR: Directed Acyclic Graph Search for Equation Discovery**
> Elyas Chikhaoui, Julie Keisler, Anastase Charantonis — INRIA, 2026.
> [PDF](paper/DragonSR.pdf) · run-level outputs on [Zenodo](https://doi.org/10.5281/zenodo.22146752)

It contains the method, the DragonSR-vs-PySR benchmark pipeline, the 27 benchmark
targets (including the Sentinel-2 remote-sensing data), and everything needed to run
DragonSR on your own CSV.

---

## Results in one table

Recovery rate over 10 independent runs (a run counts when the simplified expression is
equivalent to the ground truth). ε is the relative noise added to the target,
`y + N(0, ε·std(y))`. Full table: Table 2 of the paper.

| Target | PySR ε=0 | PySR ε≥1 % | DragonSR ε=0 | ε=1 % | ε=5 % | ε=10 % |
|---|---|---|---|---|---|---|
| Nguyen-8  `√x` | 10/10 | 0 | 9/10 | 10/10 | 9/10 | 8/10 |
| Nguyen-9  `sin x + sin y²` | 0/10 | 0 | 10/10 | 9/10 | 8/10 | 8/10 |
| Ideal gas `V = nRT/P` | 1/10 | 0 | 9/10 | 6/10 | 3/10 | 4/10 |
| Leavitt `M = −2.81 log₁₀P − 1.43` | 2/10 | 0 | 10/10 | 6/10 | 7/10 | 5/10 |
| Kepler's third law | 0/10 | 0 | 10/10 | 3/10 | 4/10 | 1/10 |
| SAVI (13 Sentinel-2 bands) | 0/10 | 0 | 10/10 | 9/10 | 10/10 | 8/10 |
| EVI2 (13 Sentinel-2 bands) | 0/10 | 0 | 9/10 | 8/10 | 3/10 | 3/10 |
| BSI (13 Sentinel-2 bands) | 0/10 | 0 | 9/10 | 6/10 | 3/10 | 9/10 |

In our protocol PySR recovers nothing on any of the 27 targets once noise is added,
while DragonSR degrades gracefully up to 10 % noise and is the only method recovering
the multi-variable remote-sensing indices. Both methods fail on targets that need
precise constants inside non-linearities (Newton, Planck, Schechter). The search
budgets of the two methods are not expressed in the same units (10 000 DAG evaluations
vs 2 000 PySR iterations × 5 populations × 120 individuals); see §6 of the paper.

---

## Quickstart (5 minutes, CPU)

```bash
git clone https://github.com/how2222/dragonsr.git
cd dragonsr
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .

python examples/quickstart.py
```

The example generates 2 000 samples of the ideal-gas law, runs DragonSR on them with a
reduced budget and prints the recovered formula:

```
Ground truth : V = n * 8.314 * T / P
DragonSR     : V = 8.314000e+00 * (n / P * T) + -9.648826e-17
R^2          : 1.000000
```

It also writes `quickstart_runs/leaderboard_standalone.html`, an interactive page
showing the DAG, the loss landscape and every candidate channel formula.

To compare against PySR as well, install the optional dependency
(`pip install -e ".[pysr]"`; PySR downloads Julia on first use).

---

## How it works

```
             ┌────────────────────────────────────────────────────────────┐
   X, y ───► │ 1. Preprocessing                                            │
             │    optional GPR denoising of y · variable augmentation      │
             │    (x², 1/x, log x, x·y, …) · XGBoost top-k feature ranking │
             └──────────────┬─────────────────────────────────────────────┘
                            ▼
             ┌────────────────────────────────────────────────────────────┐
             │ 2. DAG search (DRAGON, Mutant-UCB)                          │
             │    nodes = (combiner add/mul, primitive, params)            │
             │    primitives: SelectFeatures, Power x^{±1,±2,±3}, Ln, Exp, │
             │    Sin, Cos, Inverse, Negate, SumFeatures, ChannelBoost, … │
             │    edges carry multi-channel tensors; sub-expressions are   │
             │    stored once and reused                                   │
             │    loss: 1 − Pearson(ŷ, y) (affine-invariant) + composition │
             │    penalty; complexity levels grow during the search        │
             └──────────────┬─────────────────────────────────────────────┘
                            ▼
             ┌────────────────────────────────────────────────────────────┐
             │ 3. Constant recovery — linear least squares only            │
             │    sparse OLS over the output channels                      │
             │    nested OLS through a link function (log, sqrt, inverse)  │
             │    polynomial-rational OLS  y(1+Σbᵢφᵢ) = a₀+Σaᵢφᵢ            │
             └──────────────┬─────────────────────────────────────────────┘
                            ▼
                     formula, R², DAG, HTML leaderboard
```

Why it matters: a tree recomputes every repeated sub-expression, a DAG stores it once;
removing constants from the search turns a non-convex constant fit into a linear solve
per candidate, so the graph only has to find the *terms*.

---

## Repository layout

```
dragonsr/
├── dragon/                 DRAGON library (vendored from the sr fork, CeCILL-C):
│   └── search_space/bricks/symbolic_regression.py   the symbolic bricks
├── dragonsr/               the DragonSR pipeline and benchmark orchestrator
│   ├── Config.py           every default (budgets, losses, OLS, PySR, methods)
│   ├── main.py             CLI  →  python -m dragonsr  /  dragonsr
│   ├── searchspace/        DAG search space built from the bricks
│   ├── pipeline/           search loss, orchestrator, complexity schedule
│   ├── runner/             DragonSR worker, OLS post-processing, PySR runner
│   ├── dataprocessing/     benchmark targets, augmentation, denoising, features
│   ├── helpers/            text-config parser, stats, HTML leaderboard
│   └── data/6000_points.csv   6 000 Sentinel-2 samples (13 bands) for the indices
├── benchmarks/             configs and run plans used for the paper
├── examples/quickstart.py  ideal-gas example
└── paper/DragonSR.pdf
```

---

## Running the benchmarks

All commands work from any directory once the package is installed; outputs go to
`leaderboard_runs/` in the current directory (`Paths.OUTPUT_DIR`).

### Built-in targets

| Family | `TARGETS` ids |
|---|---|
| Nguyen | `n1` … `n12` |
| Physics | `hubble`, `newton`, `rydberg`, `idealgas`, `kepler`, `bode`, `schechter`, `leavitt`, `planck` |
| Sentinel-2 indices | `ndvi`, `savi`, `bsi`, `mndwi`, `bai`, `awei_sh`, `awei_nsh`, `wi2015`, `evi2`, `vari`, `nirv` |

Synthetic families are generated on the fly (6 000 samples, seed derived from
`RANDOM_SEED` + run id); the indices are computed from the bundled Sentinel-2 CSV.

### One target, one method

```bash
# DragonSR on Nguyen-5, paper budget
dragonsr --run_dragonsr --config_file_path benchmarks/configs/dragonsr_paper.txt   # add TARGETS = n5 to the file
# or override in a copy of the config; every key of Config.py can be set in the text file:
#   TARGETS = n5,n6        N_RUNS = 10        NOISE_STD = 0.05        SEARCH_LOSS = corr
#   Loss.COMPOSITION_PENALTY = 0.01           PySR.N_ITERATIONS = 2000

# PySR baseline
dragonsr --pysr-only --config_file_path benchmarks/configs/pysr_paper.txt

# DragonSR then PySR, into the same results.json
dragonsr --config_file_path benchmarks/configs/dragonsr_paper.txt
dragonsr --config_file_path benchmarks/configs/dragonsr_paper.txt --continue   # resume
```

### Many runs, unattended

`benchmarks/run_batch.py` executes a plan file line by line, archives each run's HTML and
rotates `leaderboard_runs/` so nothing is overwritten:

```bash
python benchmarks/run_batch.py --plan-file benchmarks/run_plan.txt
```

```
# engine|target|config[|label[|data.csv]]
dragonsr|n5|configs/dragonsr_paper.txt
pysr|n5|configs/pysr_paper.txt
```

### Rebuild the HTML leaderboard from existing results

```bash
dragonsr --leaderboard
```

### Your own dataset

Any CSV whose target is one column and whose other numeric columns are the inputs:

```bash
dragonsr --run_dragonsr --data_path my_data.csv --config_file_path my_config.txt   # with TARGETS = <column>
```

`examples/quickstart.py` is exactly this on a generated CSV.

### Outputs

`leaderboard_runs/results.json` is appended after every run: target, method, run id,
recovered formula (channel / OLS / nested / poly-rational variants), losses, R², DAG
description and SVG, loss history, runtime. `leaderboard_standalone.html` renders it.
Per-run logs and the best model are under `leaderboard_runs/<target>/<method>/run_<i>/`.

---

## Reproducing the paper

1. `pip install -e ".[paper]"` pins the exact versions used for the runs.
2. Use `benchmarks/configs/dragonsr_paper.txt` (10 000 evaluations, `K_INIT = 30`,
   `MAX_COMPLEXITY = 55`, Pearson search loss, GPR denoiser, variable augmentation)
   and `pysr_paper.txt` (2 000 iterations × 5 populations × 120).
3. Noise levels: `NOISE_STD ∈ {0, 0.01, 0.05, 0.10}`; 10 runs per (target, noise).
4. Runs are independent and were executed one at a time; the complete run-level outputs
   used for Table 2 are archived on Zenodo, organised as
   `method / noise condition / benchmark expression / independent run`.

A word of caution, also stated in the paper: the DragonSR and PySR budgets are not
matched in comparable units, no ablation isolates the individual components, and
recovery was judged by hand after simplification.

---

## Relationship with DRAGON

DragonSR was developed as an extension of DRAGON (Keisler et al., *JMLR* 2024) during a
research internship at INRIA Paris. The library code in `dragon/` is the exact version the
paper's runs used (the `dragonsr-experiments` branch of
[how2222/DRAGON](https://github.com/how2222/DRAGON), with unrelated experimental modules
removed). A cleaner integration of the symbolic-regression components into upstream
DRAGON, with tutorial notebooks, is being prepared by Julie Keisler on the
[`sr_to_dragon`](https://github.com/JulieKeisler/DRAGON/tree/sr_to_dragon) branch.

## Citation

```bibtex
@article{chikhaoui2026dragonsr,
  title   = {DragonSR: Directed Acyclic Graph Search for Equation Discovery},
  author  = {Chikhaoui, Elyas and Keisler, Julie and Charantonis, Anastase},
  year    = {2026},
  note    = {Preprint. Code: https://github.com/how2222/dragonsr, data: https://doi.org/10.5281/zenodo.22146752}
}
```

## License

CeCILL-C, inherited from DRAGON — see [LICENSE.txt](LICENSE.txt).
