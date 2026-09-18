# Benchmarks

| File | Purpose |
|---|---|
| `configs/dragonsr_paper.txt` | DragonSR, paper budget: 10 000 evaluations, 10 runs, GPR denoiser, variable augmentation |
| `configs/dragonsr_quick.txt` | Same pipeline, 500 evaluations, 2 runs, 10 % subsampling — for smoke tests |
| `configs/pysr_paper.txt` | PySR baseline, paper budget: 2 000 iterations × 5 populations × 120 |
| `configs/pysr_quick.txt` | PySR baseline, tiny budget |
| `run_plan.txt` | Example plan: one line per (engine, target, config) |
| `run_batch.py` | Runs a plan sequentially, archives HTML and rotates `leaderboard_runs/` |

Targets are set per line in the plan (`dragonsr|n5|configs/dragonsr_paper.txt`), which
overrides any `TARGETS` in the config. Config paths are relative to the plan file.

```bash
python benchmarks/run_batch.py --plan-file benchmarks/run_plan.txt
python benchmarks/run_batch.py --plan-file my_plan.txt --work-dir runs/2026-09   # outputs elsewhere
```

Every key of `dragonsr/Config.py` can be overridden in a config file
(`Experiment.NOISE_STD = 0.05`, `Loss.COMPOSITION_PENALTY = 0.01`, `PySR.MAXSIZE = 30`, …);
un-prefixed keys are looked up in `Experiment`, `Dragon`, `PySR`, `Paths`, `Loss`, `OLS`,
`MCDropout`, `Sampling` in that order.

The run-level outputs behind the paper's Table 2 are on Zenodo:
https://doi.org/10.5281/zenodo.22146752
