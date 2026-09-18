"""
DragonSR quickstart: recover the ideal-gas law  V = n R T / P  from synthetic data.

What this script does
---------------------
1. Generates 2 000 samples (P, n, T) -> V with R = 8.314 and writes them to a CSV.
2. Writes a small DragonSR configuration (search budget deliberately reduced
   so the example runs in a few minutes on a laptop CPU).
3. Runs the DragonSR pipeline on that CSV through the same CLI used for the
   paper's benchmarks.
4. Prints the best formula found and where the full outputs are.

Run it from anywhere once the package is installed:

    python examples/quickstart.py            # ~2-5 min on CPU
    python examples/quickstart.py --full     # paper-scale budget (10k evaluations)

Outputs land in ./quickstart_runs/ (results.json, HTML leaderboard, per-run logs).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make_dataset(path: Path, n_samples: int = 2000, seed: int = 42) -> None:
    R = 8.314
    rng = np.random.default_rng(seed)
    P = rng.uniform(1e4, 1e6, n_samples)
    n = rng.uniform(0.1, 10, n_samples)
    T = rng.uniform(200, 1000, n_samples)
    df = pd.DataFrame({"P": P, "n": n, "T": T})
    df["V"] = n * R * T / P
    df.to_csv(path, index=False)


def make_config(path: Path, full: bool) -> None:
    budget = (
        "N_ITERATIONS = 10000\nK_INIT = 30\nMAX_COMPLEXITY = 55\nT_PER_LEVEL = 200\n"
        if full
        else "N_ITERATIONS = 300\nK_INIT = 20\nMAX_COMPLEXITY = 12\nT_PER_LEVEL = 50\n"
    )
    path.write_text(
        "# Target column of the CSV\n"
        "TARGETS = V\n"
        "N_RUNS = 1\n"
        "INIT_STRATEGIES = random\n"
        "RANDOM_SEED = 42\n"
        "NOISE_STD = 0.0\n"
        "N_TOP_FEATURES = 3\n"
        + budget +
        "SEARCH_LOSS = corr\n"
        "SUBSAMPLING_RATIO = 1\n"
        "Paths.OUTPUT_DIR = quickstart_runs\n"
        "\n"
        "# Method-level options\n"
        "parallel_mode = smart\n"
        "loss_mode = channel\n"
        "var_aug = true\n"
        "add_noise = false\n"
        "denoiser = gpr\n"
        "sampling = false\n"
        "optimize_constants = true\n"
        "optimizer = dichotomy\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="use the paper-scale search budget")
    parser.add_argument("--workdir", default=".", help="where to write the CSV, config and outputs")
    args = parser.parse_args()

    work = Path(args.workdir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    csv_path = work / "ideal_gas.csv"
    cfg_path = work / "quickstart_config.txt"
    make_dataset(csv_path)
    make_config(cfg_path, args.full)
    print(f"Dataset : {csv_path}\nConfig  : {cfg_path}\n")

    cmd = [sys.executable, "-m", "dragonsr", "--run_dragonsr",
           "--data_path", str(csv_path), "--config_file_path", str(cfg_path)]
    print("Running :", " ".join(cmd), "\n")
    rc = subprocess.run(cmd, cwd=str(work)).returncode
    if rc != 0:
        print(f"\nDragonSR exited with code {rc}")
        return rc

    results = json.loads((work / "quickstart_runs" / "results.json").read_text())
    best = min(results, key=lambda r: r.get("loss", float("inf")))
    print("\n" + "=" * 70)
    print("Ground truth : V = n * 8.314 * T / P")
    print(f"DragonSR     : V = {best['formula']}")
    print(f"R^2          : {best.get('r2', float('nan')):.6f}")
    print(f"Leaderboard  : {work / 'quickstart_runs' / 'leaderboard_standalone.html'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
