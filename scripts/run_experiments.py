#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-aligned experiment variants.")
    parser.add_argument("--suite", choices=["ablation", "learning_rate"], required=True)
    parser.add_argument("--config", default="configs/unsw_nb15.yaml")
    args = parser.parse_args()
    python = sys.executable
    if args.suite == "ablation":
        configs = [
            "configs/experiments/residual_only.yaml",
            "configs/experiments/spatiotemporal.yaml",
            "configs/experiments/improved_no_gan.yaml",
            "configs/experiments/full.yaml",
        ]
        for config in configs:
            run([python, "scripts/train.py", "--config", config])
            run([python, "scripts/evaluate.py", "--config", config])
    else:
        for learning_rate in [0.01, 0.001, 0.0001]:
            suffix = str(learning_rate).replace(".", "p")
            overrides = [
                f"training.learning_rate={learning_rate}",
                f"experiment.output_dir=outputs/learning_rate/lr_{suffix}",
            ]
            command = [python, "scripts/train.py", "--config", args.config]
            for override in overrides:
                command.extend(["--set", override])
            run(command)
            command = [python, "scripts/evaluate.py", "--config", args.config]
            for override in overrides:
                command.extend(["--set", override])
            run(command)


if __name__ == "__main__":
    main()

