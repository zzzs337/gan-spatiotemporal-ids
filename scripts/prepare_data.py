#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gan_spatiotemporal_ids.config import load_config
from gan_spatiotemporal_ids.data import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an intrusion detection dataset.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    output_dir = prepare_dataset(load_config(args.config))
    print(f"Prepared dataset: {output_dir}")


if __name__ == "__main__":
    main()

