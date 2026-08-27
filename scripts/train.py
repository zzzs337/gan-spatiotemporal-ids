#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gan_spatiotemporal_ids.config import apply_overrides, load_config
from gan_spatiotemporal_ids.engine import train_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the intrusion detector.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    print(f"Best checkpoint: {train_detector(config)}")


if __name__ == "__main__":
    main()

