#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gan_spatiotemporal_ids.config import apply_overrides, load_config
from gan_spatiotemporal_ids.engine import evaluate_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained intrusion detector.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    print(json.dumps(evaluate_detector(config, args.checkpoint), indent=2))


if __name__ == "__main__":
    main()

