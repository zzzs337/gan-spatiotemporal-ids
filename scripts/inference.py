#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gan_spatiotemporal_ids.config import load_config, resolve_path
from gan_spatiotemporal_ids.models import build_detector
from gan_spatiotemporal_ids.preprocessing import TrafficPreprocessor, make_windows
from gan_spatiotemporal_ids.utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify network traffic records.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    processed_dir = resolve_path(config, config["data"]["processed_dir"])
    output_dir = resolve_path(config, config["experiment"]["output_dir"])
    preprocessor = TrafficPreprocessor.load(processed_dir / "preprocessor.joblib")
    frame = pd.read_csv(args.input, **config["data"].get("csv_options", {}))
    features = preprocessor.transform_features(frame)
    dummy_labels = np.zeros(len(features), dtype=np.int64)
    windows, _ = make_windows(
        features,
        dummy_labels,
        int(config["data"]["window_size"]),
        int(config["data"].get("window_stride", 1)),
    )
    metadata = read_json(output_dir / "model_metadata.json")
    model = build_detector(config, metadata["feature_count"], metadata["class_count"])
    model.load_weights(args.checkpoint or output_dir / "best.weights.h5")
    logits = model.predict(windows, batch_size=int(config["training"]["batch_size"]), verbose=0)
    class_ids = np.argmax(logits, axis=1)
    labels = preprocessor.label_encoder.inverse_transform(class_ids)
    probabilities = np.max(tf_softmax(logits), axis=1)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["window_index", "prediction", "confidence"])
        writer.writeheader()
        for index, (label, probability) in enumerate(zip(labels, probabilities)):
            writer.writerow(
                {"window_index": index, "prediction": label, "confidence": float(probability)}
            )
    print(f"Predictions: {output_path}")


def tf_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    main()

