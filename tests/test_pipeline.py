from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gan_spatiotemporal_ids.config import apply_overrides, load_config, resolve_path
from gan_spatiotemporal_ids.metrics import classification_metrics
from gan_spatiotemporal_ids.preprocessing import TrafficPreprocessor, make_windows


class PipelineTest(unittest.TestCase):
    def test_config_loading_and_override(self):
        config = load_config(PROJECT_ROOT / "configs" / "unsw_nb15.yaml")
        updated = apply_overrides(config, ["training.learning_rate=0.01"])
        self.assertEqual(updated["training"]["learning_rate"], 0.01)
        self.assertEqual(config["training"]["learning_rate"], 0.001)
        experiment = load_config(PROJECT_ROOT / "configs" / "experiments" / "full.yaml")
        self.assertEqual(resolve_path(experiment, "outputs/example"), PROJECT_ROOT / "outputs" / "example")

    def test_preprocessing_windows_and_metrics(self):
        frame = pd.DataFrame(
            {
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "protocol": ["tcp", "udp", "tcp", "udp", "tcp", "udp"],
                "label": ["normal", "attack", "normal", "attack", "normal", "attack"],
            }
        )
        processor = TrafficPreprocessor("label", ["protocol"]).fit(frame)
        features, labels = processor.transform(frame)
        windows, targets = make_windows(features, labels, window_size=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preprocessor.joblib"
            processor.save(path)
            restored = TrafficPreprocessor.load(path)
        self.assertEqual(windows.shape, (3, 4, features.shape[1]))
        np.testing.assert_array_equal(targets, labels[3:])
        self.assertEqual(restored.classes, processor.classes)
        self.assertEqual(classification_metrics(targets, targets)["accuracy"], 1.0)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
    def test_tensorflow_forward_backward_and_checkpoint(self):
        import tensorflow as tf

        from gan_spatiotemporal_ids.models import build_detector

        config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
        config["model"]["hidden_dim"] = 8
        config["model"]["num_blocks"] = 1
        model = build_detector(config, feature_count=6, class_count=3)
        optimizer = tf.keras.optimizers.Adam(0.001)
        features = tf.random.normal((5, 4, 6))
        labels = tf.constant([0, 1, 2, 1, 0])
        with tf.GradientTape() as tape:
            logits = model(features, training=True)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    labels, logits, from_logits=True
                )
            )
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "smoke.weights.h5"
            model.save_weights(checkpoint)
            restored = build_detector(config, feature_count=6, class_count=3)
            restored(features, training=False)
            restored.load_weights(checkpoint)
            self.assertTrue(checkpoint.exists())
        self.assertEqual(logits.shape, (5, 3))


if __name__ == "__main__":
    unittest.main()
