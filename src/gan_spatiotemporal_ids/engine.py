from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import resolve_path
from .data import load_prepared
from .metrics import classification_metrics
from .models import build_detector, build_discriminator, build_generator
from .utils import ensure_directory, read_json, set_seed, write_json


def _dataset(array_x, array_y, batch_size: int, shuffle: bool, seed: int):
    dataset = tf.data.Dataset.from_tensor_slices((array_x, array_y))
    if shuffle:
        dataset = dataset.shuffle(min(len(array_x), 10000), seed=seed)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_gan_for_class(
    config: dict,
    real_windows: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    gan_cfg = config["gan"]
    generator = build_generator(config, real_windows.shape[-1])
    discriminator = build_discriminator(config, real_windows.shape[-1])
    generator_optimizer = tf.keras.optimizers.Adam(float(gan_cfg["learning_rate"]))
    discriminator_optimizer = tf.keras.optimizers.Adam(float(gan_cfg["learning_rate"]))
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    batch_size = int(gan_cfg["batch_size"])
    noise_dim = int(gan_cfg["noise_dim"])
    window_size = int(config["data"]["window_size"])
    dataset = tf.data.Dataset.from_tensor_slices(real_windows).shuffle(
        len(real_windows), seed=int(config["experiment"]["seed"])
    ).batch(batch_size, drop_remainder=False)

    for _ in range(int(gan_cfg["epochs"])):
        for real_batch in dataset:
            current_batch = tf.shape(real_batch)[0]
            noise = tf.random.normal((current_batch, window_size, noise_dim))
            with tf.GradientTape() as discriminator_tape:
                fake_batch = generator(noise, training=True)
                real_logits = discriminator(real_batch, training=True)
                fake_logits = discriminator(tf.stop_gradient(fake_batch), training=True)
                discriminator_loss = loss_fn(tf.ones_like(real_logits), real_logits)
                discriminator_loss += loss_fn(tf.zeros_like(fake_logits), fake_logits)
            gradients = discriminator_tape.gradient(
                discriminator_loss, discriminator.trainable_variables
            )
            discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

            noise = tf.random.normal((current_batch, window_size, noise_dim))
            with tf.GradientTape() as generator_tape:
                fake_batch = generator(noise, training=True)
                fake_logits = discriminator(fake_batch, training=False)
                generator_loss = loss_fn(tf.ones_like(fake_logits), fake_logits)
            gradients = generator_tape.gradient(generator_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

    generated_batches = []
    remaining = sample_count
    while remaining > 0:
        current = min(remaining, batch_size)
        noise = tf.random.normal((current, window_size, noise_dim))
        generated_batches.append(generator(noise, training=False).numpy())
        remaining -= current
    return np.concatenate(generated_batches, axis=0).astype(np.float32)


def augment_minority_classes(config: dict, train_x: np.ndarray, train_y: np.ndarray):
    if not config["gan"].get("enabled", True):
        return train_x, train_y
    labels, counts = np.unique(train_y, return_counts=True)
    target = int(config["gan"].get("target_samples") or counts.max())
    generated_x = [train_x]
    generated_y = [train_y]
    minimum_real = int(config["gan"].get("minimum_real_samples", 8))
    for label, count in zip(labels, counts):
        required = max(0, target - int(count))
        class_windows = train_x[train_y == label]
        if required == 0 or len(class_windows) < minimum_real:
            continue
        synthetic = train_gan_for_class(config, class_windows, required)
        generated_x.append(synthetic)
        generated_y.append(np.full(required, label, dtype=train_y.dtype))
    return np.concatenate(generated_x), np.concatenate(generated_y)


def train_detector(config: dict) -> Path:
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    train_x, train_y, validation_x, validation_y, test_x, test_y = load_prepared(config)
    train_x, train_y = augment_minority_classes(config, train_x, train_y)
    class_count = int(max(train_y.max(), test_y.max()) + 1)
    model = build_detector(config, train_x.shape[-1], class_count)
    optimizer = tf.keras.optimizers.Adam(float(config["training"]["learning_rate"]))
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    output_dir = ensure_directory(resolve_path(config, config["experiment"]["output_dir"]))
    weights_path = output_dir / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            weights_path, monitor="val_loss", save_best_only=True, save_weights_only=True
        ),
        tf.keras.callbacks.CSVLogger(output_dir / "training.csv"),
    ]
    model.fit(
        _dataset(train_x, train_y, int(config["training"]["batch_size"]), True, seed),
        validation_data=_dataset(
            validation_x,
            validation_y,
            int(config["training"]["batch_size"]),
            False,
            seed,
        ),
        epochs=int(config["training"]["epochs"]),
        callbacks=callbacks,
        verbose=2,
    )
    write_json(
        output_dir / "model_metadata.json",
        {"feature_count": int(train_x.shape[-1]), "class_count": class_count},
    )
    return weights_path


def evaluate_detector(config: dict, weights_path: str | Path | None = None) -> dict:
    _, _, _, _, test_x, test_y = load_prepared(config)
    output_dir = ensure_directory(resolve_path(config, config["experiment"]["output_dir"]))
    metadata = read_json(output_dir / "model_metadata.json")
    model = build_detector(config, metadata["feature_count"], metadata["class_count"])
    weights_path = Path(weights_path) if weights_path else output_dir / "best.weights.h5"
    model.load_weights(weights_path)
    started = time.perf_counter()
    logits = model.predict(test_x, batch_size=int(config["training"]["batch_size"]), verbose=0)
    elapsed = time.perf_counter() - started
    predictions = np.argmax(logits, axis=1)
    results = classification_metrics(test_y, predictions)
    results["inference_seconds"] = float(elapsed)
    results["samples_per_second"] = float(len(test_x) / elapsed) if elapsed else 0.0
    write_json(output_dir / "metrics.json", results)
    np.savez_compressed(output_dir / "predictions.npz", labels=test_y, predictions=predictions)
    return results
