from __future__ import annotations

import tensorflow as tf


class SpatiotemporalResidualBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_dim: int,
        kernel_sizes: list[int],
        use_multiscale: bool = True,
        use_gru: bool = True,
        use_residual: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.use_multiscale = use_multiscale
        self.use_gru = use_gru
        self.use_residual = use_residual
        self.input_projection = tf.keras.layers.Dense(hidden_dim)
        active_kernels = kernel_sizes if use_multiscale else [kernel_sizes[0]]
        self.convolutions = [
            tf.keras.layers.Conv1D(hidden_dim, kernel, padding="same")
            for kernel in active_kernels
        ]
        self.spatial_bn = tf.keras.layers.BatchNormalization()
        self.temporal_gru = (
            tf.keras.layers.GRU(hidden_dim, return_sequences=True) if use_gru else None
        )
        self.temporal_bn = tf.keras.layers.BatchNormalization()
        self.fusion_bn = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=False):
        projected = self.input_projection(inputs)
        spatial = tf.add_n([layer(projected) for layer in self.convolutions])
        spatial = tf.nn.relu(self.spatial_bn(spatial, training=training))
        if self.temporal_gru is not None:
            temporal = self.temporal_gru(projected, training=training)
            temporal = tf.nn.relu(self.temporal_bn(temporal, training=training))
            fused = spatial + temporal
        else:
            fused = spatial
        fused = tf.nn.relu(self.fusion_bn(fused, training=training))
        fused = self.dropout(fused, training=training)
        return projected + fused if self.use_residual else fused


def build_detector(config: dict, feature_count: int, class_count: int) -> tf.keras.Model:
    model_cfg = config["model"]
    inputs = tf.keras.Input(
        shape=(int(config["data"]["window_size"]), feature_count), name="traffic_window"
    )
    x = inputs
    for index in range(int(model_cfg["num_blocks"])):
        x = SpatiotemporalResidualBlock(
            hidden_dim=int(model_cfg["hidden_dim"]),
            kernel_sizes=list(model_cfg["kernel_sizes"]),
            use_multiscale=bool(model_cfg.get("use_multiscale", True)),
            use_gru=bool(model_cfg.get("use_gru", True)),
            use_residual=bool(model_cfg.get("use_residual", True)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            name=f"spatiotemporal_block_{index + 1}",
        )(x)
    x = x[:, -1, :]
    outputs = tf.keras.layers.Dense(class_count, name="logits")(x)
    return tf.keras.Model(inputs, outputs, name="improved_spatiotemporal_resnet")


def build_generator(config: dict, feature_count: int) -> tf.keras.Model:
    gan_cfg = config["gan"]
    window_size = int(config["data"]["window_size"])
    inputs = tf.keras.Input(shape=(window_size, int(gan_cfg["noise_dim"])), name="noise")
    x = inputs
    for index in range(3):
        x = tf.keras.layers.LSTM(
            int(gan_cfg["hidden_dim"]), return_sequences=True, name=f"generator_lstm_{index + 1}"
        )(x)
    outputs = tf.keras.layers.Dense(feature_count, name="generated_features")(x)
    return tf.keras.Model(inputs, outputs, name="generator")


def build_discriminator(config: dict, feature_count: int) -> tf.keras.Model:
    gan_cfg = config["gan"]
    window_size = int(config["data"]["window_size"])
    inputs = tf.keras.Input(shape=(window_size, feature_count), name="traffic_window")
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(int(gan_cfg["hidden_dim"]), activation="relu")(x)
    x = tf.keras.layers.Dense(int(gan_cfg["hidden_dim"] // 2), activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, name="real_fake_logit")(x)
    return tf.keras.Model(inputs, outputs, name="discriminator")

