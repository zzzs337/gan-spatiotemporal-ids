from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import resolve_path
from .preprocessing import TrafficPreprocessor, infer_categorical_columns, make_windows
from .utils import ensure_directory, write_json


NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

NSL_KDD_ATTACK_GROUPS = {
    "DoS": {
        "apache2", "back", "land", "mailbomb", "neptune", "pod", "processtable",
        "smurf", "teardrop", "udpstorm",
    },
    "Probe": {"ipsweep", "mscan", "nmap", "portsweep", "saint", "satan"},
    "R2L": {
        "ftp_write", "guess_passwd", "httptunnel", "imap", "multihop", "named",
        "phf", "sendmail", "snmpgetattack", "snmpguess", "spy", "warezclient",
        "warezmaster", "xlock", "xsnoop",
    },
    "U2R": {"buffer_overflow", "loadmodule", "perl", "ps", "rootkit", "sqlattack", "xterm"},
}


def read_frame(path: Path, dataset_name: str, csv_options: dict) -> pd.DataFrame:
    options = dict(csv_options)
    if dataset_name == "nsl_kdd" and not options.get("names"):
        options.setdefault("header", None)
        options["names"] = NSL_KDD_COLUMNS
    return pd.read_csv(path, **options)


def normalize_labels(frame: pd.DataFrame, dataset_name: str, label_column: str) -> pd.DataFrame:
    frame = frame.copy()
    labels = frame[label_column].fillna("Normal").astype(str).str.strip()
    if dataset_name == "nsl_kdd":
        labels = labels.str.rstrip(".").str.lower()
        mapping = {attack: group for group, attacks in NSL_KDD_ATTACK_GROUPS.items() for attack in attacks}
        frame[label_column] = labels.map(lambda value: "Normal" if value == "normal" else mapping.get(value, value))
    elif dataset_name == "cicids2017":
        def cic_group(value: str) -> str:
            lower = value.lower()
            if lower in {"benign", "be-nign"}:
                return "BENIGN"
            if "heartbleed" in lower:
                return "Heartbleed"
            if "infiltration" in lower:
                return "Infiltration"
            if "web attack" in lower:
                return "Web Attack"
            if "patator" in lower:
                return "Patator"
            if "portscan" in lower or "port scan" in lower:
                return "PortScan"
            if "ddos" in lower:
                return "DDoS"
            if "dos" in lower:
                return "DoS"
            if "bot" in lower:
                return "Bot"
            return value

        frame[label_column] = labels.map(cic_group)
    else:
        frame[label_column] = labels
    return frame


def load_raw_splits(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_cfg = config["data"]
    name = data_cfg["dataset"]
    csv_options = data_cfg.get("csv_options", {})
    train_path = resolve_path(config, data_cfg["train_path"])
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    label_column = data_cfg["label_column"]
    frame = normalize_labels(read_frame(train_path, name, csv_options), name, label_column)
    test_value = data_cfg.get("test_path")
    if test_value:
        test_path = resolve_path(config, test_value)
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")
        test_frame = normalize_labels(read_frame(test_path, name, csv_options), name, label_column)
        return frame, test_frame
    train, test = train_test_split(
        frame,
        test_size=float(data_cfg.get("test_size", 0.4)),
        random_state=int(config["experiment"]["seed"]),
        stratify=frame[label_column].astype(str),
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def prepare_dataset(config: dict) -> Path:
    data_cfg = config["data"]
    full_train_frame, test_frame = load_raw_splits(config)
    label_column = data_cfg["label_column"]
    train_frame, validation_frame = train_test_split(
        full_train_frame,
        test_size=float(data_cfg.get("validation_size", 0.2)),
        random_state=int(config["experiment"]["seed"]),
        stratify=full_train_frame[label_column].astype(str),
    )
    train_frame = train_frame.reset_index(drop=True)
    validation_frame = validation_frame.reset_index(drop=True)
    categorical = data_cfg.get("categorical_columns") or infer_categorical_columns(
        train_frame, [label_column]
    )
    preprocessor = TrafficPreprocessor(label_column, list(categorical)).fit(train_frame)
    train_x, train_y = preprocessor.transform(train_frame)
    validation_x, validation_y = preprocessor.transform(validation_frame)
    test_x, test_y = preprocessor.transform(test_frame)
    window_size = int(data_cfg.get("window_size", 4))
    stride = int(data_cfg.get("window_stride", 1))
    train_windows, train_targets = make_windows(train_x, train_y, window_size, stride)
    validation_windows, validation_targets = make_windows(
        validation_x, validation_y, window_size, stride
    )
    test_windows, test_targets = make_windows(test_x, test_y, window_size, stride)
    output_dir = ensure_directory(resolve_path(config, data_cfg["processed_dir"]))
    np.savez_compressed(
        output_dir / "dataset.npz",
        train_x=train_windows,
        train_y=train_targets,
        validation_x=validation_windows,
        validation_y=validation_targets,
        test_x=test_windows,
        test_y=test_targets,
    )
    preprocessor.save(output_dir / "preprocessor.joblib")
    write_json(
        output_dir / "metadata.json",
        {
            "dataset": data_cfg["dataset"],
            "classes": preprocessor.classes,
            "feature_count": int(train_windows.shape[-1]),
            "window_size": window_size,
            "train_windows": int(len(train_windows)),
            "validation_windows": int(len(validation_windows)),
            "test_windows": int(len(test_windows)),
        },
    )
    return output_dir


def load_prepared(config: dict) -> tuple[np.ndarray, ...]:
    path = resolve_path(config, config["data"]["processed_dir"]) / "dataset.npz"
    if not path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {path}. Run prepare_data.py first.")
    with np.load(path) as data:
        return (
            data["train_x"],
            data["train_y"],
            data["validation_x"],
            data["validation_y"],
            data["test_x"],
            data["test_y"],
        )
