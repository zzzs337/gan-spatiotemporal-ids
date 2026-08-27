# GAN-Assisted Spatiotemporal Residual Network for Intrusion Detection

This repository implements a network intrusion detection pipeline that combines GAN-based minority-class resampling with an improved spatiotemporal residual classifier. The classifier fuses multi-scale one-dimensional convolutions with GRU temporal features and identity mappings.

## Method

The pipeline follows this sequence:

```text
CSV traffic records
→ missing-value handling, one-hot encoding, and Z-score scaling
→ four-record traffic windows
→ class-wise GAN resampling
→ multi-scale Conv1D and GRU feature fusion
→ residual blocks
→ softmax classification
→ accuracy, precision, recall, and F1
```

The generator contains three LSTM layers. The discriminator is a multilayer perceptron with two hidden layers. The detector uses Conv1D kernel sizes 3, 5, and 7, elementwise branch fusion, GRU temporal encoding, batch normalization, ReLU activation, and identity mappings.

## Requirements

Python 3.10 or newer is recommended. The published experiment used Python 3.6 and TensorFlow GPU 1.15.0; this repository uses the current TensorFlow/Keras API while retaining the reported model operations.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Project Structure

```text
configs/                 Dataset and experiment settings
data/README.md           Dataset acquisition and layout
scripts/prepare_data.py  Preprocessing and window construction
scripts/train.py         GAN resampling and detector training
scripts/evaluate.py      Test-set metrics and predictions
scripts/inference.py     Prediction on new traffic CSV files
scripts/run_experiments.py
src/gan_spatiotemporal_ids/
tests/test_pipeline.py   Pipeline and TensorFlow smoke tests
```

## Data Preparation

See `data/README.md` for the expected files. Prepare one dataset with:

```bash
python scripts/prepare_data.py --config configs/nsl_kdd.yaml
python scripts/prepare_data.py --config configs/unsw_nb15.yaml
python scripts/prepare_data.py --config configs/cicids2017.yaml
```

The preprocessing pipeline is fitted on training data and reused for the validation and test sets. A stratified validation partition is taken from the training data. When a dataset-specific test file is not configured, a deterministic stratified test split is used first.

## Training

```bash
python scripts/train.py --config configs/unsw_nb15.yaml
```

The command trains a separate GAN for each eligible minority class, augments the training windows to the configured target count, trains the detector, and retains the checkpoint with the lowest validation loss. Disable resampling without changing source code:

```bash
python scripts/train.py --config configs/unsw_nb15.yaml --set gan.enabled=false
```

## Evaluation

```bash
python scripts/evaluate.py --config configs/unsw_nb15.yaml \
  --checkpoint outputs/unsw_nb15/best.weights.h5
```

The command writes `metrics.json` and `predictions.npz` to the configured output directory. Metrics are weighted multiclass accuracy, precision, recall, and F1.

## Inference

The input CSV must contain the same feature columns used during preparation. A label column is not required.

```bash
python scripts/inference.py --config configs/unsw_nb15.yaml \
  --input path/to/traffic.csv \
  --output outputs/unsw_nb15/inference.csv
```

## Ablation and Learning-Rate Experiments

The ablation suite isolates the standard residual classifier, spatiotemporal features, the improved detector without GAN resampling, and the full pipeline. All variants inherit the same UNSW-NB15 data and training settings.

```bash
python scripts/run_experiments.py --suite ablation
python scripts/run_experiments.py --suite learning_rate --config configs/unsw_nb15.yaml
```

The learning-rate suite evaluates the reported initial rates of 0.01, 0.001, and 0.0001.

## Reproducibility

The default detector settings use a four-record window, an initial learning rate of 0.001, and 11 epochs. Random state is applied to Python, NumPy, TensorFlow, data splitting, and dataset shuffling. Generated metrics are always computed from model predictions; published table values are not stored as run outputs.

Several architectural dimensions and GAN training details are not specified by the article. `hidden_dim`, `num_blocks`, GAN noise size, GAN learning rate, GAN epochs, GAN batch size, detector batch size, and the random seed are therefore configurable engineering defaults. The CICIDS2017 split is also configurable because an exact split protocol is not stated. Private traffic from the three application scenarios is unavailable and is not represented by synthetic benchmark results. Third-party comparison methods are not reimplemented because their complete definitions are external to the article.

## Validation

```bash
python -m compileall src scripts tests
python -m unittest discover -s tests -v
```

The TensorFlow smoke test covers model construction, a forward pass, loss, backward propagation, an optimizer step, checkpoint save, and checkpoint reload. Tests that require TensorFlow are skipped when the framework is not installed.
