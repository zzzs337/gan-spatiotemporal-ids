# Datasets

Raw traffic files are not distributed with this repository. Download each dataset from its official source and place the files as follows:

```text
data/raw/
├── nsl_kdd/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── unsw_nb15/
│   └── UNSW_NB15.csv
└── cicids2017/
    └── flows.csv
```

The expected NSL-KDD files are the standard 43-column train and test text files. The UNSW-NB15 configuration expects one combined CSV with `attack_cat` as the multiclass target and applies the reported 3:2 split. The four published UNSW-NB15 source CSV files may be concatenated in their original order. The CICIDS2017 CSV must contain a `Label` column; multiple source CSV files may be combined before preprocessing.

The preparation command fits missing-value handling, one-hot encoding, and Z-score scaling on the training partition only. It saves windowed arrays, the fitted preprocessing pipeline, and metadata under `data/processed/<dataset>/`.

```bash
python scripts/prepare_data.py --config configs/unsw_nb15.yaml
```
