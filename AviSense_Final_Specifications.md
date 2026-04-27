
# AviSense Final Specifications Document

## Project Title

**AviSense: A Machine Learning-Based Bird Audio Classification System for Selected Philippine Bird Species**

---

## 1. Project Overview

AviSense is a machine learning project that classifies selected Philippine bird species using audio recordings of bird calls and songs. The system uses publicly available bird sound recordings, extracts acoustic features such as MFCCs and spectral descriptors, trains classical machine learning models, and evaluates their performance using standard classification metrics.

The project is designed for a machine learning course where classical machine learning methods are preferred. The main pipeline focuses on feature extraction and traditional classifiers such as Support Vector Machine, Random Forest, Logistic Regression, K-Nearest Neighbors, and Gradient Boosting.

---

## 2. Objectives

### Main Objective

Develop a reproducible machine learning pipeline that classifies selected Philippine bird species from audio recordings using extracted acoustic features.

### Specific Objectives

1. Collect and organize bird audio recordings from public bird sound repositories.
2. Preserve recording metadata such as species name, recordist, date, location, song type, and license.
3. Preprocess audio files into a consistent format suitable for machine learning.
4. Extract acoustic features from bird audio recordings.
5. Train and compare several classical machine learning models.
6. Evaluate model performance using accuracy, precision, recall, F1-score, macro F1-score, and confusion matrix.
7. Identify common misclassifications and possible causes.
8. Produce a reproducible repository with notebooks, scripts, outputs, and documentation.

---

## 3. Target Bird Species

| No. | Scientific Name | Common Name | Current Count |
|---:|---|---|---:|
| 1 | *Hypsipetes philippinus* | Philippine Bulbul | 89 |
| 2 | *Phapitreron leucotis* | White-eared Brown Dove | 77 |
| 3 | *Dicrurus balicassius* | Balicassiao | 68 |
| 4 | *Ninox philippensis* | Philippine Hawk-Owl | 61 |
| 5 | *Brachypteryx montana* | White-browed Shortwing | 59 |

Total current recordings:

```text
354 recordings
```

This is acceptable for a small classical machine learning class project, but it is still limited. Results should be framed as a baseline study, not a production-grade bird classifier.

---

## 4. Dataset Description

### Primary Source

The dataset consists of bird audio recordings from public bird sound repositories, especially Xeno-canto.

### Current Raw Data Layout

Your current `data/raw/` structure is:

```text
data/
└── raw/
    ├── Brachypteryx montana/
    ├── Dicrurus balicassius/
    ├── Hypsipetes philippinus/
    ├── Ninox philippensis/
    └── Phapitreron leucotis/
```

Each species folder contains:

```text
species folder/
├── A/
├── B/
├── C/
├── D/
├── E/
├── Species name.csv
├── XCxxxxx - recording name.mp3
├── XCxxxxx - recording name.mp3
└── ...
```

Example:

```text
data/raw/Brachypteryx montana/
├── A/
├── B/
├── C/
├── D/
├── E/
├── Brachypteryx montana.csv
├── XC67735 - White-browed Shortwing - Brachypteryx montana brunneiceps.mp3
├── XC67736 - White-browed Shortwing - Brachypteryx montana brunneiceps.mp3
└── ...
```

The folders `A`, `B`, `C`, `D`, and `E` may represent Xeno-canto quality ratings. The scripts should search recursively using `rglob()` so files are detected whether they are directly in the species folder or inside quality folders.

### Metadata Columns

Each species CSV may include:

```text
Common name
Scientific name
Subspecies
Recordist
Date
Time
Location
Country
Latitude
Longitude
Elevation
Songtype
Remarks
Back_latin
Catalogue number
License
```

Metadata must be preserved for attribution, filtering, and possible error analysis.

---

## 5. Recommended Final Project Structure

```text
avisense/
├── README.md
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── config.yaml
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── Brachypteryx montana/
│   │   ├── Dicrurus balicassius/
│   │   ├── Hypsipetes philippinus/
│   │   ├── Ninox philippensis/
│   │   └── Phapitreron leucotis/
│   │
│   ├── interim/
│   │   └── wav/
│   │       ├── Brachypteryx montana/
│   │       ├── Dicrurus balicassius/
│   │       ├── Hypsipetes philippinus/
│   │       ├── Ninox philippensis/
│   │       └── Phapitreron leucotis/
│   │
│   ├── processed/
│   │   ├── metadata/
│   │   │   ├── audio_inventory.csv
│   │   │   ├── all_metadata.csv
│   │   │   └── dataset_summary.csv
│   │   ├── features/
│   │   │   └── recording_features.csv
│   │   └── splits/
│   │
│   └── external/
│
├── notebooks/
│   ├── 01_dataset_inventory.ipynb
│   ├── 02_audio_preprocessing.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_exploratory_data_analysis.ipynb
│   ├── 05_model_training_baselines.ipynb
│   ├── 06_model_evaluation.ipynb
│   ├── 07_error_analysis.ipynb
│   └── 08_final_results_summary.ipynb
│
├── src/
│   └── avisense/
│       ├── __init__.py
│       ├── config.py
│       ├── utils.py
│       ├── data_inventory.py
│       ├── preprocessing.py
│       ├── features.py
│       ├── dataset.py
│       ├── train.py
│       ├── evaluate.py
│       ├── predict.py
│       └── plots.py
│
├── scripts/
│   ├── 01_make_inventory.py
│   ├── 02_convert_audio.py
│   ├── 03_extract_features.py
│   ├── 04_train_models.py
│   ├── 05_evaluate_models.py
│   └── 06_predict_file.py
│
├── models/
│   ├── best_model.joblib
│   ├── label_encoder.joblib
│   └── train_test_split.joblib
│
├── outputs/
│   ├── reports/
│   │   ├── model_comparison.csv
│   │   ├── classification_report.txt
│   │   └── final_metrics.csv
│   ├── figures/
│   │   ├── species_distribution.png
│   │   ├── model_comparison.png
│   │   └── confusion_matrix.png
│   └── predictions/
│       └── test_predictions.csv
│
├── paper/
│   ├── main.tex
│   ├── references.bib
│   └── sections/
│       ├── introduction.tex
│       ├── methodology.tex
│       ├── results.tex
│       └── conclusion.tex
│
└── docs/
    ├── AviSense_Final_Specifications.md
    ├── dataset_documentation.md
    └── user_manual.md
```

---

## 6. Environment Setup

### Recommended: Conda

```bash
conda create -n avisense python=3.11 -y
conda activate avisense
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name avisense --display-name "Python (AviSense)"
```

### Alternative: venv

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## 7. Required Files

### `requirements.txt`

```txt
numpy>=1.26.0
pandas>=2.1.0
scikit-learn>=1.3.0
librosa>=0.10.1
soundfile>=0.12.1
audioread>=3.0.0
matplotlib>=3.8.0
seaborn>=0.13.0
joblib>=1.3.0
tqdm>=4.66.0
PyYAML>=6.0.1
jupyter>=1.0.0
ipykernel>=6.25.0
```

Optional:

```txt
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
imbalanced-learn>=0.11.0
```

### `environment.yml`

```yaml
name: avisense
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - numpy
  - pandas
  - scikit-learn
  - librosa
  - soundfile
  - matplotlib
  - seaborn
  - joblib
  - tqdm
  - pyyaml
  - jupyter
  - ipykernel
  - pip:
      - audioread
```

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "avisense"
version = "0.1.0"
description = "Machine learning-based Philippine bird audio classification system"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]
```

### `config.yaml`

```yaml
project:
  name: AviSense
  random_state: 42

paths:
  raw_data_dir: data/raw
  interim_wav_dir: data/interim/wav
  processed_dir: data/processed
  metadata_dir: data/processed/metadata
  features_dir: data/processed/features
  splits_dir: data/processed/splits
  models_dir: models
  reports_dir: outputs/reports
  figures_dir: outputs/figures
  predictions_dir: outputs/predictions

audio:
  target_sample_rate: 22050
  mono: true
  duration_seconds: 10
  segment_audio: false
  segment_duration_seconds: 5
  segment_overlap_seconds: 0

features:
  n_mfcc: 20
  use_delta: true
  use_delta_delta: true
  use_spectral_features: true
  use_temporal_features: true

training:
  test_size: 0.2
  cv_folds: 5
  random_state: 42
  scoring: f1_macro

models:
  run_logistic_regression: true
  run_svm: true
  run_random_forest: true
  run_knn: true
  run_gradient_boosting: true
```

### `.gitignore`

```gitignore
__pycache__/
*.py[cod]
.venv/
venv/
env/
.ipynb_checkpoints/
.DS_Store
Thumbs.db

data/interim/
data/processed/
outputs/
models/

*.wav
*.mp3
*.flac
*.m4a
*.ogg

*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.log
*.out
*.synctex.gz
*.toc
*.lof
*.lot
```

If raw audio is too large for GitHub, do not push it. Instead, upload it to Google Drive, Kaggle, Zenodo, or another storage option and document the download instructions.

---

## 8. Complete Code

### `src/avisense/config.py`

```python
from pathlib import Path
import yaml


def load_config(config_path="config.yaml"):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_directories(config):
    path_keys = [
        "interim_wav_dir",
        "processed_dir",
        "metadata_dir",
        "features_dir",
        "splits_dir",
        "models_dir",
        "reports_dir",
        "figures_dir",
        "predictions_dir",
    ]

    for key in path_keys:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)
```

### `src/avisense/utils.py`

```python
from pathlib import Path
import re

AUDIO_EXTENSIONS = [".mp3", ".wav", ".flac", ".m4a", ".ogg"]


def safe_filename(name):
    name = name.strip()
    name = re.sub(r"[^\w\s.-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name


def find_audio_files(directory):
    directory = Path(directory)
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(directory.rglob(f"*{ext}"))
    return sorted(files)


def get_species_dirs(raw_data_dir):
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")
    return sorted([p for p in raw_data_dir.iterdir() if p.is_dir()])
```

### `src/avisense/data_inventory.py`

```python
from pathlib import Path
import pandas as pd
from avisense.utils import find_audio_files, get_species_dirs


def build_inventory(raw_data_dir):
    raw_data_dir = Path(raw_data_dir)
    rows = []

    for species_dir in get_species_dirs(raw_data_dir):
        species_name = species_dir.name
        audio_files = find_audio_files(species_dir)

        for audio_path in audio_files:
            relative_path = audio_path.relative_to(raw_data_dir)
            parts = relative_path.parts

            quality_rating = None
            if len(parts) >= 3 and parts[1] in ["A", "B", "C", "D", "E"]:
                quality_rating = parts[1]

            rows.append({
                "species": species_name,
                "filename": audio_path.name,
                "relative_path": str(relative_path),
                "full_path": str(audio_path),
                "quality_rating": quality_rating,
                "file_extension": audio_path.suffix.lower(),
            })

    return pd.DataFrame(rows)


def load_species_metadata(species_dir):
    species_dir = Path(species_dir)
    csv_files = list(species_dir.glob("*.csv"))

    if not csv_files:
        return None

    metadata_path = csv_files[0]
    df = pd.read_csv(metadata_path)
    df["metadata_file"] = metadata_path.name
    df["species_folder"] = species_dir.name
    return df


def combine_metadata(raw_data_dir):
    raw_data_dir = Path(raw_data_dir)
    frames = []

    for species_dir in get_species_dirs(raw_data_dir):
        df = load_species_metadata(species_dir)
        if df is not None:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
```

### `src/avisense/preprocessing.py`

```python
from pathlib import Path
import librosa
import soundfile as sf
from avisense.utils import safe_filename


def load_audio(audio_path, target_sr=22050):
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return y, sr


def normalize_audio(y):
    if len(y) == 0:
        return y
    max_val = max(abs(y))
    if max_val == 0:
        return y
    return y / max_val


def trim_silence(y, top_db=30):
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed


def fix_duration(y, sr, duration_seconds):
    target_length = int(sr * duration_seconds)
    return librosa.util.fix_length(y, size=target_length)


def convert_to_wav(input_path, output_path, target_sr=22050, duration_seconds=None, trim=True, normalize=True):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y, sr = load_audio(input_path, target_sr=target_sr)

    if trim:
        y = trim_silence(y)

    if normalize:
        y = normalize_audio(y)

    if duration_seconds is not None:
        y = fix_duration(y, sr, duration_seconds)

    sf.write(output_path, y, sr)
    return output_path


def make_wav_output_path(raw_audio_path, raw_data_dir, output_dir):
    raw_audio_path = Path(raw_audio_path)
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)

    relative_path = raw_audio_path.relative_to(raw_data_dir)
    species = relative_path.parts[0]
    filename = safe_filename(raw_audio_path.stem) + ".wav"

    return output_dir / species / filename
```

### `src/avisense/features.py`

```python
from pathlib import Path
import numpy as np
import pandas as pd
import librosa


def summarize_feature(values, prefix):
    values = np.asarray(values)

    if values.ndim == 1:
        return {
            f"{prefix}_mean": float(np.mean(values)),
            f"{prefix}_std": float(np.std(values)),
            f"{prefix}_min": float(np.min(values)),
            f"{prefix}_max": float(np.max(values)),
        }

    features = {}
    for i in range(values.shape[0]):
        features[f"{prefix}_{i + 1}_mean"] = float(np.mean(values[i]))
        features[f"{prefix}_{i + 1}_std"] = float(np.std(values[i]))
        features[f"{prefix}_{i + 1}_min"] = float(np.min(values[i]))
        features[f"{prefix}_{i + 1}_max"] = float(np.max(values[i]))

    return features


def extract_audio_features(audio_path, species=None, target_sr=22050, n_mfcc=20):
    audio_path = Path(audio_path)
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    if len(y) == 0:
        raise ValueError(f"Empty audio file: {audio_path}")

    row = {
        "filename": audio_path.name,
        "full_path": str(audio_path),
        "species": species,
        "duration": float(librosa.get_duration(y=y, sr=sr)),
        "sample_rate": sr,
    }

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    row.update(summarize_feature(mfcc, "mfcc"))
    row.update(summarize_feature(delta, "delta_mfcc"))
    row.update(summarize_feature(delta2, "delta2_mfcc"))

    row.update(summarize_feature(librosa.feature.spectral_centroid(y=y, sr=sr)[0], "spectral_centroid"))
    row.update(summarize_feature(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0], "spectral_bandwidth"))
    row.update(summarize_feature(librosa.feature.spectral_rolloff(y=y, sr=sr)[0], "spectral_rolloff"))
    row.update(summarize_feature(librosa.feature.spectral_contrast(y=y, sr=sr), "spectral_contrast"))
    row.update(summarize_feature(librosa.feature.zero_crossing_rate(y)[0], "zero_crossing_rate"))
    row.update(summarize_feature(librosa.feature.rms(y=y)[0], "rms"))

    return row


def extract_features_from_inventory(inventory_df, target_sr=22050, n_mfcc=20):
    rows = []

    for _, item in inventory_df.iterrows():
        try:
            features = extract_audio_features(
                audio_path=item["full_path"],
                species=item["species"],
                target_sr=target_sr,
                n_mfcc=n_mfcc,
            )
            features["relative_path"] = item.get("relative_path", None)
            features["quality_rating"] = item.get("quality_rating", None)
            rows.append(features)
        except Exception as error:
            print(f"Failed to process {item['full_path']}: {error}")

    return pd.DataFrame(rows)
```

### `src/avisense/dataset.py`

```python
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_features(features_path):
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")
    return pd.read_csv(features_path)


def prepare_xy(df, label_column="species"):
    ignore_columns = {
        label_column,
        "filename",
        "full_path",
        "relative_path",
        "quality_rating",
    }

    feature_columns = [col for col in df.columns if col not in ignore_columns]
    X = df[feature_columns].copy()
    X = X.select_dtypes(include=["number"])
    y = df[label_column].copy()
    return X, y


def encode_labels(y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


def make_train_test_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
```

### `src/avisense/train.py`

```python
from pathlib import Path
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


def get_models(random_state=42):
    return {
        "logistic_regression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }


def train_and_evaluate_models(X_train, y_train, cv_folds=5, random_state=42):
    models = get_models(random_state=random_state)

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    rows = []

    for model_name, model in models.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        rows.append({
            "model": model_name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "precision_macro_mean": scores["test_precision_macro"].mean(),
            "precision_macro_std": scores["test_precision_macro"].std(),
            "recall_macro_mean": scores["test_recall_macro"].mean(),
            "recall_macro_std": scores["test_recall_macro"].std(),
            "f1_macro_mean": scores["test_f1_macro"].mean(),
            "f1_macro_std": scores["test_f1_macro"].std(),
        })

    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)


def fit_best_model(model_name, X_train, y_train, random_state=42):
    models = get_models(random_state=random_state)

    if model_name not in models:
        raise ValueError(f"Unknown model name: {model_name}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", models[model_name]),
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(model, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
```

### `src/avisense/evaluate.py`

```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def save_classification_report(y_true, y_pred, target_names, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )

    output_path.write_text(report, encoding="utf-8")


def save_confusion_matrix(y_true, y_pred, target_names, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names,
    )

    display.plot(ax=ax, xticks_rotation=45, values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def save_predictions(filenames, y_true, y_pred, label_encoder, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "filename": filenames,
        "true_label": label_encoder.inverse_transform(y_true),
        "predicted_label": label_encoder.inverse_transform(y_pred),
        "correct": y_true == y_pred,
    })

    df.to_csv(output_path, index=False)
    return df
```

### `src/avisense/plots.py`

```python
from pathlib import Path
import matplotlib.pyplot as plt


def plot_species_distribution(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = df["species"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Species Distribution")
    ax.set_xlabel("Species")
    ax.set_ylabel("Number of Recordings")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_model_comparison(results_df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = results_df.sort_values("f1_macro_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["model"], df["f1_macro_mean"])
    ax.set_title("Model Comparison by Macro F1-score")
    ax.set_xlabel("Macro F1-score")
    ax.set_ylabel("Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
```

### `src/avisense/predict.py`

```python
from pathlib import Path
import joblib
import pandas as pd
from avisense.features import extract_audio_features


def predict_audio_file(audio_path, model_path, label_encoder_path, target_sr=22050, n_mfcc=20):
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)

    features = extract_audio_features(
        audio_path=audio_path,
        species=None,
        target_sr=target_sr,
        n_mfcc=n_mfcc,
    )

    df = pd.DataFrame([features])

    ignore_columns = {
        "species",
        "filename",
        "full_path",
        "relative_path",
        "quality_rating",
    }

    X = df[[col for col in df.columns if col not in ignore_columns]]
    X = X.select_dtypes(include=["number"])

    pred_encoded = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    result = {
        "audio_path": str(audio_path),
        "predicted_species": pred_label,
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        classes = label_encoder.inverse_transform(range(len(probabilities)))
        result["probabilities"] = {
            species: float(prob)
            for species, prob in zip(classes, probabilities)
        }

    return result
```

---

## 9. Scripts

### `scripts/01_make_inventory.py`

```python
from pathlib import Path
from avisense.config import load_config, ensure_directories
from avisense.data_inventory import build_inventory, combine_metadata


def main():
    config = load_config()
    ensure_directories(config)

    raw_data_dir = Path(config["paths"]["raw_data_dir"])
    metadata_dir = Path(config["paths"]["metadata_dir"])

    inventory_df = build_inventory(raw_data_dir)
    metadata_df = combine_metadata(raw_data_dir)

    inventory_path = metadata_dir / "audio_inventory.csv"
    metadata_path = metadata_dir / "all_metadata.csv"
    summary_path = metadata_dir / "dataset_summary.csv"

    inventory_df.to_csv(inventory_path, index=False)

    if not metadata_df.empty:
        metadata_df.to_csv(metadata_path, index=False)

    summary = (
        inventory_df.groupby("species")
        .size()
        .reset_index(name="recording_count")
        .sort_values("recording_count", ascending=False)
    )

    summary.to_csv(summary_path, index=False)

    print(f"Saved inventory to: {inventory_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved summary to: {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
```

### `scripts/02_convert_audio.py`

```python
from pathlib import Path
from tqdm import tqdm
from avisense.config import load_config, ensure_directories
from avisense.data_inventory import build_inventory
from avisense.preprocessing import convert_to_wav, make_wav_output_path


def main():
    config = load_config()
    ensure_directories(config)

    raw_data_dir = Path(config["paths"]["raw_data_dir"])
    wav_dir = Path(config["paths"]["interim_wav_dir"])

    target_sr = config["audio"]["target_sample_rate"]
    duration_seconds = config["audio"]["duration_seconds"]

    inventory_df = build_inventory(raw_data_dir)

    for _, row in tqdm(inventory_df.iterrows(), total=len(inventory_df)):
        input_path = Path(row["full_path"])
        output_path = make_wav_output_path(input_path, raw_data_dir, wav_dir)

        convert_to_wav(
            input_path=input_path,
            output_path=output_path,
            target_sr=target_sr,
            duration_seconds=duration_seconds,
            trim=True,
            normalize=True,
        )

    print(f"Converted audio files saved to: {wav_dir}")


if __name__ == "__main__":
    main()
```

### `scripts/03_extract_features.py`

```python
from pathlib import Path
from avisense.config import load_config, ensure_directories
from avisense.data_inventory import build_inventory
from avisense.features import extract_features_from_inventory


def main():
    config = load_config()
    ensure_directories(config)

    audio_dir = Path(config["paths"]["interim_wav_dir"])
    features_dir = Path(config["paths"]["features_dir"])

    target_sr = config["audio"]["target_sample_rate"]
    n_mfcc = config["features"]["n_mfcc"]

    inventory_df = build_inventory(audio_dir)

    features_df = extract_features_from_inventory(
        inventory_df=inventory_df,
        target_sr=target_sr,
        n_mfcc=n_mfcc,
    )

    output_path = features_dir / "recording_features.csv"
    features_df.to_csv(output_path, index=False)

    print(f"Saved features to: {output_path}")
    print(features_df.shape)


if __name__ == "__main__":
    main()
```

### `scripts/04_train_models.py`

```python
from pathlib import Path
import joblib
from avisense.config import load_config, ensure_directories
from avisense.dataset import load_features, prepare_xy, encode_labels, make_train_test_split
from avisense.train import train_and_evaluate_models, fit_best_model, save_model
from avisense.plots import plot_model_comparison


def main():
    config = load_config()
    ensure_directories(config)

    features_path = Path(config["paths"]["features_dir"]) / "recording_features.csv"
    reports_dir = Path(config["paths"]["reports_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    models_dir = Path(config["paths"]["models_dir"])

    df = load_features(features_path)

    X, y = prepare_xy(df, label_column="species")
    y_encoded, label_encoder = encode_labels(y)

    X_train, X_test, y_train, y_test = make_train_test_split(
        X,
        y_encoded,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
    )

    results_df = train_and_evaluate_models(
        X_train=X_train,
        y_train=y_train,
        cv_folds=config["training"]["cv_folds"],
        random_state=config["training"]["random_state"],
    )

    results_df.to_csv(reports_dir / "model_comparison.csv", index=False)

    plot_model_comparison(
        results_df=results_df,
        output_path=figures_dir / "model_comparison.png",
    )

    best_model_name = results_df.iloc[0]["model"]
    best_model = fit_best_model(
        model_name=best_model_name,
        X_train=X_train,
        y_train=y_train,
        random_state=config["training"]["random_state"],
    )

    save_model(best_model, models_dir / "best_model.joblib")
    joblib.dump(label_encoder, models_dir / "label_encoder.joblib")
    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "test_indices": X_test.index.tolist(),
        },
        models_dir / "train_test_split.joblib",
    )

    print(results_df)
    print(f"Best model: {best_model_name}")


if __name__ == "__main__":
    main()
```

### `scripts/05_evaluate_models.py`

```python
from pathlib import Path
import joblib
import pandas as pd
from avisense.config import load_config, ensure_directories
from avisense.dataset import load_features
from avisense.evaluate import save_classification_report, save_confusion_matrix, save_predictions


def main():
    config = load_config()
    ensure_directories(config)

    features_path = Path(config["paths"]["features_dir"]) / "recording_features.csv"
    models_dir = Path(config["paths"]["models_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    predictions_dir = Path(config["paths"]["predictions_dir"])

    model = joblib.load(models_dir / "best_model.joblib")
    label_encoder = joblib.load(models_dir / "label_encoder.joblib")
    split = joblib.load(models_dir / "train_test_split.joblib")
    df = load_features(features_path)

    X_test = split["X_test"]
    y_test = split["y_test"]
    test_indices = split["test_indices"]

    y_pred = model.predict(X_test)

    save_classification_report(
        y_true=y_test,
        y_pred=y_pred,
        target_names=label_encoder.classes_,
        output_path=reports_dir / "classification_report.txt",
    )

    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        target_names=label_encoder.classes_,
        output_path=figures_dir / "confusion_matrix.png",
    )

    test_filenames = df.loc[test_indices, "filename"].tolist()

    save_predictions(
        filenames=test_filenames,
        y_true=y_test,
        y_pred=y_pred,
        label_encoder=label_encoder,
        output_path=predictions_dir / "test_predictions.csv",
    )

    metrics = {
        "total_test_samples": len(y_test),
        "correct_predictions": int((y_test == y_pred).sum()),
        "incorrect_predictions": int((y_test != y_pred).sum()),
    }

    pd.DataFrame([metrics]).to_csv(reports_dir / "final_metrics.csv", index=False)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
```

### `scripts/06_predict_file.py`

```python
import argparse
from pathlib import Path
from avisense.config import load_config
from avisense.predict import predict_audio_file


def main():
    parser = argparse.ArgumentParser(description="Predict bird species from an audio file.")
    parser.add_argument("--audio", required=True, help="Path to audio file.")
    args = parser.parse_args()

    config = load_config()
    models_dir = Path(config["paths"]["models_dir"])

    result = predict_audio_file(
        audio_path=args.audio,
        model_path=models_dir / "best_model.joblib",
        label_encoder_path=models_dir / "label_encoder.joblib",
        target_sr=config["audio"]["target_sample_rate"],
        n_mfcc=config["features"]["n_mfcc"],
    )

    print(result)


if __name__ == "__main__":
    main()
```

---

## 10. Notebook Plan

### `01_dataset_inventory.ipynb`

Purpose:

- Inspect raw dataset.
- Count recordings per species.
- Check metadata CSV files.
- Generate `audio_inventory.csv`.
- Generate `dataset_summary.csv`.

Expected outputs:

```text
data/processed/metadata/audio_inventory.csv
data/processed/metadata/all_metadata.csv
data/processed/metadata/dataset_summary.csv
outputs/figures/species_distribution.png
```

### `02_audio_preprocessing.ipynb`

Purpose:

- Load and listen to sample recordings.
- Plot waveforms.
- Convert `.mp3` files to standardized `.wav`.

Expected output:

```text
data/interim/wav/
```

### `03_feature_extraction.ipynb`

Purpose:

- Extract MFCC, delta MFCC, spectral, temporal, and energy features.
- Save machine-learning-ready table.

Expected output:

```text
data/processed/features/recording_features.csv
```

### `04_exploratory_data_analysis.ipynb`

Purpose:

- Analyze feature distributions.
- Plot species distribution.
- Optional PCA visualization.

Expected outputs:

```text
outputs/figures/species_distribution.png
outputs/figures/pca_features.png
outputs/figures/feature_distributions.png
```

### `05_model_training_baselines.ipynb`

Purpose:

- Train and compare baseline models.
- Select best model using macro F1-score.

Expected outputs:

```text
outputs/reports/model_comparison.csv
models/best_model.joblib
models/label_encoder.joblib
```

### `06_model_evaluation.ipynb`

Purpose:

- Evaluate the best model on the held-out test set.
- Generate classification report and confusion matrix.

Expected outputs:

```text
outputs/reports/classification_report.txt
outputs/figures/confusion_matrix.png
outputs/predictions/test_predictions.csv
```

### `07_error_analysis.ipynb`

Purpose:

- Inspect wrong predictions.
- Identify which species are commonly confused.
- Check if low-quality recordings are harder to classify.

Expected output:

```text
outputs/reports/error_analysis.csv
```

### `08_final_results_summary.ipynb`

Purpose:

- Prepare final tables and figures for the paper and presentation.

Expected outputs:

```text
outputs/reports/final_metrics.csv
outputs/figures/final_model_comparison.png
outputs/figures/final_confusion_matrix.png
```

---

## 11. Step-by-Step Manual

### Step 1: Open Project Folder

```bash
cd avisense
```

### Step 2: Create and Activate Environment

```bash
conda create -n avisense python=3.11 -y
conda activate avisense
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### Step 4: Register Jupyter Kernel

```bash
python -m ipykernel install --user --name avisense --display-name "Python (AviSense)"
```

### Step 5: Confirm Raw Dataset

```text
data/raw/
├── Brachypteryx montana/
├── Dicrurus balicassius/
├── Hypsipetes philippinus/
├── Ninox philippensis/
└── Phapitreron leucotis/
```

### Step 6: Make Inventory

```bash
python scripts/01_make_inventory.py
```

### Step 7: Convert Audio

```bash
python scripts/02_convert_audio.py
```

### Step 8: Extract Features

```bash
python scripts/03_extract_features.py
```

### Step 9: Train Models

```bash
python scripts/04_train_models.py
```

### Step 10: Evaluate Best Model

```bash
python scripts/05_evaluate_models.py
```

### Step 11: Predict One File

```bash
python scripts/06_predict_file.py --audio "data/raw/Brachypteryx montana/XC67735 - White-browed Shortwing - Brachypteryx montana brunneiceps.mp3"
```

---

## 12. Command Summary

```bash
conda create -n avisense python=3.11 -y
conda activate avisense
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name avisense --display-name "Python (AviSense)"

python scripts/01_make_inventory.py
python scripts/02_convert_audio.py
python scripts/03_extract_features.py
python scripts/04_train_models.py
python scripts/05_evaluate_models.py
```

Optional:

```bash
python scripts/06_predict_file.py --audio "path/to/audio.mp3"
```

---

## 13. Data Processing Details

### Inventory

The inventory step recursively scans each species folder using `rglob()`. This matters because audio files may be directly inside a species folder or inside `A/B/C/D/E` folders.

### Audio Preprocessing

Recommended settings:

```text
Target sample rate: 22,050 Hz
Channel: mono
Duration: 10 seconds
Silence trimming: enabled
Amplitude normalization: enabled
Output format: WAV
```

### Feature Extraction

Features extracted:

```text
MFCC
Delta MFCC
Delta-delta MFCC
Spectral centroid
Spectral bandwidth
Spectral rolloff
Spectral contrast
Zero-crossing rate
RMS energy
Duration
```

For time-varying features, the system computes:

```text
mean
standard deviation
minimum
maximum
```

---

## 14. Modeling Details

### Models

```text
Logistic Regression
Support Vector Machine with RBF kernel
Random Forest
K-Nearest Neighbors
Gradient Boosting
```

### Split

```text
80% training
20% testing
stratified by species
```

### Cross-Validation

```text
Stratified 5-fold cross-validation
```

### Main Metric

```text
Macro F1-score
```

Macro F1-score is the main metric because the dataset is relatively small and slightly imbalanced. It treats all bird species equally.

### Additional Metrics

```text
Accuracy
Macro precision
Macro recall
Per-class precision
Per-class recall
Per-class F1-score
Confusion matrix
```

---

## 15. Expected Outputs

After the full workflow, the project should produce:

```text
data/processed/metadata/audio_inventory.csv
data/processed/metadata/all_metadata.csv
data/processed/metadata/dataset_summary.csv
data/processed/features/recording_features.csv

models/best_model.joblib
models/label_encoder.joblib
models/train_test_split.joblib

outputs/reports/model_comparison.csv
outputs/reports/classification_report.txt
outputs/reports/final_metrics.csv

outputs/figures/model_comparison.png
outputs/figures/confusion_matrix.png

outputs/predictions/test_predictions.csv
```

---

## 16. Important Warnings

### Data Leakage

If you segment long recordings into clips, do not randomly split clips from the same original recording across train and test sets.

Bad:

```text
Recording A clip 1 -> train
Recording A clip 2 -> test
```

Better:

```text
All clips from Recording A -> train only
All clips from Recording B -> test only
```

### Dataset Size

The dataset has 354 recordings. This is workable for a course project, but not enough to claim real-world deployment performance.

Use wording such as:

```text
The results demonstrate the feasibility of classifying selected Philippine bird species under the collected dataset conditions.
```

Avoid:

```text
The system can identify all Philippine birds in real-world forest environments.
```

### Audio Quality

Public recordings may vary in:

```text
noise level
recording device
distance from bird
background species
weather noise
human noise
recording duration
compression quality
```

### Licenses

Keep metadata fields such as:

```text
Recordist
Catalogue number
License
Xeno-canto link
```

These are needed for proper attribution.

---

## 17. Suggested Paper Methodology

The study followed a machine learning pipeline consisting of data collection, audio preprocessing, feature extraction, model training, and model evaluation. Audio recordings of five Philippine bird species were collected from publicly available bird sound repositories. The selected species were *Hypsipetes philippinus*, *Phapitreron leucotis*, *Dicrurus balicassius*, *Ninox philippensis*, and *Brachypteryx montana*. Each species folder contained audio recordings and a metadata file with information such as scientific name, subspecies, recordist, location, date, song type, and license.

The raw audio files were standardized before feature extraction. Recordings were converted to mono-channel WAV format, resampled to 22,050 Hz, trimmed for silence, normalized by amplitude, and adjusted to a fixed duration. This preprocessing step reduced inconsistencies in audio format and ensured that feature extraction was performed under uniform conditions.

Acoustic features were extracted using Python-based audio processing tools. The extracted features included mel-frequency cepstral coefficients, delta MFCCs, delta-delta MFCCs, spectral centroid, spectral bandwidth, spectral rolloff, spectral contrast, zero-crossing rate, root mean square energy, and duration. For time-varying features, summary statistics such as mean, standard deviation, minimum, and maximum were computed. The final feature table was used as input for classical machine learning classifiers.

Several baseline classifiers were trained and compared, including Logistic Regression, Support Vector Machine, Random Forest, K-Nearest Neighbors, and Gradient Boosting. The dataset was divided into training and testing sets using stratified sampling to preserve class distribution. Stratified k-fold cross-validation was used during model comparison, and the best-performing model was selected based primarily on macro F1-score.

Model performance was evaluated using accuracy, macro precision, macro recall, macro F1-score, per-class classification reports, and a confusion matrix. Macro F1-score was emphasized because the dataset contained unequal numbers of recordings per species and because each species was considered equally important.

---

## 18. Suggested README

```md
# AviSense

AviSense is a machine learning-based bird audio classification project for selected Philippine bird species. It uses publicly available bird sound recordings, extracts acoustic features such as MFCCs and spectral descriptors, and trains classical machine learning models to classify species.

## Target Species

- Brachypteryx montana
- Dicrurus balicassius
- Hypsipetes philippinus
- Ninox philippensis
- Phapitreron leucotis

## Setup

```bash
conda create -n avisense python=3.11 -y
conda activate avisense
pip install -r requirements.txt
pip install -e .
```

## Run Pipeline

```bash
python scripts/01_make_inventory.py
python scripts/02_convert_audio.py
python scripts/03_extract_features.py
python scripts/04_train_models.py
python scripts/05_evaluate_models.py
```

## Predict One File

```bash
python scripts/06_predict_file.py --audio "path/to/audio.mp3"
```

## Main Metric

The main evaluation metric is macro F1-score.

## Dataset Attribution

Recordings were obtained from public bird sound repositories. Original recordists and licenses should be credited according to the metadata files.
```

---

## 19. Troubleshooting

### `ModuleNotFoundError: No module named 'avisense'`

Run:

```bash
pip install -e .
```

Or temporarily set:

Windows PowerShell:

```bash
$env:PYTHONPATH="src"
```

macOS/Linux:

```bash
export PYTHONPATH=src
```

### Audio file cannot be loaded

Install audio dependencies:

```bash
pip install audioread soundfile
```

With Conda:

```bash
conda install -c conda-forge ffmpeg -y
```

### GitHub does not upload empty folders

Git does not track empty folders. Add `.gitkeep` files:

```text
data/raw/.gitkeep
data/processed/.gitkeep
outputs/.gitkeep
```

### GitHub rejects large files

Do not upload full raw audio if it is too large. Use Google Drive, Kaggle, Zenodo, or GitHub Releases and document the link.

### Low model performance

Possible reasons:

```text
small dataset
noisy recordings
background species
similar bird calls
short recordings
poor train-test split
```

Possible fixes:

```text
add more recordings
filter low-quality recordings
separate song and call recordings
try longer duration
tune hyperparameters
add more features
```

### Overly high model performance

Possible cause:

```text
data leakage
duplicate recordings
same original recording split into both train and test
model learning location or recordist instead of species
```

---

## 20. Final Deliverables Checklist

### Code

```text
requirements.txt
environment.yml
config.yaml
pyproject.toml
src/avisense/*.py
scripts/*.py
notebooks/*.ipynb
```

### Data

```text
data/raw/
data/processed/metadata/audio_inventory.csv
data/processed/metadata/all_metadata.csv
data/processed/features/recording_features.csv
```

### Models

```text
models/best_model.joblib
models/label_encoder.joblib
models/train_test_split.joblib
```

### Reports

```text
outputs/reports/model_comparison.csv
outputs/reports/classification_report.txt
outputs/reports/final_metrics.csv
outputs/predictions/test_predictions.csv
```

### Figures

```text
outputs/figures/species_distribution.png
outputs/figures/model_comparison.png
outputs/figures/confusion_matrix.png
```

### Paper

```text
paper/main.tex
paper/references.bib
paper/sections/introduction.tex
paper/sections/methodology.tex
paper/sections/results.tex
paper/sections/conclusion.tex
```

---

## 21. Recommended Group Task Division

### Member 1: Dataset and Metadata

```text
organize data/raw
check metadata CSV files
verify species labels
prepare dataset summary
handle attribution and license information
```

### Member 2: Audio Preprocessing

```text
convert audio to WAV
trim silence
normalize audio
check sample rate and duration
prepare preprocessing explanation
```

### Member 3: Feature Extraction

```text
extract MFCC features
extract spectral features
extract temporal features
save feature table
explain features in methodology
```

### Member 4: Model Training

```text
train baseline models
run cross-validation
compare model performance
save best model
prepare model comparison table
```

### Member 5: Evaluation and Paper

```text
generate classification report
generate confusion matrix
perform error analysis
write results and discussion
prepare presentation slides
```

---

## 22. Final Positioning of the Project

AviSense should be presented as a focused and reproducible baseline system for classifying selected Philippine bird species using acoustic features and classical machine learning models.

The strongest framing is:

```text
AviSense demonstrates the feasibility of using classical machine learning and acoustic feature extraction for selected Philippine bird audio classification under a limited but organized dataset.
```

The project should not claim that it is already a complete field-ready bird identification application.

---

## 23. One-Sentence Summary

**AviSense is a reproducible classical machine learning pipeline that classifies selected Philippine bird species from audio recordings using MFCC, spectral, and temporal acoustic features.**
