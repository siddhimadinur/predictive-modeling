"""Preprocess MTSamples medical transcriptions for specialty classification.

Loads mtsamples.csv, uses transcription text and medical_specialty labels,
filters to specialties with at least 50 samples, cleans text, and produces
stratified HuggingFace Dataset splits. Falls back to synthetic data when
the CSV is not available.
"""

import json
import os
import random
import re
import sys
from typing import Optional

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# Label encoder
# ---------------------------------------------------------------------------

class LabelEncoder:
    """Bidirectional mapping between category names and integer labels.

    Supports saving/loading to a JSON file so the same encoding can be
    reused at inference time.
    """

    def __init__(self) -> None:
        self.label2id: dict[str, int] = {}
        self.id2label: dict[int, str] = {}

    def fit(self, labels: list[str]) -> "LabelEncoder":
        """Build the mapping from a list of unique label strings."""
        unique = sorted(set(labels))
        self.label2id = {label: idx for idx, label in enumerate(unique)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        return self

    def encode(self, label: str) -> int:
        """Convert a category name to its integer id."""
        return self.label2id[label]

    def decode(self, idx: int) -> str:
        """Convert an integer id back to its category name."""
        return self.id2label[idx]

    def save(self, path: str) -> None:
        """Persist the encoder to a JSON file."""
        with open(path, "w") as f:
            json.dump({"label2id": self.label2id}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LabelEncoder":
        """Load an encoder from a previously saved JSON file."""
        with open(path) as f:
            data = json.load(f)
        enc = cls()
        enc.label2id = data["label2id"]
        enc.id2label = {int(v): k for k, v in enc.label2id.items()}
        return enc

    def __len__(self) -> int:
        return len(self.label2id)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_BOILERPLATE_RE = re.compile(
    r"^(admission date|discharge date|date of birth|"
    r"service|attending|chief complaint|allergies|"
    r"discharge disposition|discharge diagnosis|"
    r"discharge condition|discharge instructions):.*$",
    re.IGNORECASE | re.MULTILINE,
)


def clean_note(text: str) -> str:
    """Lowercase, strip boilerplate headers, and collapse whitespace."""
    text = str(text).lower()
    text = _BOILERPLATE_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[: config.MAX_NOTE_CHARS]


# ---------------------------------------------------------------------------
# MTSamples loading
# ---------------------------------------------------------------------------

MIN_SAMPLES_PER_SPECIALTY: int = 50


def load_mtsamples(data_dir: str) -> pd.DataFrame:
    """Load mtsamples.csv and filter to specialties with enough samples.

    Args:
        data_dir: Directory containing ``raw/mtsamples.csv``.

    Returns:
        DataFrame with columns ``text`` (cleaned transcription) and
        ``specialty`` (medical specialty label).
    """
    csv_path = os.path.join(data_dir, "raw", "mtsamples.csv")
    df = pd.read_csv(csv_path)

    # Use transcription as text, medical_specialty as label
    df = df.dropna(subset=["transcription"])
    df = df[df["medical_specialty"].notna()].copy()
    df["medical_specialty"] = df["medical_specialty"].str.strip()

    # Filter to specialties with at least MIN_SAMPLES_PER_SPECIALTY samples
    counts = df["medical_specialty"].value_counts()
    keep = counts[counts >= MIN_SAMPLES_PER_SPECIALTY].index
    df = df[df["medical_specialty"].isin(keep)].copy()

    df["text"] = df["transcription"].apply(clean_note)
    df = df.rename(columns={"medical_specialty": "specialty"})

    print(f"Loaded {len(df)} samples across {df['specialty'].nunique()} specialties")
    print("Specialties:", ", ".join(sorted(df["specialty"].unique())))
    return df[["text", "specialty"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

_SYNTH_TEMPLATES: list[str] = [
    "patient is a {age} year old {sex} who presented with {complaint}. "
    "history of {history}. physical exam showed {finding}. "
    "labs were notable for {lab}. patient was treated with {treatment} "
    "and discharged in stable condition.",
    "this is a {age} yo {sex} admitted for {complaint}. "
    "past medical history significant for {history}. "
    "on examination {finding}. workup revealed {lab}. "
    "started on {treatment}. clinical course was uncomplicated.",
    "{age} year old {sex} presenting with acute {complaint}. "
    "pmh includes {history}. exam notable for {finding}. "
    "labs showed {lab}. managed with {treatment}. "
    "patient improved and was discharged home.",
]

_SYNTH_FIELDS: dict[str, list[str]] = {
    "age": [str(a) for a in range(20, 95)],
    "sex": ["male", "female"],
    "complaint": [
        "chest pain", "shortness of breath", "abdominal pain", "fever",
        "altered mental status", "weakness", "headache", "syncope",
        "nausea and vomiting", "back pain", "cough", "dizziness",
        "lower extremity edema", "fall", "seizure", "gi bleeding",
        "urinary tract infection", "pneumonia", "diabetic ketoacidosis",
        "hip fracture",
    ],
    "history": [
        "hypertension", "diabetes mellitus type 2", "coronary artery disease",
        "atrial fibrillation", "copd", "chf", "ckd stage 3", "depression",
        "obesity", "hypothyroidism", "asthma", "gerd",
    ],
    "finding": [
        "tachycardia", "bibasilar crackles", "distended abdomen",
        "decreased breath sounds", "lower extremity edema", "confusion",
        "tenderness to palpation", "regular rate and rhythm",
        "normal neurological exam", "mild jaundice",
    ],
    "lab": [
        "elevated troponin", "leukocytosis", "elevated creatinine",
        "hyponatremia", "elevated lactate", "anemia",
        "elevated liver enzymes", "hyperglycemia", "elevated bnp",
        "positive blood cultures",
    ],
    "treatment": [
        "iv antibiotics", "heparin drip", "insulin sliding scale",
        "diuresis with lasix", "pain management", "beta blockers",
        "surgical intervention", "blood transfusion", "iv fluids",
        "anticoagulation",
    ],
}

_SYNTH_SPECIALTIES: list[str] = [
    "Orthopedic", "Cardiovascular / Pulmonary", "Gastroenterology",
    "Neurology", "General Medicine", "Urology", "Radiology",
    "Obstetrics / Gynecology", "Surgery", "Ophthalmology",
]


def generate_synthetic_dataset(n: int = config.SYNTHETIC_DATASET_SIZE) -> pd.DataFrame:
    """Create *n* fake clinical notes with random specialty labels.

    Useful for testing the full pipeline without the MTSamples CSV.
    """
    random.seed(42)
    records: list[dict[str, str]] = []
    for _ in range(n):
        template = random.choice(_SYNTH_TEMPLATES)
        fields = {k: random.choice(v) for k, v in _SYNTH_FIELDS.items()}
        text = template.format(**fields)
        specialty = random.choice(_SYNTH_SPECIALTIES)
        records.append({"text": text, "specialty": specialty})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_datasets(
    data_dir: Optional[str] = None,
) -> tuple[DatasetDict, LabelEncoder]:
    """Build train/val/test splits and fit a label encoder.

    If mtsamples.csv exists in *data_dir*/raw/, it is used. Otherwise a
    synthetic dataset is generated so the pipeline can be tested end-to-end.

    Returns:
        A ``(DatasetDict, LabelEncoder)`` tuple. The DatasetDict contains
        ``train``, ``validation``, and ``test`` splits, each with columns
        ``text`` (str) and ``label`` (int).
    """
    data_dir = data_dir or config.DATA_DIR

    csv_path = os.path.join(data_dir, "raw", "mtsamples.csv")

    if os.path.isfile(csv_path):
        print("Loading MTSamples data from", csv_path)
        df = load_mtsamples(data_dir)
    else:
        print(
            "MTSamples CSV not found. Generating synthetic dataset "
            f"({config.SYNTHETIC_DATASET_SIZE} samples) for testing."
        )
        df = generate_synthetic_dataset()

    # Fit label encoder
    encoder = LabelEncoder().fit(df["specialty"].tolist())
    df["label"] = df["specialty"].apply(encoder.encode)

    # Stratified splits
    train_df, temp_df = train_test_split(
        df,
        test_size=config.VAL_SPLIT + config.TEST_SPLIT,
        stratify=df["label"],
        random_state=42,
    )
    relative_test = config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["label"],
        random_state=42,
    )

    def _to_dataset(subset: pd.DataFrame) -> Dataset:
        return Dataset.from_dict(
            {"text": subset["text"].tolist(), "label": subset["label"].tolist()}
        )

    ds = DatasetDict(
        {
            "train": _to_dataset(train_df),
            "validation": _to_dataset(val_df),
            "test": _to_dataset(test_df),
        }
    )

    # Persist label encoder
    os.makedirs(os.path.dirname(config.LABEL_ENCODER_PATH), exist_ok=True)
    encoder.save(config.LABEL_ENCODER_PATH)
    print(f"Label encoder saved to {config.LABEL_ENCODER_PATH}")
    print(f"Number of classes: {len(encoder)}")
    print(
        f"Splits — train: {len(ds['train'])}, "
        f"val: {len(ds['validation'])}, test: {len(ds['test'])}"
    )
    return ds, encoder


if __name__ == "__main__":
    prepare_datasets()
