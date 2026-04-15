"""Configuration constants for the clinical note classification pipeline."""

import os

# Model
MODEL_NAME: str = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LENGTH: int = 256
NUM_LABELS: int = 10  # dynamically overridden by the number of filtered specialties

# Training hyperparameters
BATCH_SIZE: int = 4
EPOCHS: int = 5
LEARNING_RATE: float = 2e-5
WARMUP_STEPS: int = 200
WEIGHT_DECAY: float = 0.01
MAX_GRAD_NORM: float = 1.0

# Data splits
TRAIN_SPLIT: float = 0.7
VAL_SPLIT: float = 0.15
TEST_SPLIT: float = 0.15

# Paths
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR: str = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR: str = os.path.join(BASE_DIR, "logs")
LABEL_ENCODER_PATH: str = os.path.join(DATA_DIR, "label_encoder.json")

# Preprocessing
MAX_NOTE_CHARS: int = 2000
SYNTHETIC_DATASET_SIZE: int = 1000

# Minimum samples per specialty to include in training
MIN_SAMPLES_PER_SPECIALTY: int = 50
