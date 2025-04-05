import os
from pathlib import Path

# Directory setup
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model_artifacts"

# Model parameters
CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "embedding_dim": 128,
    "lstm_units": 64,
    "vocab_size": 10000,
    "max_len": 100,
    "class_weights": {0: 3.0}  # Boost for 'Legitimate' class
}

# Ensure directories exist
MODEL_DIR.mkdir(exist_ok=True)