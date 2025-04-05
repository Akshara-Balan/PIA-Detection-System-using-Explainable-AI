import pandas as pd
import numpy as np
import re
import joblib
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Tuple, Optional
from .config import DATA_DIR, MODEL_DIR, CONFIG

def clean_text(text: str) -> str:
    """Clean and preprocess text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def load_and_preprocess_data(file_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load and preprocess dataset."""
    try:
        df = pd.read_csv(DATA_DIR / file_name)
        df['Prompt'] = df['Prompt'].apply(clean_text)

        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['Category'])

        # Tokenize text
        tokenizer = Tokenizer(num_words=CONFIG["vocab_size"])
        tokenizer.fit_on_texts(df['Prompt'])
        X = pad_sequences(
            tokenizer.texts_to_sequences(df['Prompt']),
            maxlen=CONFIG["max_len"]
        )

        # Save artifacts
        joblib.dump(tokenizer, MODEL_DIR / "tokenizer.pkl")
        joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")
        joblib.dump(CONFIG["max_len"], MODEL_DIR / "max_len.pkl")

        return X, y
    except Exception as e:
        print(f"Data processing error: {e}")
        return None, None