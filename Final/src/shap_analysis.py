import shap
import numpy as np
import matplotlib.pyplot as plt
import joblib
from typing import List, Tuple
from .config import MODEL_DIR, CONFIG

class SHAPExplainer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_len = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load required models and tokenizers."""
        self.model = joblib.load(MODEL_DIR / "lstm_model.keras")
        self.tokenizer = joblib.load(MODEL_DIR / "tokenizer.pkl")
        self.max_len = joblib.load(MODEL_DIR / "max_len.pkl")

    def explain(self, text: str) -> List[Tuple[str, float]]:
        """Generate SHAP explanations for a single text."""
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_len)
        
        explainer = shap.KernelExplainer(
            lambda x: self.model.predict(x),
            np.zeros((1, self.max_len))
        )
        shap_values = explainer.shap_values(padded)
        
        # Process and return top influential tokens
        tokens = [self.tokenizer.index_word.get(idx, 'UNK') for idx in sequence[0]]
        return list(zip(tokens, np.abs(shap_values).mean(axis=0)[0]))[:5]