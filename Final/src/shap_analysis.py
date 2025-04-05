import shap
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List, Tuple, Optional
from pathlib import Path
from config import MODEL_DIR, CONFIG
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SHAPExplainer:
    """Class for generating SHAP explanations for LSTM model predictions."""
    
    def __init__(self):
        """Initialize the explainer and load required artifacts."""
        self.model = None
        self.tokenizer = None
        self.max_len = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Load required models and tokenizers with comprehensive validation.
        
        Raises:
            RuntimeError: If any artifact fails to load or is invalid
        """
        try:
            # Verify files exist first
            required_files = {
                "model": MODEL_DIR / "lstm_model.keras",
                "tokenizer": MODEL_DIR / "tokenizer.pkl", 
                "max_len": MODEL_DIR / "max_len.pkl"
            }
            
            logger.info("Verifying artifact files exist...")
            for name, path in required_files.items():
                if not path.exists():
                    raise FileNotFoundError(f"Missing {name} file at {path}")
                logger.info(f"Found {name} at {path}")

            # Load with validation
            logger.info("Loading model...")
            self.model = load_model(str(required_files["model"]))
            if not hasattr(self.model, 'predict'):
                raise ValueError("Loaded model missing predict method")

            # Load tokenizer and verify
            logger.info("Loading tokenizer...")
            self.tokenizer = joblib.load(required_files["tokenizer"])
            if not hasattr(self.tokenizer, 'word_index'):
                raise ValueError("Tokenizer missing word_index attribute")
            if not hasattr(self.tokenizer, 'texts_to_sequences'):
                raise ValueError("Tokenizer missing texts_to_sequences method")

            # Load max_len
            logger.info("Loading max_len...")
            self.max_len = joblib.load(required_files["max_len"])
            if not isinstance(self.max_len, int) or self.max_len <= 0:
                raise ValueError(f"max_len should be positive integer, got {self.max_len}")

            logger.info("All artifacts loaded successfully")

        except Exception as e:
            error_msg = (
                f"Failed to load artifacts: {str(e)}\n"
                f"Please ensure:\n"
                f"1. You've run model_training.py first\n"
                f"2. Files in {MODEL_DIR} aren't corrupted\n"
                f"3. You have required permissions"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def explain(self, text: str) -> List[Tuple[str, float]]:
        """
        Generate SHAP explanations for a single text input.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of (token, importance_score) tuples for top influential tokens
            
        Raises:
            ValueError: If input text is invalid
        """
        try:
            # Input validation
            if not text or not isinstance(text, str):
                raise ValueError("Input must be a non-empty string")
            
            logger.info(f"Processing text: {text[:50]}...")

            # Tokenize and pad
            sequence = self.tokenizer.texts_to_sequences([text])
            if not sequence or not sequence[0]:
                logger.warning("No tokens recognized in input text")
                return [("No valid tokens", 0.0)]
                
            padded = pad_sequences(sequence, maxlen=self.max_len)
            logger.debug(f"Padded sequence shape: {padded.shape}")

            # SHAP explanation
            background = np.zeros((1, self.max_len))
            explainer = shap.KernelExplainer(
                lambda x: self.model.predict(x, verbose=0),
                background
            )
            
            shap_values = explainer.shap_values(padded)
            logger.debug(f"SHAP values computed")

            # Process results
            tokens = [
                self.tokenizer.index_word.get(idx, 'UNK') 
                for idx in sequence[0] 
                if idx > 0  # Skip padding
            ]
            
            if isinstance(shap_values, list):
                # Multi-class classification
                shap_agg = np.mean(np.abs(shap_values), axis=0)[0]
            else:
                # Binary classification
                shap_agg = np.abs(shap_values)[0]
            
            # Get top 5 influential tokens
            token_importance = sorted(
                zip(tokens, shap_agg[:len(tokens)]),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            
            logger.info(f"Top tokens: {token_importance}")
            return token_importance

        except Exception as e:
            logger.error(f"Explanation failed: {str(e)}")
            raise RuntimeError(f"Could not generate explanation: {str(e)}")


# Example usage (for testing)
if __name__ == "__main__":
    try:
        explainer = SHAPExplainer()
        sample_text = "This is a sample prompt to test the explainer"
        print(f"Explanation for '{sample_text}':")
        print(explainer.explain(sample_text))
    except Exception as e:
        print(f"Error: {str(e)}")