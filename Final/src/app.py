import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from .data_processing import clean_text, load_and_preprocess_data
from .shap_analysis import SHAPExplainer
from .config import MODEL_DIR

def main():
    st.set_page_config(page_title="Prompt Injection Detector", layout="wide")
    st.title("🛡️ Prompt Injection Detection")

    # Load artifacts
    @st.cache_resource
    def load_artifacts():
        return {
            'model': load_model(MODEL_DIR / "lstm_model.keras"),
            'tokenizer': joblib.load(MODEL_DIR / "tokenizer.pkl"),
            'label_encoder': joblib.load(MODEL_DIR / "label_encoder.pkl"),
            'max_len': joblib.load(MODEL_DIR / "max_len.pkl")
        }

    artifacts = load_artifacts()
    explainer = SHAPExplainer()

    # UI Components
    tab1, tab2 = st.tabs(["Detector", "Info"])
    
    with tab1:
        prompt = st.text_area("Enter prompt:", height=200)
        if st.button("Analyze"):
            if prompt:
                with st.spinner("Processing..."):
                    try:
                        # Prediction
                        seq = artifacts['tokenizer'].texts_to_sequences([clean_text(prompt)])
                        padded = pad_sequences(seq, maxlen=artifacts['max_len'])
                        pred = artifacts['model'].predict(padded)[0]
                        pred_class = artifacts['label_encoder'].classes_[np.argmax(pred)]
                        
                        # Display results
                        st.success(f"Prediction: {pred_class}")
                        st.progress(max(pred))
                        
                        # SHAP Explanation
                        st.subheader("Explanation")
                        for token, score in explainer.explain(prompt):
                            st.write(f"- {token}: {score:.4f}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with tab2:
        st.write("## About")
        st.write("This tool detects prompt injection attacks using LSTM.")

if __name__ == "__main__":
    main()