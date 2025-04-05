import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from .config import CONFIG, MODEL_DIR
from .data_processing import load_and_preprocess_data

def build_model(num_classes: int) -> Sequential:
    """Construct LSTM model."""
    model = Sequential([
        Embedding(CONFIG["vocab_size"] + 1, CONFIG["embedding_dim"], input_length=CONFIG["max_len"]),
        SpatialDropout1D(0.3),
        LSTM(CONFIG["lstm_units"], dropout=0.3, recurrent_dropout=0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train():
    """Train and save model."""
    X, y = load_and_preprocess_data("PIA_Augmented_Dataset.csv")
    if X is None:
        raise ValueError("Data loading failed")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG["test_size"], 
        random_state=CONFIG["random_state"]
    )

    # Compute class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    for cls, boost in CONFIG["class_weights"].items():
        if cls in class_weights:
            class_weights[cls] *= boost

    # Train model
    model = build_model(len(classes))
    model.fit(
        X_train, y_train,
        epochs=3,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(patience=2)],
        class_weight=class_weights,
        verbose=1
    )

    # Evaluate
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=joblib.load(MODEL_DIR / "label_encoder.pkl").classes_))

    # Save model
    model.save(MODEL_DIR / "lstm_model.keras")
    return model