from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from preprocessing import enhance_contrast, gamma_correction

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "deepfake_model.h5"
IMAGE_SIZE = (128, 128)


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    """Apply training-time preprocessing and return a normalized batch of shape (1, H, W, 3)."""
    if image_bgr is None:
        raise ValueError("Could not read image.")

    processed = enhance_contrast(image_bgr)
    processed = gamma_correction(processed)
    processed = cv2.resize(processed, IMAGE_SIZE)
    processed = processed.astype(np.float32) / 255.0
    return np.expand_dims(processed, axis=0)


def preprocess_for_display(image_bgr: np.ndarray) -> np.ndarray:
    """Return preprocessed BGR image for visualization."""
    processed = enhance_contrast(image_bgr)
    processed = gamma_correction(processed)
    return cv2.resize(processed, IMAGE_SIZE)


def load_detection_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return load_model(MODEL_PATH)


def predict_image(model, image_bgr: np.ndarray) -> dict:
    """Run inference and return label, confidence, and raw score."""
    batch = preprocess_image(image_bgr)
    score = float(model.predict(batch, verbose=0)[0][0])
    is_fake = score > 0.5
    confidence = score if is_fake else 1.0 - score

    return {
        "label": "Fake" if is_fake else "Real",
        "is_fake": is_fake,
        "confidence": confidence,
        "fake_probability": score,
    }
