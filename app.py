from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from inference import load_detection_model, predict_image, preprocess_for_display

st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔍",
    layout="wide",
)

st.title("DeepFake Image Detector")
st.caption(
    "Upload a face image to classify it as Real or Fake using the trained CNN model."
)

MODEL_PATH = Path(__file__).resolve().parent / "model" / "deepfake_model.h5"


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_detection_model()


def read_uploaded_image(uploaded_file) -> np.ndarray | None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This app uses a convolutional neural network trained on Real vs Fake images.

        **Preprocessing:** contrast enhancement and gamma correction (same as training).

        **Output:** probability that the image is fake. Scores above 0.5 are labeled Fake.
        """
    )
    st.divider()
    st.subheader("Model info")
    st.write(f"Input size: 128 × 128")
    st.write(f"Architecture: CNN (2 conv blocks + dense head)")
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        st.write(f"Model file: `{MODEL_PATH.name}` ({size_mb:.1f} MB)")
    else:
        st.error("Model file not found. Train the model first using `train.ipynb`.")

    with st.expander("How to improve the model"):
        st.markdown(
            """
            - **Transfer learning:** fine-tune EfficientNet or MobileNetV2 instead of training from scratch
            - **Use the full dataset:** ~12.9k images available; training used only 4k
            - **Data augmentation:** flips, rotation, zoom, brightness during training
            - **Train longer:** more epochs with early stopping and learning-rate scheduling
            - **Higher resolution:** try 224×224 inputs with a pretrained backbone
            - **Regularization:** dropout, batch normalization, and class weights for imbalance
            - **Better evaluation:** ROC/AUC, confusion matrix, and threshold tuning
            """
        )


if not MODEL_PATH.exists():
    st.warning("Place a trained model at `model/deepfake_model.h5` or run `train.ipynb` first.")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Upload a face or portrait image for classification.",
)

if uploaded_file is not None:
    image_bgr = read_uploaded_image(uploaded_file)

    if image_bgr is None:
        st.error("Unable to decode the uploaded file. Try a different image format.")
        st.stop()

    model = get_model()
    result = predict_image(model, image_bgr)
    processed_bgr = preprocess_for_display(image_bgr)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Original")
        st.image(bgr_to_rgb(image_bgr), use_container_width=True)

    with col2:
        st.subheader("Preprocessed")
        st.image(bgr_to_rgb(processed_bgr), use_container_width=True)

    with col3:
        st.subheader("Prediction")
        label = result["label"]
        confidence_pct = result["confidence"] * 100
        fake_pct = result["fake_probability"] * 100

        if result["is_fake"]:
            st.error(f"**{label}** — {confidence_pct:.1f}% confidence")
        else:
            st.success(f"**{label}** — {confidence_pct:.1f}% confidence")

        st.progress(result["confidence"])
        st.metric("Fake probability", f"{fake_pct:.1f}%")
        st.metric("Real probability", f"{100 - fake_pct:.1f}%")

        st.info(
            "This model achieved ~68% validation accuracy during training. "
            "Treat results as indicative, not definitive."
        )
else:
    st.info("Upload an image to get started.")
