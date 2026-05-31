# DeepFake Image Detector

A proof-of-concept deep learning project that classifies face images as **Real** or **Fake** using a convolutional neural network (CNN). Includes a Streamlit web app for interactive demos.

> **Note:** Validation accuracy is ~68%. Results are indicative only—not suitable for high-stakes or forensic use.

## Features

- CNN trained on Real vs Fake image pairs (128×128 input)
- Image preprocessing: contrast enhancement and gamma correction
- **Streamlit app** — upload an image and view prediction with confidence
- **CLI** — run inference from the command line
- Python **3.13** virtual environment with TensorFlow 2.20+

## Requirements

- **Python 3.13** (TensorFlow does not support Python 3.14 yet)
- Windows with the `py` launcher (`py -3.13` should work)
- ~500 MB disk space for dependencies + model

## Project structure

```
DeepFake_Detection/
├── app.py              # Streamlit web UI
├── inference.py        # Model load, preprocess, predict
├── predict.py          # CLI prediction
├── preprocessing.py    # Contrast, gamma, edge utilities
├── train.ipynb         # Training notebook
├── model/
│   └── deepfake_model.h5
├── dataset/
│   ├── Real/           # 5,890 images
│   └── Fake/           # 7,000 images
├── requirements.txt
├── setup.ps1           # Create venv and install deps
└── run_app.ps1         # Start Streamlit
```

## Setup

From the project folder in PowerShell:

```powershell
cd path\to\DeepFake_Detection
.\setup.ps1
```

This creates `.venv` with Python 3.13 and installs all dependencies.

**Manual setup:**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> Use `.\.venv\Scripts\python.exe` for all commands below—not the system `python` if it points to 3.14.

## Run the demo (Streamlit)

```powershell
.\run_app.ps1
```

Or:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501), upload a JPG/PNG image, and view the Real/Fake prediction.

## Command-line prediction

```powershell
.\.venv\Scripts\python.exe predict.py dataset\Fake\001DDU0NI4.jpg
```

Example output:

```
Fake Image (72.3% confidence)
Fake probability: 72.3%
```

## Training

Open `train.ipynb` in Jupyter or VS Code with the **3.13 venv** interpreter selected.

Before retraining, set the dataset path in the notebook to your local folder, for example:

```python
dataset_path = "C:/Users/dell/Downloads/DeepFake_Detection/dataset"
```

Training pipeline (see notebook):

1. Sample up to 2,000 images per class (configurable)
2. Preprocess: contrast + gamma, resize to 128×128
3. Train/test split 80/20
4. CNN: 2 conv blocks + dense head, 5 epochs
5. Save model to `model/deepfake_model.h5`

## Model details

| Item | Value |
|------|--------|
| Input size | 128 × 128 × 3 |
| Output | Sigmoid (Fake if score > 0.5) |
| Optimizer | Adam |
| Loss | Binary crossentropy |
| Validation accuracy | ~68% |

## Improving accuracy

Ideas for future work (see sidebar in the Streamlit app):

- Transfer learning (EfficientNet, MobileNetV2)
- Use the full dataset (~12.9k images)
- Data augmentation and more epochs with early stopping
- Higher resolution (224×224) with a pretrained backbone


## License

Use for educational and demo purposes. Dataset and model usage may be subject to your data source terms.
