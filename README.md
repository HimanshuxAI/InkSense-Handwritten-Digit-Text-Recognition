# InkSense — Handwritten Digit & Text Recognition

A unified platform that integrates **handwritten digit recognition** and **handwritten text recognition** using deep learning.

## 🧠 Integrated Projects

| Feature | Source | Model |
|---------|--------|-------|
| Digit Recognition | [aakashjhawar/handwritten-digit-recognition](https://github.com/aakashjhawar/handwritten-digit-recognition) | CNN on MNIST (99%+ accuracy) |
| Text Recognition | [githubharald/SimpleHTR](https://github.com/githubharald/SimpleHTR) | CNN + RNN (LSTM) + CTC on IAM dataset |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the Server

```bash
cd backend
python app.py
```

The app will start at **http://localhost:5000**

> **Note:** On first run, the digit recognition model will automatically train on MNIST (~2 minutes).

### 3. (Optional) Setup Text Recognition Model

For full HTR functionality, download the pre-trained model:

1. Download the [word model](https://www.dropbox.com/s/mya8hw6jyzqm0a3/word-model.zip?dl=1) from SimpleHTR
2. Extract contents into `backend/models/htr_model/`
3. Ensure `charList.txt` and `snapshot-*` files are in that directory

Without this, text recognition will show a placeholder message. Digit recognition works out of the box.

## 📁 Project Structure

```
H_D_T/
├── backend/
│   ├── app.py                      # Flask API server
│   ├── requirements.txt            # Python dependencies
│   ├── digit_recognition/
│   │   ├── __init__.py
│   │   └── model.py                # CNN digit model (MNIST)
│   ├── text_recognition/
│   │   ├── __init__.py
│   │   ├── model.py                # HTR model (SimpleHTR)
│   │   ├── preprocessor.py         # Image preprocessing
│   │   └── dataloader_iam.py       # IAM dataset loader
│   └── models/                     # Pre-trained model files
├── frontend/
│   ├── index.html                  # Main UI
│   ├── style.css                   # Styling
│   └── script.js                   # Frontend logic
└── README.md
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict-digit` | POST | Predict digit from uploaded image |
| `/api/predict-text` | POST | Recognize text from uploaded image |
| `/api/predict-canvas` | POST | Predict digit from canvas drawing |
| `/api/health` | GET | Health check |

## 🛠 Tech Stack

- **Backend:** Python, Flask, TensorFlow/Keras, OpenCV
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
- **Models:** CNN (digits), CNN+RNN+CTC (text)
- **Dataset:** MNIST (digits), IAM (text)
