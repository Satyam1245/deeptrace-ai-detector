# DeepTrace AI Detector

DeepTrace AI Detector is a multimodal AI-powered deepfake detection platform developed for analyzing manipulated images, videos, and audio using deep learning and digital forensic analysis.

The system combines neural inference, face detection, GradCAM explainability, FFT spectrum analysis, Error Level Analysis (ELA), and confidence-based threat scoring to identify synthetic or altered media content.

---

# Features

## Multimodal Deepfake Detection
- Image deepfake analysis
- Video frame-level detection
- Audio manipulation analysis
- Real-time media inference

---

## AI & Forensic Analysis
- EfficientNet-based classification
- MTCNN face detection
- GradCAM X-Ray visualization
- FFT spectrum analysis
- Error Level Analysis (ELA)
- Threat confidence scoring

---

## Backend System
- FastAPI inference server
- REST API architecture
- File upload support
- JSON response handling
- Multi-format media processing

---

## Dashboard Interface
- Dark forensic dashboard UI
- X-Ray visualization mode
- Spectral analysis panel
- Threat-level indicators
- Real-time system logs

---

# Technology Stack

## Backend
- Python
- FastAPI
- Uvicorn

## Deep Learning
- PyTorch
- EfficientNet
- facenet-pytorch

## Computer Vision
- OpenCV
- Albumentations
- Pillow

## Data Processing
- NumPy
- Pandas
- SciPy
- Scikit-learn

## Visualization
- Matplotlib
- GradCAM

---

# Project Structure

```text
deeptrace-ai-detector/
│
├── app.py
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── deepfake_detector/
│   │
│   ├── __init__.py
│   │
│   ├── audio/
│   │   ├── audio_inference.py
│   │   ├── audio_processor.py
│   │   └── train_audio.py
│   │
│   ├── video/
│   │   ├── video_inference.py
│   │   └── video_processor.py
│   │
│   ├── fusion/
│   │   └── multimodal_fusion.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── audio_model.py
│   │   └── efficientnet.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── loader.py
│   │   └── transforms.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   │
│   └── config/
│       ├── __init__.py
│       └── config.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── samples/
│
├── scripts/
│   ├── augment_dataset.py
│   ├── extract_faces.py
│   ├── inference.py
│   ├── test.py
│   └── train.py
│
├── assets/
│
├── test_images/
│
├── outputs/
│   └── checkpoints/
│
└── test_output/
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Satyam1245/deeptrace-ai-detector.git

cd deeptrace-ai-detector
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Required Dependencies

```txt
torch
torchvision
opencv-python
facenet-pytorch
efficientnet-pytorch
numpy
pandas
scikit-learn
Pillow
albumentations
matplotlib
scipy
tqdm
PyYAML
fastapi
uvicorn
python-multipart
```

---

# Running the Application

Start the FastAPI server:

```bash
python app.py
```

Application runs on:

```text
http://127.0.0.1:8000
```

---

# API Endpoint

## POST `/predict`

Supported media:
- Images
- Videos
- Audio files

The API response includes:
- Prediction result
- Confidence score
- Threat level
- Forensic analysis data
- Visualization outputs

---

# Detection Workflow

## Image Pipeline
1. Upload image
2. Face detection using MTCNN
3. Image preprocessing
4. Neural inference
5. GradCAM heatmap generation
6. FFT spectrum analysis
7. ELA forensic analysis
8. Threat score calculation

---

## Video Pipeline
1. Upload video
2. Frame extraction
3. Face localization
4. Frame-level inference
5. Fake frame aggregation
6. Final classification

---

## Audio Pipeline
1. Upload audio
2. Feature extraction
3. Audio heuristic analysis
4. Manipulation scoring
5. Final prediction

---

# Training

Run model training:

```bash
python scripts/train.py
```

---

# Inference

Run standalone inference:

```bash
python scripts/inference.py
```

---

# Testing

Evaluate model performance:

```bash
python scripts/test.py
```

---

# Current Capabilities

| Module | Status |
|---|---|
| Image Detection | Available |
| Video Detection | Available |
| Audio Detection | Available |
| GradCAM Visualization | Available |
| FFT Analysis | Available |
| ELA Analysis | Available |
| Threat Scoring | Available |
| FastAPI Backend | Available |

---

# Future Improvements

- Transformer-based multimodal fusion
- Real-time webcam monitoring
- Advanced audio synthesis detection
- Cloud deployment support
- Mobile optimization
- Explainable AI dashboards

---

# Use Cases

- Media authenticity verification
- Social media content analysis
- Digital forensic investigation
- AI-generated media detection
- Research and experimentation
- Cybersecurity applications

---

# License

This project is licensed under the MIT License.

---

# Author

Developed by Satyam Chaudhary

GitHub:
https://github.com/Satyam1245
