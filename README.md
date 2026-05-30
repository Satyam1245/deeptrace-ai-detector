# DeepTrace AI Detector

AI-powered multimodal deepfake detection system using computer vision, deep learning, and digital forensic analysis.

DeepTrace AI Detector analyzes images, videos, and audio files to identify manipulated or AI-generated content through a combination of neural network inference, forensic feature extraction, and explainable AI visualization techniques.

---

## Features

### Image Deepfake Detection
- Face detection using MTCNN
- EfficientNet-based classification
- GradCAM X-Ray visualization
- FFT spectrum analysis
- Error Level Analysis (ELA)
- Confidence scoring

### Video Deepfake Detection
- Frame extraction and processing
- Frame-level deepfake classification
- Temporal consistency analysis
- Aggregated video prediction

### Audio Analysis
- Audio preprocessing pipeline
- Synthetic speech detection heuristics
- Audio manipulation scoring

### Forensic Dashboard
- Interactive web interface
- Real-time analysis
- Threat level visualization
- System activity logs
- Multi-view forensic inspection

---

## Technology Stack

### Backend
- Python
- FastAPI
- Uvicorn

### Deep Learning
- PyTorch
- EfficientNet
- facenet-pytorch

### Computer Vision
- OpenCV
- Albumentations
- Pillow

### Data Processing
- NumPy
- Pandas
- SciPy
- Scikit-learn

### Visualization
- Matplotlib
- GradCAM

---

## Project Structure

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
├── test_images/
│
└── test_output/
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Satyam1245/deeptrace-ai-detector.git

cd deeptrace-ai-detector
```

### Create Virtual Environment

#### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the FastAPI server:

```bash
python app.py
```

Application URL:

```text
http://127.0.0.1:8000
```

---

## Detection Workflow

### Image Pipeline

1. Upload image
2. Face detection using MTCNN
3. Image preprocessing
4. EfficientNet inference
5. GradCAM visualization
6. FFT spectrum analysis
7. Error Level Analysis (ELA)
8. Threat score generation

### Video Pipeline

1. Upload video
2. Frame extraction
3. Face localization
4. Frame classification
5. Prediction aggregation
6. Final classification

### Audio Pipeline

1. Upload audio
2. Feature extraction
3. Heuristic analysis
4. Manipulation scoring
5. Final prediction

---

## Dashboard Capabilities

The DeepTrace AI Detector dashboard provides:

- Image, video, and audio uploads
- X-Ray visualization mode
- Spectrum analysis view
- Artifact inspection tools
- Threat level assessment
- Confidence scoring
- Real-time forensic logs

---

## Testing Assets

Sample validation files are included in:

```text
test_output/
```

These files are provided to verify inference workflows and demonstrate system functionality.

---

## Model Checkpoints

Model checkpoint files (`.pth`, `.pt`) are excluded from version control due to file size limitations.

To train the model and generate checkpoints:

```bash
python scripts/train.py
```

Generated checkpoints will be stored locally and are not included in this repository.

---

## Current Capabilities

| Module | Status |
|----------|----------|
| Image Detection | Available |
| Video Detection | Available |
| Audio Detection | Available |
| GradCAM Visualization | Available |
| FFT Analysis | Available |
| ELA Analysis | Available |
| Threat Scoring | Available |
| FastAPI Backend | Available |

---

## Future Enhancements

- Transformer-based multimodal fusion
- Real-time webcam monitoring
- Cloud deployment support
- Mobile optimization
- Advanced explainable AI dashboards
- Distributed inference pipeline

---

## License

This project is licensed under the MIT License.

---

## Author

**Satyam Chaudhary**

GitHub: https://github.com/Satyam1245
