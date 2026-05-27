import torch
import cv2
import numpy as np
from PIL import Image
import io
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import sys
import tempfile
from pathlib import Path
from scipy.special import softmax
from deepfake_detector.audio.audio_processor import extract_audio_heuristics

# Local module imports
sys.path.insert(0, str(Path(__file__).parent))

from deepfake_detector.models import DeepFakeDetector
from deepfake_detector.data import get_val_transforms, get_test_time_augmentation_transforms

# MTCNN for face detection
from facenet_pytorch import MTCNN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DeepTrace AI Detector API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import base64
from io import BytesIO

# Global runtime objects
model = None
device = None
transform = None
tta_transforms = None
grad_cam = None
mtcnn = None  # Face Detector

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
AUDIO_MODEL_PATH = Path("outputs/checkpoints/audio_model.pth")
IMAGE_REAL_THRESHOLD = 0.52
IMAGE_FAKE_THRESHOLD = 0.52
VIDEO_REAL_THRESHOLD = 0.52
VIDEO_FAKE_THRESHOLD = 0.52
UNCERTAIN_MARGIN = 0.03
VIDEO_FRAME_FAKE_THRESHOLD = 0.60
VIDEO_MIN_FAKE_FRAME_RATIO = 0.30
MULTI_FACE_BORDERLINE_COUNT = 4
MULTI_FACE_FAKE_FLOOR = 0.47
MULTI_FACE_MARGIN = 0.08

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activation = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x):
        self.gradients = None
        self.activation = None
        output = self.model(x)
        output[:, 1].backward()
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activation = self.activation[0]
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8
        return heatmap, output

@app.on_event("startup")
async def load_model():
    global model, device, transform, tta_transforms, grad_cam, mtcnn
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize MTCNN Face Detector
    mtcnn = MTCNN(keep_all=False, device=device, post_process=False)
    logger.info("MTCNN Face Detector initialized.")
    
    checkpoint_path = "outputs/checkpoints/best_model.pth"
    model_name = "efficientnet-b1"
    
    try:
        logger.info(f"Loading model from {checkpoint_path}")
        model = DeepFakeDetector(model_name=model_name, pretrained=False)
        model.load_checkpoint(checkpoint_path, device=str(device))
        model = model.to(device)
        model.eval()
        
        target_layer = model.backbone._conv_head
        grad_cam = GradCAM(model, target_layer)
        
        image_size = 240 if 'b1' in model_name else 224
        transform = get_val_transforms(image_size)
        tta_transforms = get_test_time_augmentation_transforms(image_size)
        
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Could not load model: {e}")

# Forensic Analysis Helpers
def generate_fft(image_np):
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        magnitude_spectrum = np.uint8(magnitude_spectrum)
        heatmap = cv2.applyColorMap(magnitude_spectrum, cv2.COLORMAP_VIRIDIS)
        return heatmap
    except Exception as e:
        logger.error(f"FFT Error: {e}")
        return np.zeros_like(image_np)

def generate_ela(image_np, quality=90):
    try:
        _, encoded = cv2.imencode('.jpg', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        ela_image = cv2.absdiff(image_np, decoded)
        ela_image = cv2.scaleAdd(ela_image, 20, np.zeros_like(ela_image))
        return ela_image
    except Exception as e:
        logger.error(f"ELA Error: {e}")
        return np.zeros_like(image_np)

def calculate_ensemble_score(neural_conf, ela_img, fft_img):
    """Calculate Ensemble Forensic Threat Level (0-100)"""
    # Neural Score (main weight)
    neural_score = neural_conf * 100
    
    # ELA Score: Higher mean brightness = more manipulation
    ela_brightness = np.mean(ela_img) / 255.0 * 100
    
    # FFT Score: Look for irregularities (simplified: high variance = suspicious)
    fft_gray = cv2.cvtColor(fft_img, cv2.COLOR_BGR2GRAY) if len(fft_img.shape) == 3 else fft_img
    fft_variance = np.std(fft_gray) / 128.0 * 100  # Normalized
    
    # Weighted Ensemble (Neural is primary, others are secondary indicators)
    ensemble = (neural_score * 0.6) + (ela_brightness * 0.25) + (fft_variance * 0.15)
    return min(max(ensemble, 0), 100)  # Clamp 0-100


def detect_media_type(file: UploadFile) -> str:
    content_type = (file.content_type or "").lower()
    suffix = Path(file.filename or "").suffix.lower()

    if content_type.startswith("image/") or suffix in IMAGE_EXTENSIONS:
        return "image"
    if content_type.startswith("video/") or suffix in VIDEO_EXTENSIONS:
        return "video"
    if content_type.startswith("audio/") or suffix in AUDIO_EXTENSIONS:
        return "audio"

    raise HTTPException(
        status_code=400,
        detail="Unsupported file type. Please upload an image, video, or audio file.",
    )


def apply_image_model(image_np):
    probs_list = []

    for current_transform in tta_transforms or [transform]:
        transformed = current_transform(image=image_np)
        image_tensor = transformed["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probs = softmax(output.cpu().numpy(), axis=1)[0]

        probs_list.append(probs)

    mean_probs = np.mean(probs_list, axis=0)
    real_prob = float(mean_probs[0])
    fake_prob = float(mean_probs[1])
    return real_prob, fake_prob


def choose_prediction(real_prob, fake_prob, modality="image", fake_frame_ratio=None, face_count=0):
    # Borderline cases: resolve by whichever probability is slightly higher.
    if abs(fake_prob - real_prob) < UNCERTAIN_MARGIN:
        return "FAKE" if fake_prob > real_prob else "REAL"

    if modality == "video":
        strong_fake_consensus = fake_frame_ratio is not None and fake_frame_ratio >= VIDEO_MIN_FAKE_FRAME_RATIO
        if fake_prob >= VIDEO_FAKE_THRESHOLD and strong_fake_consensus:
            return "FAKE"
        if real_prob >= VIDEO_REAL_THRESHOLD:
            return "REAL"
        return "FAKE" if fake_prob > real_prob else "REAL"

    if (
        face_count >= MULTI_FACE_BORDERLINE_COUNT
        and fake_prob >= MULTI_FACE_FAKE_FLOOR
        and abs(fake_prob - real_prob) <= MULTI_FACE_MARGIN
    ):
        return "FAKE"

    if fake_prob >= IMAGE_FAKE_THRESHOLD:
        return "FAKE"
    if real_prob >= IMAGE_REAL_THRESHOLD:
        return "REAL"
    return "FAKE" if fake_prob > real_prob else "REAL"


def run_image_inference(image_np, original_image_np):
    face_detected = False
    face_crop = None
    face_count = 0

    try:
        pil_image = Image.fromarray(image_np)
        boxes, _ = mtcnn.detect(pil_image)
        if boxes is not None and len(boxes) > 0:
            face_count = len(boxes)
            x1, y1, x2, y2 = [int(b) for b in boxes[0]]
            margin = 30
            h, w = image_np.shape[:2]
            x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
            x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
            face_crop = image_np[y1:y2, x1:x2]
            image_np = face_crop
            face_detected = True
            logger.info(f"Face detected and cropped: [{x1},{y1},{x2},{y2}]")
    except Exception as e:
        logger.warning(f"MTCNN detection failed, using full image: {e}")

    try:
        transformed = transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(device)
        image_tensor.requires_grad = True
        model.zero_grad()

        heatmap, _ = grad_cam(image_tensor)
        full_real_prob, full_fake_prob = apply_image_model(original_image_np)

        if face_crop is not None and face_crop.size > 0:
            crop_real_prob, crop_fake_prob = apply_image_model(face_crop)
            real_prob = (full_real_prob * 0.8) + (crop_real_prob * 0.2)
            fake_prob = (full_fake_prob * 0.8) + (crop_fake_prob * 0.2)
        else:
            real_prob = full_real_prob
            fake_prob = full_fake_prob

        prediction = choose_prediction(real_prob, fake_prob, modality="image", face_count=face_count)
        confidence = max(fake_prob, real_prob)

        heatmap_resized = cv2.resize(heatmap, (original_image_np.shape[1], original_image_np.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original_image_np, 0.6, heatmap_colored, 0.4, 0)

        fft_img = generate_fft(original_image_np)
        ela_img = generate_ela(original_image_np)
        threat_level = calculate_ensemble_score(fake_prob, ela_img, fft_img)

        def to_base64(img_arr):
            img_pil = Image.fromarray(img_arr)
            buff = BytesIO()
            img_pil.save(buff, format="JPEG")
            return base64.b64encode(buff.getvalue()).decode("utf-8")

        return {
            "modality": "image",
            "prediction": prediction,
            "confidence": confidence,
            "real_prob": real_prob,
            "fake_prob": fake_prob,
            "threat_level": round(threat_level, 2),
            "face_detected": face_detected,
            "face_count": face_count,
            "heatmap": to_base64(heatmap_colored),
            "overlay": to_base64(overlay),
            "fft": to_base64(fft_img),
            "ela": to_base64(ela_img),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Error running inference: {e}")


def analyze_video_file(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Unable to open video file")

    frame_probs = []
    strong_fake_frames = 0
    face_detected = False
    processed_frames = 0
    frame_index = 0
    sample_every = 5
    max_frames = 24

    try:
        while cap.isOpened() and processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % sample_every != 0:
                frame_index += 1
                continue

            frame_index += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_to_analyze = rgb_frame

            try:
                boxes, _ = mtcnn.detect(Image.fromarray(rgb_frame))
                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = [int(b) for b in boxes[0]]
                    margin = 20
                    h, w = rgb_frame.shape[:2]
                    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
                    x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
                    frame_to_analyze = rgb_frame[y1:y2, x1:x2]
                    face_detected = True
            except Exception as e:
                logger.warning(f"Video face detection failed on frame {processed_frames}: {e}")

            real_prob, fake_prob = apply_image_model(frame_to_analyze)
            probs = np.array([real_prob, fake_prob], dtype=np.float32)
            if fake_prob >= VIDEO_FRAME_FAKE_THRESHOLD:
                strong_fake_frames += 1

            frame_probs.append(probs)
            processed_frames += 1

        if not frame_probs:
            raise HTTPException(status_code=400, detail="No readable frames were found in the uploaded video")

        mean_probs = np.mean(frame_probs, axis=0)
        real_prob = float(mean_probs[0])
        fake_prob = float(mean_probs[1])
        fake_frame_ratio = strong_fake_frames / processed_frames if processed_frames else 0.0
        prediction = choose_prediction(
            real_prob,
            fake_prob,
            modality="video",
            fake_frame_ratio=fake_frame_ratio,
        )
        confidence = max(fake_prob, real_prob)

        return {
            "modality": "video",
            "prediction": prediction,
            "confidence": confidence,
            "real_prob": real_prob,
            "fake_prob": fake_prob,
            "threat_level": round(fake_prob * 100, 2),
            "face_detected": face_detected,
            "frames_analyzed": processed_frames,
            "fake_frame_ratio": round(fake_frame_ratio, 3),
        }
    finally:
        cap.release()


def analyze_audio_file(audio_path: Path):
    try:
        heuristics = extract_audio_heuristics(str(audio_path))
    except Exception as e:
        logger.error(f"Audio inference error: {e}")
        raise HTTPException(status_code=400, detail=f"Unable to process audio file: {e}")

    heuristic_fake_prob = heuristics["heuristic_fake_prob"]
    fake_prob = min(max(heuristics["final_fake_prob"], 0.0), 1.0)
    real_prob = 1.0 - fake_prob
    prediction = "FAKE" if fake_prob > real_prob else "REAL"
    confidence = max(fake_prob, real_prob)

    return {
        "modality": "audio",
        "prediction": prediction,
        "confidence": confidence,
        "real_prob": real_prob,
        "fake_prob": fake_prob,
        "threat_level": round(fake_prob * 100, 2),
        "face_detected": False,
        "audio_heuristic_fake_prob": heuristic_fake_prob,
        "audio_music_energy": heuristics["music_energy"],
        "audio_speech_like_fake_prob": heuristics["speech_like_fake_prob"],
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    media_type = detect_media_type(file)
    contents = await file.read()

    if media_type == "image":
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            image_np = np.array(image)
            return run_image_inference(image_np, image_np.copy())
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")

    suffix = Path(file.filename or "").suffix or (".mp4" if media_type == "video" else ".wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(contents)
        temp_path = Path(temp_file.name)

    try:
        if media_type == "video":
            return analyze_video_file(temp_path)
        return analyze_audio_file(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

# Serve frontend static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

