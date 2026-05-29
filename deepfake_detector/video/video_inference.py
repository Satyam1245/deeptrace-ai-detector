import torch
from deepfake_detector.models.efficientnet import DeepFakeDetector
from deepfake_detector.video.video_processor import extract_frames

def predict_video(video_path, model_path):
    model = DeepFakeDetector()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    frames = extract_frames(video_path)

    predictions = []

    for frame in frames:
        pred = model.predict(frame)  # reuse your existing predict logic
        predictions.append(pred)

    # Aggregate results
    return sum(predictions) / len(predictions)