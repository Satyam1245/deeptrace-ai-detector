import torch
from deepfake_detector.models.audio_model import AudioCNN
from deepfake_detector.audio.audio_processor import extract_mfcc

def predict_audio(audio_path, model_path):
    model = AudioCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    mfcc = extract_mfcc(audio_path)
    mfcc = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        pred = model(mfcc)

    return pred.item()
