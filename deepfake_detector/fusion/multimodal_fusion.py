def fuse_predictions(image_pred=None, video_pred=None, audio_pred=None):
    preds = []

    if image_pred is not None:
        preds.append(image_pred)
    if video_pred is not None:
        preds.append(video_pred)
    if audio_pred is not None:
        preds.append(audio_pred)

    return sum(preds) / len(preds)