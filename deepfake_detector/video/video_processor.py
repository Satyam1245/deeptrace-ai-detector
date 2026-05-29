import cv2

def extract_frames(video_path, frame_rate=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_rate == 0:
            frames.append(frame)

        count += 1

    cap.release()
    return frames