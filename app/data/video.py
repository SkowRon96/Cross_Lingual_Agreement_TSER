"""Face-frame extraction from RAVDESS videos (OpenCV + Haar cascade, CPU).

Verified on real RAVDESS 720p footage: the built-in frontal-face Haar cascade
detects the (single, frontal, well-lit) actor face without any extra model.
Returns RGB face crops; running the FER classifier over them is the P3 model's job.
"""
from pathlib import Path

import cv2
import numpy as np

_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def sample_frame_indices(total: int, n: int) -> list[int]:
    if total <= 0:
        return []
    if total <= n:
        return list(range(total))
    return [round(i * (total - 1) / (n - 1)) for i in range(n)]


def detect_and_crop_face(frame_bgr: np.ndarray, margin: float = 0.15) -> np.ndarray | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])  # largest face
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(frame_bgr.shape[1], x + w + mx), min(frame_bgr.shape[0], y + h + my)
    crop_bgr = frame_bgr[y0:y1, x0:x1]
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)


def extract_face_frames(video_path: Path, num_frames: int = 8) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        crops: list[np.ndarray] = []
        for idx in sample_frame_indices(total, num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            face = detect_and_crop_face(frame)
            if face is not None:
                crops.append(face)
        return crops
    finally:
        cap.release()
