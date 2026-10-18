from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

MODEL_PATH = Path(__file__).with_name("models") / "hand_landmarker.task"
HAND_CONNECTIONS = [
    (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (0, 17), (1, 2), (2, 3), 
    (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14), 
    (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
]

def open_camera():
    """Try common Windows camera backends and indexes."""
    attempts = [
        (0, cv2.CAP_DSHOW),
        (1, cv2.CAP_DSHOW),
        (0, cv2.CAP_MSMF),
        (0, cv2.CAP_ANY),
    ]

    for index, backend in attempts:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        # Ekran genisletme ayarlari
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        success, frame = cap.read()
        if success and frame is not None:
            print(f"Kamera acildi: index={index}, backend={backend}")
            return cap

        cap.release()

    return None

def create_landmarker():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"El takip modeli bulunamadi: {MODEL_PATH}\n"
            "Model dosyasini models klasorune hand_landmarker.task adiyla koyun."
        )

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return HandLandmarker.create_from_options(options)

def draw_hand_skeleton(frame, landmarks):
    height, width = frame.shape[:2]
    points = [(int(point.x * width), int(point.y * height)) for point in landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)

cap = open_camera()
if cap is None:
    raise RuntimeError(
        "Kamera baglantisi kurulamadi. Kameranin baska bir uygulamada acik "
        "olmadigindan ve Windows kamera izinlerinin acik oldugundan emin olun."
    )

landmarker = create_landmarker()
canvas = None
prev_x, prev_y = 0, 0

try:
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print("Kameradan goruntu okunamadi.")
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.monotonic() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                x = int(hand_landmarks[8].x * frame.shape[1])
                y = int(hand_landmarks[8].y * frame.shape[0])

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                # Saf kirmizi ve kalinligi 10 yapildi
                cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 10)
                prev_x, prev_y = x, y

                draw_hand_skeleton(frame, hand_landmarks)
        else:
            prev_x, prev_y = 0, 0

        # Maskeleme yontemi ile silikligin giderilmesi
        imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(imgGray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        frame_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        frame = cv2.add(frame_bg, frame_fg)

        cv2.imshow("Havada Yazi Yazma", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            canvas = np.zeros_like(frame)
finally:
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()