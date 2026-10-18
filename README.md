# Air Writing with Hand Tracking

Write and draw in the air with your index finger. The webcam tracks your hand in real time and turns the movement of your fingertip into lines on the screen.

Built with Python, OpenCV and MediaPipe Hand Landmarker.

## How It Works

1. The webcam image is captured with OpenCV and mirrored, so the screen behaves like a mirror.
2. MediaPipe Hand Landmarker detects the hand and returns 21 landmark points for it.
3. The tip of the index finger (landmark 8) is used as the pen. Each new position is connected to the previous one with a line drawn on a separate canvas.
4. The canvas is merged onto the live camera image using a mask, so the drawing appears sharp and fully opaque instead of faded.
5. The detected hand skeleton is drawn on top of the image so you can see what the model is tracking.

When the hand leaves the frame, the pen is lifted. Bringing the hand back starts a new stroke instead of connecting to the old one.

## Controls

| Key | Action |
|---|---|
| `c` | Clear the drawing |
| `q` | Quit the application |

## Requirements

- Python 3.9 or newer
- A webcam
- The packages listed in `requirements.txt`: `opencv-python`, `mediapipe`, `numpy`

The hand tracking model (`models/hand_landmarker.task`) is included in the repository, so no extra download is needed.

## Installation

```bash
# Clone the repository
git clone https://github.com/elanurgocel/air-writing-hand-tracking.git
cd air-writing-hand-tracking

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

A window titled "Havada Yazi Yazma" opens with the camera image. Hold one hand up in front of the camera and move your index finger to write.

## Notes

- Only one hand is tracked at a time.
- The camera is opened at 1280x720 when the device supports it.
- On Windows, several camera backends (DirectShow, Media Foundation) and camera indexes are tried automatically until one works. If the camera cannot be opened, make sure it is not being used by another application and that camera access is allowed in Windows privacy settings.
- Good lighting and a plain background improve tracking accuracy.

## Project Structure

```
air-writing-hand-tracking/
├── main.py                     # Application code
├── requirements.txt            # Python dependencies
└── models/
    └── hand_landmarker.task    # MediaPipe hand tracking model
```

## Author

Ela Nur Göçel - [GitHub](https://github.com/elanurgocel)
