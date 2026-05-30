import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

# Camera source (0=default webcam). Can also be rtsp/http URL
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")

# Alert thresholds
CROWD_COUNT_THRESHOLD = int(os.getenv("CROWD_COUNT_THRESHOLD", 8))
LOITERING_SECONDS = int(os.getenv("LOITERING_SECONDS", 120))
ABANDONED_BAG_SECONDS = int(os.getenv("ABANDONED_BAG_SECONDS", 30))
STILL_MOVEMENT_PX_RADIUS = int(os.getenv("STILL_MOVEMENT_PX_RADIUS", 50))

# DB
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'surveillance.db'}")

# Use only local YOLO model best.pt
DEFAULT_MODEL_PATH = BASE_DIR / 'best.pt'
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", str(DEFAULT_MODEL_PATH))

# Fighting detection model (best1.pt)
FIGHTING_MODEL_PATH = BASE_DIR / 'best1.pt'
FIGHTING_MODEL_NAME = os.getenv("FIGHTING_MODEL_NAME", str(FIGHTING_MODEL_PATH))

# Explicit custom class names for the trained model (index aligned) 0-14
CUSTOM_CLASS_NAMES = [
    'formal beard', 'formal hair', 'formal id card', 'formal shoes', 'formal tuck in',
    'in', 'informal beard', 'informal hair', 'informal id card', 'informal shoes', 'informal tuck in',
    'wrong bag', 'school bag', 'person', 'knife'
]

# Fighting model class names (update these based on your best1.pt model classes)
FIGHTING_CLASS_NAMES = [
    'fighting'  # Add your actual fighting model classes here
]

# Performance
INFERENCE_FRAME_SKIP = int(os.getenv("INFERENCE_FRAME_SKIP", 1))
TARGET_FPS = int(os.getenv("TARGET_FPS", 15))

# Websocket broadcast queue size
ALERT_QUEUE_MAX = int(os.getenv("ALERT_QUEUE_MAX", 100))

# If set, runs without actual camera (generates dummy frames)
HEADLESS_MODE = os.getenv("HEADLESS_MODE", "false").lower() in {"1", "true", "yes"}

# Debug / development mode
DEBUG = os.getenv("DEBUG", "false").lower() in {"1", "true", "yes"}

# Knife alert cooldown
KNIFE_ALERT_COOLDOWN = int(os.getenv("KNIFE_ALERT_COOLDOWN", 30))
# Knife to person max association distance (pixels)
KNIFE_PERSON_MAX_DIST = int(os.getenv("KNIFE_PERSON_MAX_DIST", 200))

# Stationary bag seconds (2 hours default)
BAG_STATIONARY_SECONDS = int(os.getenv("BAG_STATIONARY_SECONDS", 7200))

# Fighting detection settings
FIGHTING_ALERT_COOLDOWN = float(os.getenv("FIGHTING_ALERT_COOLDOWN", 5.0))  # Reduced cooldown for testing
FIGHTING_CONFIDENCE_THRESHOLD = float(os.getenv("FIGHTING_CONFIDENCE_THRESHOLD", 0.3))  # Lower threshold for better detection
FIGHTING_PREDICTION_INTERVAL = float(os.getenv("FIGHTING_PREDICTION_INTERVAL", 1.0))  # Faster predictions
