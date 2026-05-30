import cv2
import numpy as np
from collections import deque
import time
import traceback
from typing import Optional, List, Tuple, Dict
import threading
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Import config for fighting parameters
try:
    from . import config
except ImportError:
    import config

class ActivityRecognizer:
    """Activity Recognition for fighting/violence detection using local YOLO model."""

    def __init__(self, model_type="yolo", confidence_threshold=None):
        self.model_type = model_type
        # Use config values if available, otherwise use defaults
        self.confidence_threshold = confidence_threshold or getattr(config, 'FIGHTING_CONFIDENCE_THRESHOLD', 0.3)  # Lower threshold for better detection
        self.frame_buffer = deque(maxlen=8)  # Smaller buffer for fast detection

        self.last_prediction_time = 0
        # Use configurable prediction interval
        self.prediction_interval = getattr(config, 'FIGHTING_PREDICTION_INTERVAL', 1.0)  # Faster for local model
        self.is_initialized = False
        self.lock = threading.Lock()

        # Fighting detection model
        self.fighting_model = None
        self.fighting_class_names = getattr(config, 'FIGHTING_CLASS_NAMES', ['fighting'])

        if YOLO_AVAILABLE:
            self._initialize_fighting_model()
        else:
            print("[ActivityRecognizer] YOLO not available for fighting detection")
            self.is_initialized = False

    def _initialize_fighting_model(self):
        """Initialize the fighting detection YOLO model with proper safety settings."""
        try:
            fighting_model_path = getattr(config, 'FIGHTING_MODEL_NAME', str(config.BASE_DIR / 'best1.pt'))

            if not os.path.isfile(fighting_model_path):
                print(f"[ActivityRecognizer] Fighting model not found: {fighting_model_path}")
                self.is_initialized = False
                return

            print(f"[ActivityRecognizer] Loading fighting detection model: {fighting_model_path}")

            # Handle PyTorch safety settings for model loading
            try:
                import torch
                # Set weights_only to False for older model compatibility
                _orig_torch_load = torch.load
                def _patched_torch_load(*args, **kwargs):
                    kwargs.setdefault('weights_only', False)
                    return _orig_torch_load(*args, **kwargs)
                torch.load = _patched_torch_load

                # Add safe globals for ultralytics models
                try:
                    from torch.serialization import add_safe_globals
                    from ultralytics.nn.tasks import DetectionModel
                    add_safe_globals([DetectionModel])
                except ImportError:
                    pass

                self.fighting_model = YOLO(fighting_model_path)

                # Print model information for debugging
                if hasattr(self.fighting_model, 'names'):
                    print(f"[ActivityRecognizer] Model classes: {self.fighting_model.names}")
                    # Update fighting class names based on actual model
                    self.fighting_class_names = list(self.fighting_model.names.values())

                self.is_initialized = True
                print(f"[ActivityRecognizer] Fighting detection model initialized successfully with classes: {self.fighting_class_names}")

            except Exception as e:
                print(f"[ActivityRecognizer] Error loading model with safety settings: {e}")
                # Try alternative loading method
                try:
                    from torch.serialization import safe_globals
                    from ultralytics.nn.tasks import DetectionModel

                    with safe_globals([DetectionModel]):
                        self.fighting_model = YOLO(fighting_model_path)

                    if hasattr(self.fighting_model, 'names'):
                        print(f"[ActivityRecognizer] Model classes: {self.fighting_model.names}")
                        self.fighting_class_names = list(self.fighting_model.names.values())

                    self.is_initialized = True
                    print(f"[ActivityRecognizer] Fighting detection model loaded with safe_globals: {self.fighting_class_names}")

                except Exception as e2:
                    print(f"[ActivityRecognizer] Failed to load model with safe_globals: {e2}")
                    self.is_initialized = False

        except Exception as e:
            print(f"[ActivityRecognizer] Failed to initialize fighting model: {e}")
            traceback.print_exc()
            self.is_initialized = False

    def add_frame(self, frame: np.ndarray):
        """Add a frame to the buffer for analysis."""
        if not self.is_initialized:
            return

        with self.lock:
            # Keep original frame for better detection
            self.frame_buffer.append(frame.copy())

    def predict_activity(self) -> Optional[Dict]:
        """Predict fighting activity using local YOLO model."""
        if not self.is_initialized or not self.fighting_model:
            return None

        # Check if we have any frames and enough time has passed
        if len(self.frame_buffer) == 0:
            return None

        current_time = time.time()
        if current_time - self.last_prediction_time < self.prediction_interval:
            return None

        self.last_prediction_time = current_time

        try:
            with self.lock:
                if not self.frame_buffer:
                    return None
                # Use the latest frame for detection
                frame = self.frame_buffer[-1]

            # Run fighting detection on the frame
            results = self.fighting_model(frame, verbose=False)

            # Process the results
            return self._process_fighting_results(results, frame)

        except Exception as e:
            print(f"[ActivityRecognizer] Error in fighting prediction: {e}")
            traceback.print_exc()
            return None

    def _process_fighting_results(self, results, frame) -> Optional[Dict]:
        """Process YOLO fighting detection results."""
        try:
            fighting_detections = []

            for r in results:
                if not hasattr(r, 'boxes') or r.boxes is None:
                    continue

                for box in r.boxes:
                    cls_id = int(box.cls[0]) if box.cls is not None else -1
                    confidence = float(box.conf[0]) if box.conf is not None else 0.0

                    # Get class name from fighting model
                    if hasattr(self.fighting_model, 'names') and cls_id in self.fighting_model.names:
                        class_name = self.fighting_model.names[cls_id]
                    else:
                        class_name = f"class_{cls_id}"

                    # Check if this is a fighting-related detection
                    # Accept any detection above threshold - let the model decide what's fighting
                    if confidence >= self.confidence_threshold:
                        xyxy = box.xyxy[0].tolist() if box.xyxy is not None else [0, 0, 0, 0]
                        x1, y1, x2, y2 = map(int, xyxy)

                        fighting_detections.append({
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                            'class_id': cls_id
                        })

                        print(f"[ActivityRecognizer] Detection: Class {class_name} (ID: {cls_id}), Confidence: {confidence:.3f}")

            if fighting_detections:
                # Get the highest confidence detection
                best_detection = max(fighting_detections, key=lambda x: x['confidence'])

                print(f"[ActivityRecognizer] Fighting activity detected! Best: {best_detection['class_name']}, Confidence: {best_detection['confidence']:.3f}")

                return {
                    'activity': 'fighting',
                    'confidence': best_detection['confidence'],
                    'method': 'yolo_local',
                    'class_name': best_detection['class_name'],
                    'bbox': best_detection['bbox'],
                    'detections': fighting_detections,
                    'detection_count': len(fighting_detections)
                }

            return None

        except Exception as e:
            print(f"[ActivityRecognizer] Error processing fighting results: {e}")
            traceback.print_exc()
            return None

    def get_status(self) -> Dict:
        """Get current status of the activity recognizer."""
        return {
            'initialized': self.is_initialized,
            'model_type': self.model_type,
            'buffer_size': len(self.frame_buffer),
            'confidence_threshold': self.confidence_threshold,
            'yolo_available': YOLO_AVAILABLE,
            'fighting_model_loaded': self.fighting_model is not None,
            'fighting_classes': self.fighting_class_names
        }
