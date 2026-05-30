import os
import cv2
import time
import threading
import traceback
import queue
import json
from typing import Optional, List, Tuple, Dict, Any
import platform

import numpy as np

from . import config
from .alert_rules import RuleEngine, PERSON_CLASS_NAME, BAG_CLASS_NAMES
from .activity_recognizer import ActivityRecognizer

# Attempt to import YOLO and Deep SORT
YOLO_AVAILABLE = True
DEEPSORT_AVAILABLE = True
try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO_AVAILABLE = False

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore
except Exception:  # pragma: no cover
    DEEPSORT_AVAILABLE = False

def detect_available_cameras(max_cameras=10):
    """Detect available camera devices."""
    available_cameras = []

    # Test camera indices 0-9
    for camera_id in range(max_cameras):
        try:
            # Use DirectShow backend on Windows for better compatibility
            backend = cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_ANY
            cap = cv2.VideoCapture(camera_id, backend)

            if cap.isOpened():
                # Try to read a frame to confirm camera is working
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Get camera properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    camera_info = {
                        'id': camera_id,
                        'name': f'Camera {camera_id}',
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'type': 'USB/Built-in'
                    }

                    # Try to get more detailed camera name (Windows specific)
                    if platform.system() == 'Windows':
                        try:
                            import winreg
                            camera_info['name'] = f'Camera {camera_id} ({width}x{height})'
                        except:
                            pass

                    available_cameras.append(camera_info)

            cap.release()

        except Exception as e:
            # Camera not available or error occurred
            continue

    # Add option for IP cameras and video files
    available_cameras.append({
        'id': 'ip_camera',
        'name': 'IP Camera (Custom URL)',
        'width': 0,
        'height': 0,
        'fps': 0,
        'type': 'IP'
    })

    available_cameras.append({
        'id': 'video_file',
        'name': 'Video File',
        'width': 0,
        'height': 0,
        'fps': 0,
        'type': 'File'
    })

    return available_cameras

class SimpleTracker:
    """Fallback centroid tracker if Deep SORT not available."""
    def __init__(self, max_distance: float = 80.0, max_age: float = 1.0):
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.max_distance = max_distance
        self.max_age = max_age

    def update(self, detections: List[Tuple[int, int, int, int, float, str]]):
        # detections: list (x1,y1,x2,y2,conf,cls_name)
        now = time.time()
        # Mark tracks as unmatched initially
        for tid, tr in self.tracks.items():
            tr['matched'] = False
        for (x1,y1,x2,y2,conf,cls_name) in detections:
            cx = (x1 + x2)/2
            cy = (y1 + y2)/2
            chosen_id = None
            chosen_dist = 1e9
            for tid, tr in self.tracks.items():
                if tr['cls'] != cls_name:
                    continue
                dist = ((cx - tr['cx'])**2 + (cy - tr['cy'])**2)**0.5
                if dist < self.max_distance and dist < chosen_dist:
                    chosen_dist = dist
                    chosen_id = tid
            if chosen_id is None:
                chosen_id = self.next_id
                self.next_id += 1
                self.tracks[chosen_id] = {'cls': cls_name, 'cx': cx, 'cy': cy, 'last_seen': now, 'matched': True,
                                          'box': (x1,y1,x2,y2), 'conf': conf}
            else:
                tr = self.tracks[chosen_id]
                tr.update({'cx': cx, 'cy': cy, 'last_seen': now, 'matched': True, 'box': (x1,y1,x2,y2), 'conf': conf})
        # Remove stale
        stale = [tid for tid,tr in self.tracks.items() if (now - tr['last_seen']) > self.max_age]
        for tid in stale:
            del self.tracks[tid]
        # Return track-like objects
        out = []
        for tid, tr in self.tracks.items():
            d = {
                'track_id': tid,
                'cls_name': tr['cls'],
                'bbox': tr['box'],
                'centroid': (tr['cx'], tr['cy']),
                'conf': tr.get('conf', 0.0)
            }
            out.append(d)
        return out

class VideoProcessor:
    def __init__(self, alert_queue: 'queue.Queue[dict]'):
        self.alert_queue = alert_queue
        self.rule_engine = RuleEngine(self._emit_alert)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.model = None
        self.class_names: Dict[int, str] = {}
        self.tracker = None
        self.frame_lock = threading.Lock()
        self.latest_frame_bytes: Optional[bytes] = None
        self.frame_index = 0
        self.last_inference_time = 0.0
        self.camera_source = 0 if config.CAMERA_SOURCE == '0' else config.CAMERA_SOURCE
        self.cap: Optional[cv2.VideoCapture] = None
        self.headless = config.HEADLESS_MODE or not YOLO_AVAILABLE
        # Diagnostic flags
        self.model_loaded = False
        self.camera_open = False
        self.load_error: Optional[str] = None
        self.last_knife_alert_time = 0.0
        self.last_fighting_alert_time = 0.0
        # Initialize Activity Recognizer for fighting detection
        self.activity_recognizer = ActivityRecognizer(model_type="yolo", confidence_threshold=config.FIGHTING_CONFIDENCE_THRESHOLD)
        # Removed Roboflow / supplemental book detection state
        self._init_components()

    def _init_components(self):
        if config.HEADLESS_MODE:
            self.headless = True
            if self.tracker is None:
                self.tracker = SimpleTracker()
            print('[VideoProcessor] HEADLESS_MODE set -> synthetic frames only')
            return
        if YOLO_AVAILABLE and not self.headless:
            model_path = str(config.YOLO_MODEL_NAME)
            if not os.path.isfile(model_path):
                self.load_error = f"Model file not found: {model_path}"
                print(f"[VideoProcessor] {self.load_error}")
                self.headless = True
                if self.tracker is None:
                    self.tracker = SimpleTracker()
                return
            else:
                try:
                    sz = os.path.getsize(model_path)
                    print(f"[VideoProcessor] Found model file {model_path} ({sz/1024/1024:.2f} MB)")
                except Exception:
                    print(f"[VideoProcessor] Found model file {model_path}")
            try:
                import torch  # noqa: F401
                try:
                    _orig_torch_load = torch.load  # type: ignore
                    def _patched_torch_load(*args, **kwargs):
                        kwargs.setdefault('weights_only', False)
                        return _orig_torch_load(*args, **kwargs)
                    torch.load = _patched_torch_load  # type: ignore
                except Exception:
                    pass
                DetectionModel = None
                add_safe_globals = None
                safe_globals_fn = None
                try:
                    from torch.serialization import add_safe_globals, safe_globals  # type: ignore
                    safe_globals_fn = safe_globals
                except Exception:
                    pass
                try:
                    from ultralytics.nn.tasks import DetectionModel  # type: ignore
                except Exception:
                    DetectionModel = None
                if add_safe_globals and DetectionModel:
                    try:
                        add_safe_globals([DetectionModel])
                    except Exception:
                        pass
                try:
                    self.model = YOLO(config.YOLO_MODEL_NAME)
                    if config.CUSTOM_CLASS_NAMES:
                        try:
                            if hasattr(self.model, 'names') and isinstance(self.model.names, dict):
                                self.model.names = {i: name for i, name in enumerate(config.CUSTOM_CLASS_NAMES)}
                            self.class_names = {i: name for i, name in enumerate(config.CUSTOM_CLASS_NAMES)}
                        except Exception:
                            self.class_names = getattr(self.model, 'names', {})
                    else:
                        self.class_names = getattr(self.model, 'names', {})
                    self.model_loaded = True
                    print('[VideoProcessor] Custom YOLO model loaded:', config.YOLO_MODEL_NAME)
                except Exception as e:
                    if safe_globals_fn and DetectionModel:
                        try:
                            with safe_globals_fn([DetectionModel]):
                                self.model = YOLO(config.YOLO_MODEL_NAME)
                                if config.CUSTOM_CLASS_NAMES:
                                    if hasattr(self.model, 'names') and isinstance(self.model.names, dict):
                                        self.model.names = {i: name for i, name in enumerate(config.CUSTOM_CLASS_NAMES)}
                                    self.class_names = {i: name for i, name in enumerate(config.CUSTOM_CLASS_NAMES)}
                                else:
                                    self.class_names = getattr(self.model, 'names', {})
                                self.model_loaded = True
                                print('[VideoProcessor] Custom YOLO model loaded (safe_globals context):', config.YOLO_MODEL_NAME)
                        except Exception:
                            self.load_error = str(e)
                            traceback.print_exc()
                    else:
                        self.load_error = str(e)
                        traceback.print_exc()
            except Exception as e:
                self.load_error = str(e)
                traceback.print_exc()
            if not self.model_loaded:
                self.headless = True
                print('[VideoProcessor] Falling back to HEADLESS (model load failed) ->', self.load_error)
        else:
            self.headless = True
            if not YOLO_AVAILABLE:
                self.load_error = 'YOLO module unavailable'
                print('[VideoProcessor] YOLO unavailable -> headless mode')
        if DEEPSORT_AVAILABLE and not self.headless and self.tracker is None:
            try:
                self.tracker = DeepSort(max_age=30)
                print('[VideoProcessor] Deep SORT tracker initialized')
            except Exception:
                traceback.print_exc()
                self.tracker = None
        if self.tracker is None:
            self.tracker = SimpleTracker()
            print('[VideoProcessor] Using SimpleTracker fallback')

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()

    def _emit_alert(self, **kwargs):
        try:
            self.alert_queue.put_nowait(kwargs)
        except queue.Full:
            pass

    def _get_frame(self) -> Optional[np.ndarray]:
        if self.headless:
            # retry logic: if model is loaded and environment not forcing headless, attempt reopen every 150 frames
            if (not config.HEADLESS_MODE) and self.model_loaded and (self.frame_index % 150 == 0):
                print('[VideoProcessor] Headless retry: attempting to open camera again')
                self.headless = False
                self.cap = None
            else:
                img = np.zeros((480,640,3), dtype=np.uint8)
                cv2.putText(img, 'HEADLESS MODE', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
                reason = 'MODEL' if not self.model_loaded else 'CAM'
                cv2.putText(img, f'Reason:{reason}', (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)
                cv2.putText(img, time.strftime('%H:%M:%S'), (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                return img
        if self.cap is None:
            # choose backend for Windows
            backend_flag = 0
            if isinstance(self.camera_source, int) and platform.system() == 'Windows':
                backend_flag = cv2.CAP_DSHOW
            try:
                if backend_flag:
                    self.cap = cv2.VideoCapture(self.camera_source, backend_flag)
                else:
                    self.cap = cv2.VideoCapture(self.camera_source)
            except Exception as e:
                print('[VideoProcessor] Exception opening camera:', e)
                self.headless = True
                return self._get_frame()
            self.camera_open = bool(self.cap and self.cap.isOpened())
            if not self.camera_open:
                print(f'[VideoProcessor] Failed to open camera source {self.camera_source} -> switching to headless synthetic frames')
                self.headless = True
                return self._get_frame()
            else:
                print(f'[VideoProcessor] Camera opened successfully: {self.camera_source}')

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print('[VideoProcessor] Frame grab failed (ret=False)')
                # attempt soft reset next cycle
                if self.cap:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                self.cap = None
                return None
            return frame
        except Exception as e:
            print(f'[VideoProcessor] OpenCV exception during frame read: {e}')
            # Release problematic camera and switch to headless
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = None
            self.headless = True
            return self._get_frame()

    def change_camera_source(self, new_source):
        try:
            # release existing
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = None
            # update source
            self.camera_source = int(new_source) if str(new_source).isdigit() else new_source
            # if user requested change, attempt to leave headless (unless forced by env)
            if not config.HEADLESS_MODE and self.model_loaded:
                self.headless = False
            print(f"[VideoProcessor] Camera source changed -> {self.camera_source}; headless={self.headless}")
        except Exception as e:
            print('[VideoProcessor] change_camera_source error:', e)

    def get_status(self) -> dict:
        return {
            'headless': self.headless,
            'model_loaded': self.model_loaded,
            'camera_open': self.camera_open,
            'camera_source': self.camera_source,
            'tracker': type(self.tracker).__name__ if self.tracker else None,
            'load_error': self.load_error,
            'frame_index': self.frame_index,
            'config': {
                'CROWD_THRESHOLD': config.CROWD_COUNT_THRESHOLD,
                'LOITERING_SECONDS': config.LOITERING_SECONDS,
                'ABANDONED_BAG_SECONDS': config.ABANDONED_BAG_SECONDS
            }
        }

    def _process_frame(self, frame: np.ndarray):
        detections = []  # list (x1,y1,x2,y2,conf,cls_name)
        knife_detections = []  # store knife boxes for post association
        if not self.headless and self.model is not None:
            try:
                results = self.model(frame, verbose=False)
                for r in results:
                    if not hasattr(r, 'boxes'):
                        continue
                    for box in r.boxes:
                        cls_id = int(box.cls[0]) if box.cls is not None else -1
                        conf = float(box.conf[0]) if box.conf is not None else 0.0
                        cls_name = self.class_names.get(cls_id, str(cls_id))
                        xyxy = box.xyxy[0].tolist()  # x1,y1,x2,y2
                        x1,y1,x2,y2 = map(int, xyxy)
                        detections.append((x1,y1,x2,y2,conf,cls_name))
                        if cls_name == 'knife' and conf >= 0.4:
                            knife_detections.append((x1,y1,x2,y2,conf))
            except Exception:
                traceback.print_exc()

        tracks_out = []
        if isinstance(self.tracker, SimpleTracker):
            tracks_out = self.tracker.update(detections)
        else:
            ds_dets = [ ((x1,y1,x2,y2), conf, cls_name) for (x1,y1,x2,y2,conf,cls_name) in detections ]
            try:
                deep_tracks = self.tracker.update_tracks(ds_dets, frame=frame)
                for t in deep_tracks:
                    if not t.is_confirmed():
                        continue
                    track_id = t.track_id
                    ltrb = t.to_ltrb()
                    x1,y1,x2,y2 = map(int, ltrb)
                    cls_attr = None
                    if hasattr(t, 'det_class'):
                        cls_attr = getattr(t, 'det_class')
                    elif hasattr(t, 'cls'):
                        cls_attr = getattr(t, 'cls')
                    if cls_attr is None and hasattr(t, 'get_class'):
                        try:
                            cls_attr = t.get_class()
                        except Exception:
                            cls_attr = None
                    if not cls_attr:
                        cls_attr = self._infer_class_from_detection((x1,y1,x2,y2), detections)
                    cls_name = str(cls_attr)
                    conf_val = 0.0
                    for (dx1,dy1,dx2,dy2,conf,cn) in detections:
                        if cn == cls_name:
                            iou = self._iou((x1,y1,x2,y2), (dx1,dy1,dx2,dy2))
                            if iou > 0.5:
                                conf_val = conf
                                break
                    cx = (x1+x2)/2
                    cy = (y1+y2)/2
                    tracks_out.append({
                        'track_id': track_id,
                        'cls_name': cls_name,
                        'bbox': (x1,y1,x2,y2),
                        'centroid': (cx, cy),
                        'conf': conf_val
                    })
            except Exception:
                traceback.print_exc()

        persons = []
        bags = []
        for tr in tracks_out:
            track_id = str(tr['track_id'])
            cls_name = tr['cls_name']
            (x1,y1,x2,y2) = tr['bbox']
            cx, cy = tr['centroid']
            self.rule_engine.update_track(track_id, cls_name, cx, cy)
            if cls_name == PERSON_CLASS_NAME:
                persons.append((track_id, cx, cy))
            if cls_name in BAG_CLASS_NAMES:
                bags.append((track_id, cx, cy))

        # Associate bags with nearest person within radius
        for bag_id, bcx, bcy in bags:
            nearest_person = None
            nearest_dist = 1e9
            for pid, pcx, pcy in persons:
                dist = ((bcx-pcx)**2 + (bcy-pcy)**2)**0.5
                if dist < 150 and dist < nearest_dist:
                    nearest_dist = dist
                    nearest_person = pid
            if nearest_person:
                self.rule_engine.mark_bag_near_person(bag_id, nearest_person)

        # Knife alerts with nearest person association and cooldown
        if knife_detections:
            now_time = time.time()
            for (kx1,ky1,kx2,ky2,kconf) in knife_detections:
                if (now_time - self.last_knife_alert_time) > config.KNIFE_ALERT_COOLDOWN:
                    kcx = (kx1+kx2)/2
                    kcy = (ky1+ky2)/2
                    nearest_pid = None
                    nearest_pd = 1e9
                    for pid, pcx, pcy in persons:
                        dist = ((kcx-pcx)**2 + (kcy-pcy)**2)**0.5
                        if dist < config.KNIFE_PERSON_MAX_DIST and dist < nearest_pd:
                            nearest_pd = dist
                            nearest_pid = pid
                    self.last_knife_alert_time = now_time
                    owner_txt = f" near person {nearest_pid}" if nearest_pid else ""
                    self._emit_alert(
                        type="Knife Detected",
                        severity="critical",
                        description=f"Knife detected{owner_txt} (conf {kconf:.2f})",
                        data={"confidence": kconf, "nearest_person_id": nearest_pid}
                    )

        # YOLO Fighting Detection using best1.pt - Always add frames for analysis
        self.activity_recognizer.add_frame(frame)

        # Fighting Detection using local YOLO model - No person requirement needed
        activity_result = self.activity_recognizer.predict_activity()
        fighting_detections = []

        # Always store fighting detections for visual overlay, regardless of alert cooldown
        if activity_result and activity_result.get('activity') == 'fighting':
            # Store fighting detections for visual overlay
            if 'detections' in activity_result:
                fighting_detections = activity_result['detections']

            # Only send alerts if cooldown period has passed
            now_time = time.time()
            if (now_time - self.last_fighting_alert_time) > config.FIGHTING_ALERT_COOLDOWN:
                self.last_fighting_alert_time = now_time

                confidence = activity_result.get('confidence', 0.0)
                method = activity_result.get('method', 'yolo_local')
                class_name = activity_result.get('class_name', 'fighting')
                bbox = activity_result.get('bbox', {})

                # Enhanced alert with YOLO detection details
                print(f"[VideoProcessor] Fighting detected by YOLO best1.pt! Class: {class_name}, Confidence: {confidence:.2f}")
                print(f"[VideoProcessor] Detection details: {activity_result}")

                self._emit_alert(
                    type="Fighting Detected",
                    severity="critical",
                    description=f"Fighting detected by local AI model: {class_name} (confidence: {confidence:.2f})",
                    data={
                        "confidence": confidence,
                        "detection_method": method,
                        "class_name": class_name,
                        "bbox": bbox,
                        "detection_count": activity_result.get('detection_count', 1),
                        "person_count": len(persons) if persons else 0
                    }
                )

        self.rule_engine.evaluate_all()

        # Draw fighting detection overlays FIRST (so they appear prominently)
        for fighting_det in fighting_detections:
            bbox = fighting_det.get('bbox', {})
            if bbox:
                fx1, fy1, fx2, fy2 = bbox.get('x1', 0), bbox.get('y1', 0), bbox.get('x2', 0), bbox.get('y2', 0)
                conf = fighting_det.get('confidence', 0.0)
                class_name = fighting_det.get('class_name', 'fighting')

                # Draw bright red/orange box for fighting detection
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 69, 255), 4)  # Thicker border

                # Add "FIGHTING DETECTED" label with background
                label = f"FIGHTING {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, (fx1, fy1-30), (fx1 + label_size[0] + 10, fy1), (0, 69, 255), -1)
                cv2.putText(frame, label, (fx1+5, fy1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                print(f"[VideoProcessor] Drawing fighting bbox: ({fx1},{fy1}) to ({fx2},{fy2}) conf={conf:.3f}")

        # Draw overlays for other detections
        for tr in tracks_out:
            (x1,y1,x2,y2) = tr['bbox']
            cls_name = tr['cls_name']
            track_id = tr['track_id']
            conf = tr.get('conf', 0.0)

            color = (0,255,0)
            if cls_name in BAG_CLASS_NAMES:
                color = (0,200,255)
            elif cls_name == PERSON_CLASS_NAME:
                color = (255,0,0)
            elif cls_name == 'knife':
                color = (0,0,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label = f"{cls_name} {conf:.2f} ID{track_id}"
            cv2.putText(frame, label, (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add status line
        cv2.putText(frame, f"Tracks: {len(tracks_out)}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        if self.headless:
            cv2.putText(frame, f"HEADLESS (model unavailable)", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
        return frame

    def _infer_class_from_detection(self, bbox, detections):
        x1,y1,x2,y2 = bbox
        for (dx1,dy1,dx2,dy2,conf,cls_name) in detections:
            iou = self._iou((x1,y1,x2,y2), (dx1,dy1,dx2,dy2))
            if iou > 0.5:
                return cls_name
        return 'object'

    @staticmethod
    def _iou(a,b):
        ax1,ay1,ax2,ay2 = a
        bx1,by1,bx2,by2 = b
        inter_x1 = max(ax1,bx1)
        inter_y1 = max(ay1,by1)
        inter_x2 = min(ax2,bx2)
        inter_y2 = min(ay2,by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2-inter_x1)*(inter_y2-inter_y1)
        area_a = (ax2-ax1)*(ay2-ay1)
        area_b = (bx2-bx1)*(by2-by1)
        return inter / float(area_a + area_b - inter + 1e-6)

    def _loop(self):
        target_delay = 1.0 / max(1, config.TARGET_FPS)
        while self.running:
            start = time.time()
            frame = self._get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            self.frame_index += 1
            if self.frame_index % config.INFERENCE_FRAME_SKIP == 0:
                frame = self._process_frame(frame)
            # Encode JPEG
            try:
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    with self.frame_lock:
                        self.latest_frame_bytes = jpeg.tobytes()
            except Exception:
                traceback.print_exc()
            elapsed = time.time() - start
            sleep_time = target_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_latest_frame(self) -> Optional[bytes]:
        with self.frame_lock:
            return self.latest_frame_bytes
