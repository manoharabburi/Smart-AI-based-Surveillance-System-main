"""Standalone inference script for the custom YOLO model (best.pt).

Usage examples:
  python run_inference.py --source 0               # webcam
  python run_inference.py --source path/to/image.jpg
  python run_inference.py --source path/to/folder   # folder of images
  python run_inference.py --source path/to/video.mp4

Options:
  --model PATH        Path to custom model (default: ../best.pt relative to this file)
  --output DIR        Output directory (default: runs/outputs/<timestamp>)
  --imgsz N           Inference image size (default 640)
  --conf  FLOAT       Confidence threshold (default 0.25)
  --show              Display window (if GUI available)
  --save-txt          Save YOLO-format txt results alongside outputs
  --device cpu|cuda:0 CUDA device selection (default auto)

Outputs:
  - Annotated images saved for image sources
  - Annotated video (MP4) for video/webcam at <output>/annotated.mp4
  - Optional .txt per-image detection (class_id x_center y_center w h conf) normalized

NOTE: Only uses the provided custom model (no pretrained auto-download).
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import cv2
import torch
from typing import List

try:
    from ultralytics import YOLO  # type: ignore
except Exception as e:  # pragma: no cover
    print("[FATAL] ultralytics not installed:", e)
    sys.exit(1)

# Custom class list (index aligned)
CUSTOM_CLASS_NAMES = [
    'formal beard',
    'formal hair',
    'formal id card',
    'formal shoes',
    'formal tuck in',
    'in',
    'informal beard',
    'informal hair',
    'informal id card',
    'informal shoes',
    'informal tuck in',
    'wrong bag',
    'school bag',
    'person',
    'knife'
]

SUPPORTED_IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
SUPPORTED_VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

def parse_args():
    p = argparse.ArgumentParser(description="Custom YOLO inference (best.pt)")
    default_model = (Path(__file__).resolve().parent.parent / 'best.pt').as_posix()
    p.add_argument('--source', required=True, help='0 for webcam, path to image/video/folder')
    p.add_argument('--model', default=default_model, help='Path to custom model weights (best.pt)')
    p.add_argument('--output', default='runs/outputs', help='Base output directory')
    p.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    p.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    p.add_argument('--show', action='store_true', help='Show result window')
    p.add_argument('--save-txt', action='store_true', help='Save YOLO txt results')
    p.add_argument('--device', default=None, help='cpu or cuda device (e.g., cuda:0)')
    return p.parse_args()

def is_image(p: Path) -> bool:
    return p.suffix.lower() in SUPPORTED_IMAGE_EXT

def is_video(p: Path) -> bool:
    return p.suffix.lower() in SUPPORTED_VIDEO_EXT

def collect_sources(src: str) -> List[Path]:
    path = Path(src)
    if not path.exists():
        return []
    if path.is_dir():
        files = [p for p in sorted(path.iterdir()) if p.is_file() and (is_image(p) or is_video(p))]
        return files
    if path.is_file():
        return [path]
    return []

def load_model(weights_path: str):
    if not Path(weights_path).exists():
        print(f"[ERROR] Model file not found: {weights_path}")
        sys.exit(1)
    model = YOLO(weights_path)
    # Override names with custom list
    if hasattr(model, 'names') and isinstance(model.names, dict):
        model.names = {i: name for i, name in enumerate(CUSTOM_CLASS_NAMES)}
    return model

def draw_detections(img, boxes, names):
    for box in boxes:
        cls_id = int(box.cls[0]) if box.cls is not None else -1
        conf = float(box.conf[0]) if box.conf is not None else 0.0
        label = names.get(cls_id, str(cls_id))
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        color = (0, 255, 0)
        if label == 'person':
            color = (255, 0, 0)
        elif label == 'knife':
            color = (0, 0, 255)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def save_txt(results, names, txt_dir: Path, stem: str):
    for r in results:
        if not hasattr(r, 'boxes'):
            continue
        lines = []
        h, w = r.orig_shape[:2]
        for box in r.boxes:
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            # Convert to YOLO format normalized
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}")
        if lines:
            out_file = txt_dir / f"{stem}.txt"
            out_file.write_text("\n".join(lines))

def inference_image(model, path: Path, out_dir: Path, conf: float, imgsz: int, save_txt: bool):
    results = model(path.as_posix(), conf=conf, imgsz=imgsz, verbose=False)
    img = cv2.imread(path.as_posix())
    for r in results:
        if not hasattr(r, 'boxes'):
            continue
        img = draw_detections(img, r.boxes, model.names)
    out_path = out_dir / path.name
    cv2.imwrite(out_path.as_posix(), img)
    if save_txt:
        txt_dir = out_dir / 'labels'
        txt_dir.mkdir(exist_ok=True, parents=True)
        save_txt(results, model.names, txt_dir, path.stem)
    print(f"[IMAGE] Saved: {out_path}")

def inference_video(model, path: Path | int, out_dir: Path, conf: float, imgsz: int, show: bool, save_txt: bool):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {path}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_path = out_dir / 'annotated.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path.as_posix(), fourcc, fps, (width, height))
    frame_index = 0
    txt_dir = out_dir / 'labels' if save_txt else None
    if txt_dir:
        txt_dir.mkdir(exist_ok=True, parents=True)
    print(f"[VIDEO] Writing to {out_path}")
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        results = model(frame, conf=conf, imgsz=imgsz, verbose=False)
        for r in results:
            if not hasattr(r, 'boxes'):
                continue
            frame = draw_detections(frame, r.boxes, model.names)
            if save_txt and txt_dir is not None:
                save_txt([r], model.names, txt_dir, f"frame_{frame_index:06d}")
        writer.write(frame)
        if show:
            cv2.imshow('inference', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    print(f"[VIDEO] Completed {frame_index} frames in {elapsed:.2f}s ({frame_index/elapsed if elapsed>0 else 0:.2f} FPS)")

def main():
    args = parse_args()
    # Device management
    if args.device:
        if 'cuda' in args.device and not torch.cuda.is_available():
            print('[WARN] CUDA requested but not available, falling back to CPU')
            device = 'cpu'
        else:
            device = args.device
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model)
    model.to(device)
    # Determine source type
    if args.source == '0':
        # Webcam
        ts_dir = Path(args.output) / time.strftime('%Y%m%d-%H%M%S')
        ts_dir.mkdir(parents=True, exist_ok=True)
        inference_video(model, 0, ts_dir, args.conf, args.imgsz, args.show, args.save_txt)
        return
    sources = collect_sources(args.source)
    if not sources:
        print(f"[ERROR] Source not found or unsupported: {args.source}")
        sys.exit(1)
    ts_dir = Path(args.output) / time.strftime('%Y%m%d-%H%M%S')
    ts_dir.mkdir(parents=True, exist_ok=True)
    for p in sources:
        if is_image(p):
            inference_image(model, p, ts_dir, args.conf, args.imgsz, args.save_txt)
        elif is_video(p):
            vid_dir = ts_dir / p.stem
            vid_dir.mkdir(exist_ok=True)
            inference_video(model, p, vid_dir, args.conf, args.imgsz, args.show, args.save_txt)
        else:
            print(f"[SKIP] Unsupported file: {p}")
    print(f"[DONE] Outputs saved under {ts_dir}")

if __name__ == '__main__':
    main()

