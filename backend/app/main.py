import asyncio
import json
import queue
import threading
import time
from typing import List, Set
import traceback

from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, Query, Body, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .database import init_db, get_db
from . import config
from .models import Alert
from .schemas import AlertsList, AlertOut
from .video_processor import VideoProcessor, detect_available_cameras

app = FastAPI(title="Smart Surveillance System", version="0.1.0")

# CORS (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

alert_queue: "queue.Queue[dict]" = queue.Queue(maxsize=config.ALERT_QUEUE_MAX)
video_processor = VideoProcessor(alert_queue=alert_queue)
alert_ws_clients: Set[WebSocket] = set()
stop_event = threading.Event()
loop = None  # will be set on startup


def persist_alert(db: Session, payload: dict) -> Alert:
    alert = Alert(
        camera_id=payload.get("camera_id", "cam_1"),
        type=payload.get("type", "Unknown"),
        description=payload.get("description", ""),
        severity=payload.get("severity", "info"),
        data=payload.get("data", {}),
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


async def broadcast_alert(alert: Alert):
    if not alert_ws_clients:
        return
    message = json.dumps({
        "id": alert.id,
        "timestamp": alert.timestamp.isoformat(),
        "camera_id": alert.camera_id,
        "type": alert.type,
        "description": alert.description,
        "severity": alert.severity,
        "data": alert.data or {},
        "resolved": alert.resolved
    })
    dead: List[WebSocket] = []
    for ws in alert_ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        alert_ws_clients.discard(ws)


def alert_consumer_loop():
    from sqlalchemy.orm import Session
    from .database import SessionLocal
    while not stop_event.is_set():
        try:
            payload = alert_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        db: Session = SessionLocal()
        try:
            alert = persist_alert(db, payload)
            if loop is not None and loop.is_running():
                asyncio.run_coroutine_threadsafe(broadcast_alert(alert), loop)
        except Exception:
            db.rollback()
        finally:
            db.close()


@app.on_event("startup")
async def startup_event():
    init_db()
    video_processor.start()
    global loop
    loop = asyncio.get_running_loop()
    t = threading.Thread(target=alert_consumer_loop, daemon=True)
    t.start()


@app.on_event("shutdown")
async def shutdown_event():
    video_processor.stop()
    stop_event.set()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/alerts", response_model=AlertsList)
def list_alerts(
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    type: str | None = None,
    camera_id: str | None = None,
):
    q = db.query(Alert)
    if type:
        q = q.filter(Alert.type == type)
    if camera_id:
        q = q.filter(Alert.camera_id == camera_id)
    total = q.count()
    rows = q.order_by(Alert.timestamp.desc()).offset(offset).limit(limit).all()
    return {"items": rows, "total": total}


@app.get("/recent_alerts", response_model=list[AlertOut])
def recent_alerts(db: Session = Depends(get_db), limit: int = Query(10, ge=1, le=100)):
    rows = db.query(Alert).order_by(Alert.timestamp.desc()).limit(limit).all()
    return rows


def mjpeg_stream_generator():
    boundary = b"--frame"
    while True:
        frame = video_processor.get_latest_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        yield boundary + b"\r\n" + b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.03)


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(mjpeg_stream_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket):
    await ws.accept()
    alert_ws_clients.add(ws)
    try:
        from .database import SessionLocal
        db = SessionLocal()
        try:
            last = db.query(Alert).order_by(Alert.timestamp.desc()).limit(10).all()
            for a in reversed(last):
                await ws.send_text(json.dumps(a.to_dict()))
        finally:
            db.close()
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        alert_ws_clients.discard(ws)


@app.get("/status")
async def status():
    return video_processor.get_status()


@app.post("/camera/source")
async def change_camera_source(payload: dict = Body(...)):
    src = payload.get("source")
    if src is None:
        return {"error": "missing 'source'"}
    video_processor.change_camera_source(src)
    return {"ok": True, "source": src, "status": video_processor.get_status()}


@app.post("/alerts/{alert_id}/resolve", response_model=AlertOut)
async def resolve_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.resolved = True
    db.commit()
    db.refresh(alert)

    # Broadcast the updated alert to all connected clients
    if loop is not None and loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast_alert(alert), loop)

    return alert


@app.get("/cameras/available")
async def get_available_cameras():
    """Get list of available cameras."""
    try:
        cameras = detect_available_cameras()
        return {"cameras": cameras, "status": "success"}
    except Exception as e:
        print(f"[API] Error detecting cameras: {e}")
        traceback.print_exc()
        return {"cameras": [], "status": "error", "error": str(e)}


@app.get("/cameras/current")
async def get_current_camera():
    """Get current camera information."""
    try:
        status = video_processor.get_status()
        return {
            "current_camera": status["camera_source"],
            "camera_open": status["camera_open"],
            "headless": status["headless"],
            "status": "success"
        }
    except Exception as e:
        print(f"[API] Error getting current camera: {e}")
        return {
            "current_camera": 0,
            "camera_open": False,
            "headless": True,
            "status": "error",
            "error": str(e)
        }


if config.DEBUG:
    @app.post("/debug/alert", response_model=AlertOut)
    def debug_alert(
        payload: dict = Body(..., example={"type": "Test Alert", "description": "Debug injection", "severity": "low"}),
        db: Session = Depends(get_db),
    ):
        """Inject a synthetic alert (DEBUG only)."""
        alert = persist_alert(db, payload)
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(broadcast_alert(alert), loop)
        return alert

# For local dev run: uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
