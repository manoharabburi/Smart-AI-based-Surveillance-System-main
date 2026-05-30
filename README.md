# Smart AI-based Surveillance System

🚀 **Advanced AI-powered surveillance system with real-time fighting detection using custom YOLO models**

A comprehensive surveillance solution featuring custom trained YOLOv8 models, real-time fighting detection, multi-object tracking, and intelligent threat analysis with a modern React dashboard.

## 🌟 Key Features

### 🤖 **Dual YOLO Model System**
- **Primary Detection**: Custom YOLO model (`best.pt`) for general surveillance with 15 classes
- **Fighting Detection**: Specialized YOLO model (`best1.pt`) for real-time violence/fighting detection
- **Smart Integration**: Both models work together for comprehensive threat detection

### 🥊 **Advanced Fighting Detection**
- **Real-time Analysis**: Live fighting detection using custom-trained `best1.pt` model
- **Visual Indicators**: Bright orange-red bounding boxes with confidence scores
- **Smart Alerts**: Configurable cooldown periods to prevent alert spam
- **High Accuracy**: Optimized confidence thresholds for reliable detection

### 📊 **Multi-Object Tracking & Analysis**
- **Deep SORT Integration**: Advanced multi-object tracking with fallback centroid tracker
- **Persistent Tracking**: Maintains object identities across frames
- **Behavioral Analysis**: Detects loitering, abandoned objects, and crowd dynamics

### 🚨 **Intelligent Alert System**
- **Real-time Alerts**: WebSocket-based instant notifications
- **Multiple Alert Types**:
  - 🥊 Fighting Detection (Critical)
  - 🔪 Knife Detection (Critical) 
  - 👥 Crowd Surge Detection
  - 🎒 Abandoned Bag Detection
  - 🚶 Loitering Detection
- **Alert Persistence**: SQLite database with full alert history
- **Smart Filtering**: Configurable cooldown periods and confidence thresholds

### 🎥 **Video Processing**
- **Multiple Sources**: Support for webcam, IP cameras, and video files
- **Real-time Streaming**: MJPEG video feed with live detection overlays
- **Headless Mode**: Synthetic frame generation when camera unavailable
- **Performance Optimized**: Configurable frame skipping and target FPS

### 🌐 **Modern Web Dashboard**
- **React + Vite**: Fast, modern frontend with Tailwind CSS
- **Real-time Updates**: Live video feed with detection overlays
- **Animated Alerts**: Beautiful alert notifications with Framer Motion
- **Responsive Design**: Works on desktop and mobile devices
- **Camera Controls**: Easy camera source switching

## 🏗️ **System Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend       │    │   AI Models     │
│   (React)       │◄──►│   (FastAPI)      │◄──►│                 │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • Live Video    │    │ • Video Proc.    ���    │ • best.pt       │
│ • Alert Panel   │    │ • YOLO Detection │    │   (15 classes)  │
│ • Camera Ctrl   │    │ • Fighting AI    │    │ • best1.pt      │
│ • Metrics       │    │ • Alert Engine   │    │   (Fighting)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 **Model Classes**

### **Primary Model (best.pt) - 15 Classes:**
```
0: formal beard      8: informal id card
1: formal hair       9: informal shoes  
2: formal id card   10: informal tuck in
3: formal shoes     11: wrong bag
4: formal tuck in   12: school bag
5: in               13: person
6: informal beard   14: knife
7: informal hair
```

### **Fighting Model (best1.pt) - 1 Class:**
```
0: Fighting
```

## 🚀 **Quick Start**

### **Backend Setup**
```bash
cd backend
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional)
set DEBUG=true
set CAMERA_SOURCE=0  # Use camera index 0, or set to RTSP URL

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### **Frontend Setup**
```bash
cd frontend
npm install
npm run dev
```

### **Access the Application**
- 🌐 **Web Dashboard**: http://localhost:5173
- 📡 **API Docs**: http://localhost:8000/docs
- 🎥 **Video Feed**: http://localhost:8000/video_feed
- 🔌 **WebSocket Alerts**: ws://localhost:8000/ws/alerts

## ⚙️ **Configuration**

### **Environment Variables**
```bash
# Camera Settings
CAMERA_SOURCE=0                    # Camera index or RTSP URL

# Detection Thresholds
CROWD_COUNT_THRESHOLD=8            # People count for crowd alert
LOITERING_SECONDS=120              # Seconds before loitering alert
ABANDONED_BAG_SECONDS=30           # Seconds before abandoned bag alert

# Fighting Detection
FIGHTING_CONFIDENCE_THRESHOLD=0.3   # Fighting detection sensitivity
FIGHTING_ALERT_COOLDOWN=5.0        # Seconds between fighting alerts
FIGHTING_PREDICTION_INTERVAL=1.0   # Prediction frequency

# Performance
TARGET_FPS=15                      # Target processing FPS
INFERENCE_FRAME_SKIP=1             # Process every N frames

# System
HEADLESS_MODE=false                # Enable for no camera mode
DEBUG=false                        # Enable debug logging
```

### **Model Configuration**
```python
# Update in backend/app/config.py
YOLO_MODEL_NAME = "path/to/your/best.pt"
FIGHTING_MODEL_NAME = "path/to/your/best1.pt"
```

## 🧪 **Testing Fighting Detection**

We've included a comprehensive test suite for the fighting detection system:

```bash
cd backend
python test_fighting_detection.py
```

**Test Coverage:**
- ✅ Model loading verification
- ✅ Class detection validation  
- ✅ Dummy frame processing
- ✅ Confidence threshold testing

## 📁 **Project Structure**

```
Smart-AI-based-Surveillance-System/
├── README.md
├── .gitignore
├── backend/
│   ├── best.pt                    # Primary YOLO model
│   ├── best1.pt                   # Fighting detection model
│   ├── requirements.txt
│   ├── test_fighting_detection.py # Fighting detection tests
│   └── app/
│       ├── main.py               # FastAPI application
│       ├── config.py             # Configuration settings
│       ├── video_processor_clean.py # Main video processing
│       ├── activity_recognizer.py    # Fighting detection AI
│       ├── alert_rules.py        # Alert logic engine
│       ├── database.py           # Database operations
│       └── models.py             # Data models
└── frontend/
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx               # Main React app
        ├── components/
        │   ├── VideoFeed.jsx     # Live video component
        │   ├── AlertsPanel.jsx   # Alert notifications
        │   └── CameraSelector.jsx # Camera controls
        └── hooks/
            └── useAlerts.js      # Alert management
```

## 🛠️ **API Endpoints**

### **Video & Detection**
- `GET /video_feed` - Live MJPEG video stream with detection overlays
- `GET /status` - System status and diagnostics
- `POST /camera/change` - Change camera source

### **Alerts**
- `GET /alerts` - Retrieve all alerts with pagination
- `GET /recent_alerts` - Get recent alerts (last 24 hours)
- `WebSocket /ws/alerts` - Real-time alert notifications

### **System**
- `GET /` - API health check
- `GET /docs` - Interactive API documentation

## 🔧 **Troubleshooting**

### **Fighting Detection Issues**
```bash
# Test model loading
python backend/test_fighting_detection.py

# Check model files exist
ls backend/*.pt

# Verify configuration
python -c "from app import config; print(config.FIGHTING_MODEL_NAME)"
```

### **Camera Issues**
```bash
# Test available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"

# Force headless mode
set HEADLESS_MODE=true
```

### **Performance Optimization**
```bash
# Reduce processing load
set INFERENCE_FRAME_SKIP=3
set TARGET_FPS=10

# Increase detection sensitivity
set FIGHTING_CONFIDENCE_THRESHOLD=0.2
```

## 🎯 **Use Cases**

- **🏫 School Security**: Monitor for fights, weapons, and unauthorized items
- **🏢 Office Buildings**: Detect workplace violence and security breaches  
- **🏪 Retail Stores**: Prevent theft and monitor customer behavior
- **🏥 Healthcare**: Ensure patient and staff safety
- **🚇 Public Spaces**: Monitor crowds and detect suspicious activities

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **YOLOv8** by Ultralytics for object detection
- **Deep SORT** for multi-object tracking
- **FastAPI** for the high-performance backend
- **React + Vite** for the modern frontend
- **OpenCV** for computer vision processing

## 📞 **Support**

For issues and questions:
- 🐛 **Bug Reports**: Open an issue on GitHub
- 💡 **Feature Requests**: Open an issue with the enhancement label
- 📧 **Contact**: [Your Contact Information]

---

**⚡ Built with AI-powered surveillance technology for enhanced security and safety**
