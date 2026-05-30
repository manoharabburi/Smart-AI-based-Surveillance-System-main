import React, { useState, useEffect } from 'react';
import { Card } from './ui/primitives.jsx';

export default function CameraSelector() {
  const [cameras, setCameras] = useState([]);
  const [currentCamera, setCurrentCamera] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch available cameras on component mount
  useEffect(() => {
    fetchAvailableCameras();
    fetchCurrentCamera();
  }, []);

  const fetchAvailableCameras = async () => {
    try {
      const response = await fetch('/api/cameras/available');
      const data = await response.json();
      if (data.cameras) {
        setCameras(data.cameras);
      } else if (data.error) {
        setError(data.error);
      }
    } catch (err) {
      setError('Failed to fetch available cameras');
      console.error('Error fetching cameras:', err);
    }
  };

  const fetchCurrentCamera = async () => {
    try {
      const response = await fetch('/api/cameras/current');
      const data = await response.json();
      setCurrentCamera(data.current_camera);
    } catch (err) {
      console.error('Error fetching current camera:', err);
    }
  };

  const handleCameraChange = async (event) => {
    const selectedCameraId = event.target.value;

    if (selectedCameraId === 'refresh') {
      fetchAvailableCameras();
      return;
    }

    setLoading(true);
    setError(null);

    try {
      let cameraSource = selectedCameraId;

      // Handle special cases
      if (selectedCameraId === 'ip_camera') {
        const url = prompt('Enter IP camera URL (e.g., http://192.168.1.100:8080/video):');
        if (!url) {
          setLoading(false);
          return;
        }
        cameraSource = url;
      } else if (selectedCameraId === 'video_file') {
        const filePath = prompt('Enter video file path:');
        if (!filePath) {
          setLoading(false);
          return;
        }
        cameraSource = filePath;
      }

      const response = await fetch('/api/camera/source', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ source: cameraSource }),
      });

      const result = await response.json();

      if (result.ok) {
        setCurrentCamera(cameraSource);
        // Refresh camera list to get updated status
        setTimeout(fetchCurrentCamera, 1000);
      } else {
        setError(result.error || 'Failed to change camera');
      }
    } catch (err) {
      setError('Failed to change camera source');
      console.error('Error changing camera:', err);
    } finally {
      setLoading(false);
    }
  };

  const getCameraDisplayName = (camera) => {
    if (camera.type === 'IP') {
      return `${camera.name}`;
    } else if (camera.type === 'File') {
      return `${camera.name}`;
    } else {
      return `${camera.name} (${camera.width}x${camera.height})`;
    }
  };

  const getCurrentCameraName = () => {
    if (currentCamera === null) return 'Loading...';

    const camera = cameras.find(c => c.id === currentCamera);
    if (camera) {
      return getCameraDisplayName(camera);
    }

    // Handle custom IP cameras or file paths
    if (typeof currentCamera === 'string' && currentCamera.startsWith('http')) {
      return `IP Camera: ${currentCamera}`;
    } else if (typeof currentCamera === 'string' && currentCamera.includes('.')) {
      return `Video File: ${currentCamera}`;
    }

    return `Camera ${currentCamera}`;
  };

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-neutral-300">Camera Source</h3>
        <button
          onClick={fetchAvailableCameras}
          className="text-xs px-2 py-1 bg-blue-600/70 hover:bg-blue-600 text-white rounded transition-colors"
          title="Refresh camera list"
        >
          🔄
        </button>
      </div>

      <div className="space-y-3">
        <div className="text-xs text-neutral-400">
          Current: <span className="text-neutral-200">{getCurrentCameraName()}</span>
        </div>

        <select
          value={currentCamera || ''}
          onChange={handleCameraChange}
          disabled={loading}
          className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded text-white text-sm focus:outline-none focus:border-blue-500"
        >
          <option value="">Select Camera...</option>
          <option value="refresh" disabled>--- Refresh List ---</option>
          {cameras.map((camera) => (
            <option key={camera.id} value={camera.id}>
              {getCameraDisplayName(camera)}
            </option>
          ))}
        </select>

        {loading && (
          <div className="text-xs text-blue-400 flex items-center">
            <div className="animate-spin mr-2">⏳</div>
            Switching camera...
          </div>
        )}

        {error && (
          <div className="text-xs text-red-400 bg-red-900/20 p-2 rounded">
            {error}
          </div>
        )}

        {cameras.length === 0 && !loading && (
          <div className="text-xs text-yellow-400 bg-yellow-900/20 p-2 rounded">
            No cameras detected. Make sure cameras are connected and try refreshing.
          </div>
        )}

        <div className="text-xs text-neutral-500 mt-2">
          <div>• USB/Built-in cameras are detected automatically</div>
          <div>• IP Camera: Enter custom URL (RTSP/HTTP streams)</div>
          <div>• Video File: Enter path to video file for testing</div>
        </div>
      </div>
    </Card>
  );
}
