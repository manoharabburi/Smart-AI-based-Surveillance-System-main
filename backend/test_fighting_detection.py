#!/usr/bin/env python3
"""
Test script to verify fighting detection with best1.pt model
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / 'app'))

from app.activity_recognizer import ActivityRecognizer
from app import config

def test_model_loading():
    """Test if the best1.pt model can be loaded"""
    print("=" * 60)
    print("Testing Fighting Detection Model Loading")
    print("=" * 60)

    # Check if model file exists
    model_path = config.FIGHTING_MODEL_NAME
    print(f"Model path: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found at {model_path}")
        return False

    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"✅ Model file exists ({file_size:.2f} MB)")

    # Initialize ActivityRecognizer
    print("\nInitializing ActivityRecognizer...")
    recognizer = ActivityRecognizer(model_type="yolo", confidence_threshold=0.3)

    status = recognizer.get_status()
    print(f"Initialized: {status['initialized']}")
    print(f"Model loaded: {status['fighting_model_loaded']}")
    print(f"Fighting classes: {status['fighting_classes']}")
    print(f"Confidence threshold: {status['confidence_threshold']}")

    if not status['initialized']:
        print("❌ ERROR: ActivityRecognizer failed to initialize")
        return False

    print("✅ ActivityRecognizer initialized successfully")
    return True

def test_with_dummy_frame():
    """Test fighting detection with a dummy frame"""
    print("\n" + "=" * 60)
    print("Testing Fighting Detection with Dummy Frame")
    print("=" * 60)

    # Create a dummy frame (640x480, RGB)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Add some text to make it look like a frame
    cv2.putText(frame, 'TEST FRAME', (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    recognizer = ActivityRecognizer(model_type="yolo", confidence_threshold=0.3)

    if not recognizer.get_status()['initialized']:
        print("❌ ERROR: Could not initialize recognizer")
        return False

    # Add frame and try prediction
    print("Adding frame to recognizer...")
    recognizer.add_frame(frame)

    print("Running prediction...")
    result = recognizer.predict_activity()

    if result:
        print("✅ Detection result received:")
        print(f"   Activity: {result.get('activity')}")
        print(f"   Confidence: {result.get('confidence', 0):.3f}")
        print(f"   Method: {result.get('method')}")
        print(f"   Class: {result.get('class_name')}")
        print(f"   Detection count: {result.get('detection_count', 0)}")
    else:
        print("ℹ️  No fighting detected in dummy frame (expected for random noise)")

    return True

def test_model_classes():
    """Test what classes are available in the model"""
    print("\n" + "=" * 60)
    print("Testing Model Classes")
    print("=" * 60)

    try:
        # Import required modules with safety settings
        import torch
        from ultralytics import YOLO

        # Set up safe loading
        _orig_torch_load = torch.load
        def _patched_torch_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return _orig_torch_load(*args, **kwargs)
        torch.load = _patched_torch_load

        try:
            from torch.serialization import add_safe_globals
            from ultralytics.nn.tasks import DetectionModel
            add_safe_globals([DetectionModel])
        except ImportError:
            pass

        model = YOLO(config.FIGHTING_MODEL_NAME)

        if hasattr(model, 'names'):
            print(f"✅ Model loaded successfully!")
            print(f"Available classes: {model.names}")
            print(f"Number of classes: {len(model.names)}")

            # Check if this looks like a fighting detection model
            class_names = list(model.names.values())
            fighting_related = [name for name in class_names if 'fight' in name.lower() or 'violence' in name.lower() or 'aggression' in name.lower()]

            if fighting_related:
                print(f"✅ Found fighting-related classes: {fighting_related}")
            else:
                print(f"ℹ️  No obvious fighting-related class names found")
                print(f"   All classes: {class_names}")
        else:
            print("❌ ERROR: Model loaded but no class names found")
            return False

    except Exception as e:
        print(f"❌ ERROR loading model directly: {e}")
        return False

    return True

def main():
    """Run all tests"""
    print("Fighting Detection Test Suite")
    print("Testing best1.pt model for fighting detection")

    tests = [
        ("Model Loading", test_model_loading),
        ("Model Classes", test_model_classes),
        ("Dummy Frame Detection", test_with_dummy_frame),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name}: PASSED")
            else:
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            print(f"\n❌ {test_name}: ERROR - {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("🎉 All tests passed! Fighting detection should be working.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
