# Architecture

This project is intentionally split into small modules so it looks and feels
like a professional computer-vision prototype instead of one large script.

## Pipeline

1. `main.py` captures webcam frames with OpenCV.
2. `hand_tracker.py` sends each frame to MediaPipe and converts the result into
   plain Python data classes.
3. `gesture_analyzer.py` calculates reusable hand signals, such as fingertip
   positions, distances, and pinch gestures.
4. `visual_effects.py` renders AR-style effects using OpenCV overlays, blur,
   and alpha blending.

## Why This Structure Helps

- Tracking logic is separate from rendering logic.
- Visual modes can be added without touching MediaPipe setup.
- Gesture calculations are reusable for future features.
- Configuration lives in one place, which makes experimentation easier.

## Gesture Controls

The app detects a thumb-index pinch by comparing the distance between landmark
4 and landmark 8. The threshold is scaled using the wrist-to-middle-knuckle
distance, which makes the gesture more stable when the hand moves closer to or
farther from the camera.

`main.py` applies a short cooldown and only switches visual mode on the first
frame of a new pinch. This avoids rapid mode cycling when the user holds the
gesture for more than one frame.

## Visual Effect Technique

The neon effect is built from two layers:

1. A thick colored line or circle is drawn onto a black overlay.
2. The overlay is blurred and blended back into the camera frame.
3. A thin bright line is drawn on top to create a crisp AR core.

This is a common real-time graphics trick because it produces a strong glow
without needing a heavy rendering engine.

## Performance Notes

The tracker runs MediaPipe on a smaller copy of the camera frame, then maps the
normalized landmarks back to the display frame. This keeps detection cheaper
without changing the visual output size.

The renderer accumulates all glow lines and circles onto one overlay, blurs that
overlay once, and blends it into the frame. This avoids the expensive pattern of
blurring a full-size image for every individual line.
