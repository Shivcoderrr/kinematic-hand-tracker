"""Application entry point for the real-time AR hand visualizer."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from config import CAMERA, INTERACTION, TRACKER, VISUALS
from gesture_analyzer import GestureAnalyzer
from hand_tracker import HandTracker
from visual_effects import VisualEffects


WINDOW_NAME = "Kinematic Hand Tracker"


def create_capture() -> cv2.VideoCapture:
    """Create and configure the webcam capture device."""

    capture = cv2.VideoCapture(CAMERA.device_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA.height)
    return capture


def save_screenshot(frame) -> Path:
    """Save the current frame to the captures folder with a timestamped name."""

    output_dir = Path("captures")
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / f"hand_visualizer_{int(time.time())}.png"
    cv2.imwrite(str(filename), frame)
    return filename


def main() -> None:
    """Run the webcam loop, tracking pipeline, and visual renderer."""

    capture = create_capture()
    if not capture.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions or device index.")

    tracker = HandTracker(TRACKER)
    gestures = GestureAnalyzer()
    visuals = VisualEffects(VISUALS)

    previous_time = time.perf_counter()
    last_pinch_switch_time = 0.0
    was_pinching = False

    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")

            if CAMERA.flip_horizontal:
                # Mirror mode feels natural because moving your right hand moves
                # the hand on the right side of the screen.
                frame = cv2.flip(frame, 1)

            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            hands = tracker.detect(frame)
            pinch_active = any(gestures.is_pinching(hand) for hand in hands)

            # The pinch gesture switches modes only on the first pinch frame.
            # The cooldown prevents a single held pinch from cycling too fast.
            can_switch_mode = current_time - last_pinch_switch_time >= INTERACTION.pinch_cooldown_seconds
            if pinch_active and not was_pinching and can_switch_mode:
                visuals.cycle_mode()
                last_pinch_switch_time = current_time

            was_pinching = pinch_active
            output = visuals.render(frame, hands, fps, pinch_active=pinch_active)

            cv2.imshow(WINDOW_NAME, output)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key in (ord("1"), ord("2"), ord("3")):
                visuals.mode = int(chr(key))
            if key == ord("m"):
                visuals.cycle_mode()
            if key == ord("s"):
                path = save_screenshot(output)
                print(f"Saved screenshot: {path}")
    finally:
        tracker.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
