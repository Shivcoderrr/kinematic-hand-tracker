"""Application entry point for the real-time AR hand visualizer."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2

try:
    import winsound
except ImportError:  # pragma: no cover - only relevant on non-Windows systems.
    winsound = None

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


def create_video_writer(frame, fps: float) -> tuple[cv2.VideoWriter, Path]:
    """Create a recording writer that matches the rendered frame size."""

    output_dir = Path("captures")
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / f"hand_visualizer_recording_{int(time.time())}.mp4"
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(filename), fourcc, max(20.0, min(fps, 60.0)), (width, height))
    return writer, filename


def play_sound(frequency: int = 880, duration_ms: int = 80) -> None:
    """Play a tiny non-blocking cue when Windows sound support is available."""

    if winsound is None:
        return

    threading.Thread(
        target=winsound.Beep,
        args=(frequency, duration_ms),
        daemon=True,
    ).start()


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
    pinch_candidate_frames = 0
    pinch_latched = False
    video_writer: cv2.VideoWriter | None = None
    recording_path: Path | None = None
    combine_portal_active = False

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
            close_signal = any(
                gestures.is_pinching(
                    hand,
                    INTERACTION.pinch_close_ratio,
                    INTERACTION.pinch_min_close_pixels,
                )
                for hand in hands
            )
            release_signal = not hands or all(
                not gestures.is_pinching(
                    hand,
                    INTERACTION.pinch_release_ratio,
                    INTERACTION.pinch_min_close_pixels,
                )
                for hand in hands
            )

            # Require a stable pinch for a few frames, then latch it until the
            # fingers open again. This keeps fast hand motion from changing mode.
            if release_signal:
                pinch_candidate_frames = 0
                pinch_latched = False
            elif close_signal:
                pinch_candidate_frames += 1
            else:
                pinch_candidate_frames = 0

            pinch_confirmed = pinch_candidate_frames >= INTERACTION.pinch_confirm_frames
            can_switch_mode = current_time - last_pinch_switch_time >= INTERACTION.pinch_cooldown_seconds
            if pinch_confirmed and not pinch_latched:
                if can_switch_mode:
                    visuals.cycle_mode()
                    play_sound(980 if visuals.mode == 4 else 720)
                    last_pinch_switch_time = current_time
                pinch_latched = True

            pinch_active = pinch_latched or pinch_confirmed
            output = visuals.render(
                frame,
                hands,
                fps,
                pinch_active=pinch_active,
                recording_active=video_writer is not None,
                combine_portal_active=combine_portal_active,
            )

            if video_writer is not None:
                video_writer.write(output)

            cv2.imshow(WINDOW_NAME, output)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key in (ord("1"), ord("2"), ord("3"), ord("4")):
                visuals.mode = int(chr(key))
                play_sound(980 if visuals.mode == 4 else 720)
            if key == ord("m"):
                visuals.cycle_mode()
                play_sound(980 if visuals.mode == 4 else 720)
            if key == ord("b"):
                combine_portal_active = not combine_portal_active
                play_sound(1040 if combine_portal_active else 540, 70)
                print(f"Combined portal: {'on' if combine_portal_active else 'off'}")
            if key in (ord("s"), ord("c")):
                path = save_screenshot(output)
                play_sound(1180, 65)
                print(f"Saved screenshot: {path}")
            if key == ord("r"):
                if video_writer is None:
                    video_writer, recording_path = create_video_writer(output, fps)
                    if not video_writer.isOpened():
                        video_writer.release()
                        video_writer = None
                        print("Could not start recording. OpenCV video writer failed to open.")
                    else:
                        play_sound(1320, 80)
                        print(f"Started recording: {recording_path}")
                else:
                    video_writer.release()
                    video_writer = None
                    play_sound(520, 90)
                    print(f"Saved recording: {recording_path}")
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"Saved recording: {recording_path}")
        tracker.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
