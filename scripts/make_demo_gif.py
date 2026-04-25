"""Create a lightweight README GIF from assets/demo.mp4.

This script avoids requiring ffmpeg. It uses OpenCV to read the recorded video
and Pillow to write an optimized GIF that GitHub can preview in the README.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image


INPUT_VIDEO = Path("assets/demo.mp4")
OUTPUT_GIF = Path("assets/demo.gif")
TARGET_WIDTH = 640
TARGET_FPS = 10
MAX_DURATION_SECONDS = 12


def resize_frame(frame, target_width: int):
    """Resize while preserving aspect ratio."""

    height, width = frame.shape[:2]
    scale = target_width / width
    target_height = int(height * scale)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def main() -> None:
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(f"Could not find {INPUT_VIDEO}")

    capture = cv2.VideoCapture(str(INPUT_VIDEO))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, round(source_fps / TARGET_FPS))
    max_frames = TARGET_FPS * MAX_DURATION_SECONDS

    frames: list[Image.Image] = []
    frame_index = 0

    while len(frames) < max_frames:
        success, frame_bgr = capture.read()
        if not success:
            break

        if frame_index % frame_interval == 0:
            resized = resize_frame(frame_bgr, TARGET_WIDTH)
            frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

        frame_index += 1

    capture.release()

    if not frames:
        raise RuntimeError("No frames were extracted from the demo video.")

    frame_duration_ms = round(1000 / TARGET_FPS)
    first_frame, *remaining_frames = frames
    first_frame.save(
        OUTPUT_GIF,
        save_all=True,
        append_images=remaining_frames,
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )

    print(f"Created {OUTPUT_GIF} with {len(frames)} frames")


if __name__ == "__main__":
    main()
