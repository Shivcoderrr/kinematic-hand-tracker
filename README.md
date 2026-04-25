# Kinematic Hand Tracker

A real-time AR hand-tracking visualizer built with Python, OpenCV, and MediaPipe.
It detects 21 hand landmarks from a webcam feed, renders interactive neon
visuals, and supports gesture-based mode switching using a thumb-index pinch.

## Demo

![Kinematic Hand Tracker demo](assets/demo.gif)

The source recording used to generate this GIF is kept locally as
`assets/demo.mp4` and is ignored by Git to avoid committing large media files.

## Highlights

- Real-time hand landmark detection with MediaPipe
- Three AR visual modes: neon web, skeleton glow, and fingertip lasers
- Thumb-index pinch gesture for hands-free mode switching
- Multi-hand fingertip connections for interactive visual effects
- Optimized glow renderer with one blur pass per frame
- Downscaled inference pipeline for steadier FPS
- Modular Python architecture with separate tracking, gesture, rendering, and app layers
- Screenshot capture and on-screen runtime HUD

## Skills Demonstrated

| Area | What this project shows |
| --- | --- |
| Computer vision | Webcam capture, hand landmark detection, coordinate mapping |
| Real-time systems | Frame loop design, FPS monitoring, performance optimization |
| Graphics programming | OpenCV overlays, glow effects, alpha blending, visual modes |
| Interaction design | Gesture-based controls with cooldown and state handling |
| Python engineering | Modular OOP structure, dataclasses, configuration separation |
| Portfolio polish | Demo GIF, architecture docs, reproducible setup |

## Tech Stack

- Python 3.11
- OpenCV
- MediaPipe 0.10.21
- NumPy
- Pillow, for demo GIF generation

## Project Structure

```text
HAND-TRACKING-VISUALIZER/
|-- assets/
|   |-- demo.gif
|   `-- demo.mp4
|-- docs/
|   `-- architecture.md
|-- scripts/
|   |-- install.bat
|   |-- make_demo_gif.py
|   `-- run.bat
|-- src/
|   |-- config.py
|   |-- gesture_analyzer.py
|   |-- hand_tracker.py
|   |-- main.py
|   `-- visual_effects.py
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## How It Works

1. OpenCV captures frames from the webcam.
2. MediaPipe detects up to two hands and returns 21 normalized landmarks per hand.
3. The tracker converts landmarks into screen-space coordinates.
4. The gesture analyzer calculates fingertip distances and detects pinch gestures.
5. The renderer draws visual effects using OpenCV lines, circles, blur, and blending.
6. The app loop handles keyboard input, screenshot capture, FPS display, and gesture mode switching.

The neon glow is built by drawing all glow lines and circles onto one overlay,
blurring that overlay once, and blending it back into the webcam frame. This is
much faster than blurring a separate image for every line.

## Getting Started

This project is designed for **Python 3.11**. Avoid Python 3.14 for this project
because MediaPipe support can lag behind the newest Python releases.

Create a virtual environment:

```bash
py -3.11 -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it in Git Bash:

```bash
source .venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python src/main.py
```

If activation is confusing, run the virtual environment Python directly:

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe src/main.py
```

On Windows, you can also use:

```bat
scripts\install.bat
scripts\run.bat
```

## Controls

| Input | Action |
| --- | --- |
| `1` | Neon web mode |
| `2` | Skeleton glow mode |
| `3` | Fingertip laser mode |
| `m` | Cycle visual mode |
| Thumb-index pinch | Cycle visual mode |
| `s` | Save screenshot to `captures/` |
| `q` | Quit |

## Performance Notes

- MediaPipe runs on a resized inference frame, while visuals render on the display frame.
- The hand model uses `model_complexity = 0` for faster CPU inference.
- The glow renderer accumulates all glow primitives into one overlay and blurs once per frame.
- Default camera resolution is `960x540`, which balances quality and real-time performance.

If your machine still struggles, lower `processing_width` in `src/config.py`:

```python
processing_width: int = 480
```

## Regenerate The Demo GIF

After replacing `assets/demo.mp4` with a new recording, regenerate the README GIF:

```bash
.venv/Scripts/python.exe scripts/make_demo_gif.py
```

## Documentation

For a deeper explanation of the design, read [docs/architecture.md](docs/architecture.md).

## Future Improvements

- Add gesture-based screenshot capture
- Add configurable visual themes
- Add a benchmark mode for measuring tracker and renderer time separately
- Build a web version with React, TypeScript, and MediaPipe Web

## License

This project is available under the MIT License.
