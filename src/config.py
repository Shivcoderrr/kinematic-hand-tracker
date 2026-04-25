"""Central configuration for the hand visualizer.

Keeping tunable values here makes the project easier to explain and maintain:
the tracking pipeline, rendering style, and keyboard behavior can evolve
without scattering magic numbers across the codebase.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:
    """Webcam capture settings."""

    device_index: int = 0
    width: int = 960
    height: int = 540
    flip_horizontal: bool = True


@dataclass(frozen=True)
class TrackerConfig:
    """MediaPipe hand tracking settings."""

    max_num_hands: int = 2
    model_complexity: int = 0
    processing_width: int = 640
    min_detection_confidence: float = 0.65
    min_tracking_confidence: float = 0.65


@dataclass(frozen=True)
class VisualConfig:
    """Rendering settings for the AR effect."""

    glow_radius: int = 12
    glow_blur_sigma: int = 8
    glow_strength: float = 0.45
    core_line_thickness: int = 2
    skeleton_line_thickness: int = 2
    landmark_radius: int = 4
    show_fps: bool = True


@dataclass(frozen=True)
class InteractionConfig:
    """Gesture-control settings."""

    pinch_cooldown_seconds: float = 0.8


CAMERA = CameraConfig()
TRACKER = TrackerConfig()
VISUALS = VisualConfig()
INTERACTION = InteractionConfig()
