"""OpenCV rendering engine for AR-style hand visuals."""

from __future__ import annotations

import cv2
import numpy as np

from config import VisualConfig
from gesture_analyzer import FINGER_TIPS, GestureAnalyzer
from hand_tracker import TrackedHand


# MediaPipe's hand graph: each pair defines one anatomical connection.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class VisualEffects:
    """Draws skeletons, glowing lines, labels, and small HUD elements."""

    def __init__(self, config: VisualConfig) -> None:
        self.config = config
        self.gestures = GestureAnalyzer()
        self.mode = 1

    def cycle_mode(self) -> None:
        """Switch between available visual styles."""

        self.mode = 1 if self.mode >= 3 else self.mode + 1

    def render(self, frame, hands: list[TrackedHand], fps: float, pinch_active: bool = False) -> np.ndarray:
        """Render the selected visual mode onto the frame."""

        # Performance note:
        # The first version blurred a new full-frame overlay for every line.
        # That looked nice, but it was extremely expensive when hands appeared.
        # Now each frame gets one glow layer, one blur, and many cheap line draws.
        glow_layer = np.zeros_like(frame)

        if self.mode == 1:
            self._draw_neon_web(frame, glow_layer, hands)
        elif self.mode == 2:
            self._draw_skeleton_glow(frame, glow_layer, hands)
        else:
            self._draw_fingertip_lasers(frame, glow_layer, hands)

        self._apply_glow_layer(frame, glow_layer)

        self._draw_hand_labels(frame, hands)

        if self.config.show_fps:
            self._draw_hud(frame, fps, pinch_active)

        return frame

    def _draw_neon_web(self, frame, glow_layer, hands: list[TrackedHand]) -> None:
        """Draw fingertip-to-fingertip connections like an AR energy web."""

        fingertip_groups = [self.gestures.fingertip_points(hand) for hand in hands]

        for hand, fingertips in zip(hands, fingertip_groups):
            openness = self.gestures.openness_score(hand)
            color = self._color_from_score(openness)

            for point in fingertips:
                self._draw_glow_circle(frame, glow_layer, point, color)

            # Connect fingertips within one hand so the visual remains useful
            # even when only one hand is visible.
            for start_index in range(len(fingertips)):
                for end_index in range(start_index + 1, len(fingertips)):
                    self._draw_glow_line(frame, glow_layer, fingertips[start_index], fingertips[end_index], color)

        # When two hands are visible, connect matching fingertips across hands.
        if len(fingertip_groups) >= 2:
            for left_point, right_point in zip(fingertip_groups[0], fingertip_groups[1]):
                distance = self.gestures.distance(left_point, right_point)
                color = self._color_from_distance(distance)
                self._draw_glow_line(frame, glow_layer, left_point, right_point, color, thickness=3)

    def _draw_skeleton_glow(self, frame, glow_layer, hands: list[TrackedHand]) -> None:
        """Draw the anatomical 21-point skeleton with a professional glow."""

        for hand in hands:
            for start, end in HAND_CONNECTIONS:
                self._draw_glow_line(
                    frame,
                    glow_layer,
                    hand.point(start),
                    hand.point(end),
                    color=(255, 180, 70),
                    thickness=self.config.skeleton_line_thickness,
                )

            for landmark in hand.landmarks:
                self._draw_glow_circle(frame, glow_layer, (landmark.x, landmark.y), color=(255, 255, 255))

    def _draw_fingertip_lasers(self, frame, glow_layer, hands: list[TrackedHand]) -> None:
        """Draw bright lines from the wrist to each fingertip."""

        for hand in hands:
            wrist = hand.point(0)
            for tip_index in FINGER_TIPS:
                tip = hand.point(tip_index)
                distance = self.gestures.distance(wrist, tip)
                color = self._color_from_distance(distance)
                self._draw_glow_line(frame, glow_layer, wrist, tip, color, thickness=3)
                self._draw_glow_circle(frame, glow_layer, tip, color)

    def _draw_glow_line(
        self,
        frame,
        glow_layer,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int | None = None,
    ) -> None:
        """Queue a blurred outer line and draw its sharp inner line."""

        line_thickness = thickness or self.config.core_line_thickness
        cv2.line(glow_layer, start, end, color, self.config.glow_radius, cv2.LINE_AA)
        cv2.line(frame, start, end, (255, 255, 255), line_thickness, cv2.LINE_AA)
        cv2.line(frame, start, end, color, max(1, line_thickness - 1), cv2.LINE_AA)

    def _draw_glow_circle(
        self,
        frame,
        glow_layer,
        center: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        cv2.circle(glow_layer, center, self.config.glow_radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, center, self.config.landmark_radius, (255, 255, 255), -1, cv2.LINE_AA)

    def _apply_glow_layer(self, frame, glow_layer) -> None:
        """Blur and blend the accumulated glow layer once per frame."""

        if not glow_layer.any():
            return

        blurred = cv2.GaussianBlur(
            glow_layer,
            (0, 0),
            sigmaX=self.config.glow_blur_sigma,
            sigmaY=self.config.glow_blur_sigma,
        )
        cv2.addWeighted(blurred, self.config.glow_strength, frame, 1.0, 0, dst=frame)

    def _draw_hand_labels(self, frame, hands: list[TrackedHand]) -> None:
        for hand in hands:
            wrist = hand.point(0)
            cv2.putText(
                frame,
                hand.handedness,
                (wrist[0] + 12, wrist[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (230, 240, 255),
                2,
                cv2.LINE_AA,
            )

    def _draw_hud(self, frame, fps: float, pinch_active: bool) -> None:
        """Draw minimal runtime information without covering the hand area."""

        mode_name = {
            1: "Neon Web",
            2: "Skeleton Glow",
            3: "Fingertip Lasers",
        }[self.mode]

        cv2.putText(frame, f"FPS: {fps:05.1f}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Mode: {mode_name}", (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 255, 220), 2)
        pinch_text = "Pinch: ACTIVE" if pinch_active else "Pinch: ready"
        pinch_color = (80, 255, 120) if pinch_active else (180, 190, 200)
        cv2.putText(frame, pinch_text, (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.62, pinch_color, 2)
        cv2.putText(frame, "Pinch: switch mode  1/2/3: mode  S: screenshot  Q: quit", (20, frame.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 220, 230), 2)

    @staticmethod
    def _color_from_distance(distance: float) -> tuple[int, int, int]:
        """Map distance to a BGR color.

        OpenCV uses BGR, not RGB. Near points become warmer; far points become
        cooler, which makes interaction feel responsive.
        """

        normalized = max(0.0, min(distance / 500.0, 1.0))
        blue = int(255 * normalized)
        green = int(255 * (1.0 - abs(normalized - 0.5) * 1.4))
        red = int(255 * (1.0 - normalized))
        return blue, green, red

    @staticmethod
    def _color_from_score(score: float) -> tuple[int, int, int]:
        normalized = max(0.0, min(score / 260.0, 1.0))
        return int(230 * normalized), 255, int(255 * (1.0 - normalized))
