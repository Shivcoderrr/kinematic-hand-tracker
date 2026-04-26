"""OpenCV rendering engine for AR-style hand visuals."""

from __future__ import annotations

from dataclasses import dataclass
import math

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


@dataclass
class ArcaneParticle:
    position: tuple[float, float]
    velocity: tuple[float, float]
    life: float
    max_life: float
    radius: int
    color: tuple[int, int, int]


@dataclass
class ArcaneTrail:
    center: tuple[int, int]
    axes: tuple[int, int]
    angle: float
    life: float
    max_life: float


class VisualEffects:
    """Draws skeletons, glowing lines, labels, and small HUD elements."""

    def __init__(self, config: VisualConfig) -> None:
        self.config = config
        self.gestures = GestureAnalyzer()
        self.mode = 1
        self._arcane_phase = 0.0
        self._arcane_particles: list[ArcaneParticle] = []
        self._arcane_trails: list[ArcaneTrail] = []

    def cycle_mode(self) -> None:
        """Switch between available visual styles."""

        self.mode = 1 if self.mode >= 4 else self.mode + 1

    def render(
        self,
        frame,
        hands: list[TrackedHand],
        fps: float,
        pinch_active: bool = False,
        recording_active: bool = False,
        combine_portal_active: bool = False,
    ) -> np.ndarray:
        """Render the selected visual mode onto the frame."""

        # Performance note:
        # The first version blurred a new full-frame overlay for every line.
        # That looked nice, but it was extremely expensive when hands appeared.
        # Now each frame gets one glow layer, one blur, and many cheap line draws.
        glow_layer = np.zeros_like(frame)
        dt = max(1.0 / 120.0, min(1.0 / 15.0, 1.0 / max(fps, 1.0)))

        if self.mode == 1:
            self._draw_neon_web(frame, glow_layer, hands)
        elif self.mode == 2:
            self._draw_skeleton_glow(frame, glow_layer, hands)
        elif self.mode == 3:
            self._draw_fingertip_lasers(frame, glow_layer, hands)
        else:
            self._draw_arcane_portals(frame, glow_layer, hands, pinch_active, combine_portal_active, dt)

        self._apply_glow_layer(frame, glow_layer)
        self._arcane_phase += 4.8 * dt

        self._draw_hand_labels(frame, hands)

        if self.config.show_fps:
            self._draw_hud(frame, fps, pinch_active, recording_active, combine_portal_active)

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

    def _draw_arcane_portals(
        self,
        frame,
        glow_layer,
        hands: list[TrackedHand],
        pinch_active: bool,
        combine_portal_active: bool,
        dt: float,
    ) -> None:
        """Draw orange-gold spell shields with cinematic movement and depth."""

        self._update_arcane_trails(dt)
        self._update_arcane_particles(dt)
        self._draw_arcane_trails(frame, glow_layer)

        for hand in hands:
            center, axes, angle, radius, energy = self._arcane_geometry(hand, pinch_active)
            self._add_arcane_trail(center, axes, angle, energy)
            self._spawn_arcane_particles(center, axes, angle, radius, energy)

            self._darken_around_portal(frame, center, int(max(axes) * 1.75))

            deep_orange = self._scale_color((0, 82, 255), energy)
            portal_orange = self._scale_color((0, 162, 255), energy)
            hot_gold = self._scale_color((25, 235, 255), energy)
            white_hot = self._scale_color((170, 255, 255), min(1.25, energy + 0.18))

            self._draw_arcane_beams(frame, glow_layer, center, axes, angle, deep_orange, energy)
            self._draw_arcane_bands(frame, glow_layer, center, axes, angle, energy)
            self._draw_arcane_ticks(frame, glow_layer, center, axes, angle, portal_orange, energy)
            self._draw_arcane_star(frame, glow_layer, center, axes, angle, hot_gold)
            self._draw_arcane_core(frame, glow_layer, center, axes, angle, white_hot, energy)

            for tip in self.gestures.fingertip_points(hand):
                self._draw_glow_line(frame, glow_layer, center, tip, deep_orange, thickness=2)
                cv2.circle(glow_layer, tip, int(self.config.glow_radius * energy) + 5, hot_gold, -1, cv2.LINE_AA)
                cv2.circle(frame, tip, self.config.landmark_radius + 2, white_hot, -1, cv2.LINE_AA)

        if combine_portal_active and len(hands) >= 2:
            self._draw_dual_hand_portal(frame, glow_layer, hands[:2], pinch_active)

        self._draw_arcane_particles(frame, glow_layer)

    def _arcane_geometry(
        self,
        hand: TrackedHand,
        pinch_active: bool,
    ) -> tuple[tuple[int, int], tuple[int, int], float, int, float]:
        center = self._palm_center(hand)
        hand_scale = self.gestures.hand_scale(hand)
        openness = self.gestures.openness_score(hand)
        open_ratio = max(0.48, min(openness / max(hand_scale * 2.5, 1.0), 1.35))
        pinch_boost = 1.28 if pinch_active else 1.0
        energy = max(0.52, min(open_ratio * pinch_boost, 1.45))

        radius = int(max(38.0, min(hand_scale * 1.22 * energy, 170.0)))
        palm_width = self.gestures.distance(hand.point(5), hand.point(17))
        squash = max(0.42, min(palm_width / max(hand_scale * 1.18, 1.0), 0.94))
        axes = (radius, max(30, int(radius * squash)))

        index_mcp = hand.point(5)
        pinky_mcp = hand.point(17)
        angle = math.degrees(math.atan2(pinky_mcp[1] - index_mcp[1], pinky_mcp[0] - index_mcp[0]))
        return center, axes, angle, radius, energy

    def _draw_dual_hand_portal(
        self,
        frame,
        glow_layer,
        hands: list[TrackedHand],
        pinch_active: bool,
    ) -> None:
        left_center = self._palm_center(hands[0])
        right_center = self._palm_center(hands[1])
        center = (
            (left_center[0] + right_center[0]) // 2,
            (left_center[1] + right_center[1]) // 2,
        )
        hand_distance = self.gestures.distance(left_center, right_center)
        if hand_distance < 80:
            return

        angle = math.degrees(math.atan2(right_center[1] - left_center[1], right_center[0] - left_center[0]))
        energy = 1.35 if pinch_active else 1.08
        axes = (
            int(max(70.0, min(hand_distance * 0.62, 220.0))),
            int(max(42.0, min(hand_distance * 0.34, 140.0))),
        )
        self._darken_around_portal(frame, center, int(max(axes) * 1.45))
        self._draw_arcane_beams(frame, glow_layer, center, axes, angle, (0, 95, 255), energy)
        self._draw_arcane_bands(frame, glow_layer, center, axes, angle, energy)
        self._draw_arcane_ticks(frame, glow_layer, center, axes, angle, (0, 185, 255), energy)
        self._draw_arcane_star(frame, glow_layer, center, axes, angle, (45, 245, 255))
        self._draw_arcane_core(frame, glow_layer, center, axes, angle, (185, 255, 255), energy)

    @staticmethod
    def _darken_around_portal(frame, center: tuple[int, int], radius: int) -> None:
        overlay = frame.copy()
        cv2.circle(overlay, center, radius, (4, 8, 18), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.26, frame, 0.74, 0, dst=frame)

    def _draw_arcane_beams(
        self,
        frame,
        glow_layer,
        center: tuple[int, int],
        axes: tuple[int, int],
        ellipse_angle: float,
        color: tuple[int, int, int],
        energy: float,
    ) -> None:
        for beam_index in range(12):
            angle = self._arcane_phase * 0.42 + beam_index * math.tau / 12.0
            start = self._ellipse_point(center, self._scale_axes(axes, 0.18), ellipse_angle, angle)
            end = self._ellipse_point(center, self._scale_axes(axes, 1.35), ellipse_angle, angle)
            cv2.line(glow_layer, start, end, color, int(self.config.glow_radius * energy) + 8, cv2.LINE_AA)
            cv2.line(frame, start, end, (0, 140, 255), 1, cv2.LINE_AA)

    def _draw_arcane_bands(
        self,
        frame,
        glow_layer,
        center: tuple[int, int],
        axes: tuple[int, int],
        ellipse_angle: float,
        energy: float,
    ) -> None:
        band_specs = (
            (1.14, 0.55, 18, 34, (0, 92, 255), 2),
            (1.0, -0.72, 28, 42, (0, 190, 255), 3),
            (0.86, 0.95, 20, 30, (35, 245, 255), 2),
            (0.68, -1.1, 36, 50, (0, 155, 255), 2),
            (0.42, 1.35, 18, 28, (125, 255, 255), 2),
        )

        for axis_scale, speed, arc_length, gap, color, thickness in band_specs:
            start_angle = (self._arcane_phase * speed * 70.0) % 360.0
            ring_axes = self._scale_axes(axes, axis_scale)
            color = self._scale_color(color, energy)
            cv2.ellipse(glow_layer, center, ring_axes, ellipse_angle, 0, 360, color, self.config.glow_radius + thickness, cv2.LINE_AA)
            for arc_start in range(0, 360, gap):
                start = start_angle + arc_start
                end = start + arc_length
                cv2.ellipse(glow_layer, center, ring_axes, ellipse_angle, start, end, color, 5, cv2.LINE_AA)
                cv2.ellipse(frame, center, ring_axes, ellipse_angle, start, end, color, thickness, cv2.LINE_AA)

    def _draw_arcane_ticks(
        self,
        frame,
        glow_layer,
        center: tuple[int, int],
        axes: tuple[int, int],
        ellipse_angle: float,
        color: tuple[int, int, int],
        energy: float,
    ) -> None:
        for tick_index in range(64):
            angle = self._arcane_phase * 1.35 + tick_index * math.tau / 64.0
            start_scale = 0.68 if tick_index % 2 else 0.52
            end_scale = 1.14 if tick_index % 8 == 0 else 1.07
            start = self._ellipse_point(center, self._scale_axes(axes, start_scale), ellipse_angle, angle)
            end = self._ellipse_point(center, self._scale_axes(axes, end_scale), ellipse_angle, angle)
            thickness = 2 if tick_index % 4 else 3
            cv2.line(glow_layer, start, end, color, int(self.config.glow_radius * energy) + 2, cv2.LINE_AA)
            cv2.line(frame, start, end, (20, 245, 255), thickness, cv2.LINE_AA)

        for glyph_index in range(24):
            angle = -self._arcane_phase * 0.9 + glyph_index * math.tau / 24.0
            glyph_center = self._ellipse_point(center, self._scale_axes(axes, 0.91), ellipse_angle, angle)
            tangent = angle + math.pi / 2.0
            half_length = 5 + (glyph_index % 3) * 3
            start = self._ellipse_point(glyph_center, (half_length, half_length), ellipse_angle, tangent)
            end = self._ellipse_point(glyph_center, (half_length, half_length), ellipse_angle, tangent + math.pi)
            cv2.line(frame, start, end, color, 1, cv2.LINE_AA)

    def _draw_arcane_core(
        self,
        frame,
        glow_layer,
        center: tuple[int, int],
        axes: tuple[int, int],
        ellipse_angle: float,
        color: tuple[int, int, int],
        energy: float,
    ) -> None:
        core_axes = self._scale_axes(axes, 0.22)
        for scale in (1.0, 0.62, 1.38):
            ring_axes = self._scale_axes(core_axes, scale)
            cv2.ellipse(glow_layer, center, ring_axes, ellipse_angle, 0, 360, color, int(self.config.glow_radius * energy) + 6, cv2.LINE_AA)
            cv2.ellipse(frame, center, ring_axes, ellipse_angle, 0, 360, color, 2, cv2.LINE_AA)

        for spoke_index in range(12):
            angle = self._arcane_phase * -1.8 + spoke_index * math.tau / 12.0
            start = self._ellipse_point(center, self._scale_axes(core_axes, 0.35), ellipse_angle, angle)
            end = self._ellipse_point(center, self._scale_axes(core_axes, 1.5), ellipse_angle, angle)
            cv2.line(frame, start, end, color, 1, cv2.LINE_AA)

        cv2.ellipse(glow_layer, center, self._scale_axes(core_axes, 0.36), ellipse_angle, 0, 360, color, -1, cv2.LINE_AA)
        cv2.circle(frame, center, max(3, int(min(core_axes) * 0.18)), (230, 255, 255), -1, cv2.LINE_AA)

    def _draw_arcane_star(
        self,
        frame,
        glow_layer,
        center: tuple[int, int],
        axes: tuple[int, int],
        ellipse_angle: float,
        color: tuple[int, int, int],
    ) -> None:
        triangles = (
            (0.0, 3),
            (math.pi / 3.0, 3),
            (math.pi / 6.0, 6),
        )
        for offset, sides in triangles:
            points = [
                self._ellipse_point(
                    center,
                    self._scale_axes(axes, 0.52 if sides == 3 else 0.74),
                    ellipse_angle,
                    self._arcane_phase * -0.7 + offset + index * math.tau / sides,
                )
                for index in range(sides)
            ]
            for index, start in enumerate(points):
                end = points[(index + 1) % len(points)]
                cv2.line(glow_layer, start, end, color, self.config.glow_radius, cv2.LINE_AA)
                cv2.line(frame, start, end, color, 1, cv2.LINE_AA)

        for index in range(6):
            angle = self._arcane_phase + index * math.tau / 6.0
            start = self._ellipse_point(center, self._scale_axes(axes, 0.28), ellipse_angle, angle)
            end = self._ellipse_point(center, self._scale_axes(axes, 0.78), ellipse_angle, angle)
            cv2.line(frame, start, end, (80, 255, 255), 1, cv2.LINE_AA)

    def _add_arcane_trail(
        self,
        center: tuple[int, int],
        axes: tuple[int, int],
        angle: float,
        energy: float,
    ) -> None:
        self._arcane_trails.append(ArcaneTrail(center, axes, angle, 0.28 * energy, 0.28 * energy))
        self._arcane_trails = self._arcane_trails[-18:]

    def _update_arcane_trails(self, dt: float) -> None:
        for trail in self._arcane_trails:
            trail.life -= dt
        self._arcane_trails = [trail for trail in self._arcane_trails if trail.life > 0]

    def _draw_arcane_trails(self, frame, glow_layer) -> None:
        for trail in self._arcane_trails:
            alpha = trail.life / trail.max_life
            color = self._scale_color((0, 120, 255), 0.45 * alpha)
            cv2.ellipse(glow_layer, trail.center, trail.axes, trail.angle, 0, 360, color, 10, cv2.LINE_AA)
            cv2.ellipse(frame, trail.center, trail.axes, trail.angle, 0, 360, color, 1, cv2.LINE_AA)

    def _spawn_arcane_particles(
        self,
        center: tuple[int, int],
        axes: tuple[int, int],
        ellipse_angle: float,
        radius: int,
        energy: float,
    ) -> None:
        spawn_count = 2 if energy < 1.2 else 4
        for index in range(spawn_count):
            seed = len(self._arcane_particles) + index * 7
            angle = self._arcane_phase * (1.2 + index * 0.15) + seed * 2.399
            position = self._ellipse_point(center, self._scale_axes(axes, 1.18), ellipse_angle, angle)
            speed = radius * (0.45 + 0.12 * (seed % 5)) * energy
            velocity = (math.cos(angle + 0.38) * speed, math.sin(angle + 0.38) * speed)
            color = (0, 120 + seed % 4 * 30, 255)
            self._arcane_particles.append(ArcaneParticle(position, velocity, 0.55, 0.55, 2 + seed % 3, color))

        self._arcane_particles = self._arcane_particles[-120:]

    def _update_arcane_particles(self, dt: float) -> None:
        for particle in self._arcane_particles:
            particle.life -= dt
            particle.position = (
                particle.position[0] + particle.velocity[0] * dt,
                particle.position[1] + particle.velocity[1] * dt,
            )
            particle.velocity = (particle.velocity[0] * 0.96, particle.velocity[1] * 0.96)

        self._arcane_particles = [particle for particle in self._arcane_particles if particle.life > 0]

    def _draw_arcane_particles(self, frame, glow_layer) -> None:
        for particle in self._arcane_particles:
            alpha = max(0.0, particle.life / particle.max_life)
            color = self._scale_color(particle.color, alpha)
            center = (int(particle.position[0]), int(particle.position[1]))
            cv2.circle(glow_layer, center, particle.radius + 7, color, -1, cv2.LINE_AA)
            cv2.circle(frame, center, particle.radius, color, -1, cv2.LINE_AA)

    @staticmethod
    def _palm_center(hand: TrackedHand) -> tuple[int, int]:
        palm_indexes = (0, 5, 9, 13, 17)
        x = sum(hand.point(index)[0] for index in palm_indexes) // len(palm_indexes)
        y = sum(hand.point(index)[1] for index in palm_indexes) // len(palm_indexes)
        return x, y

    @staticmethod
    def _polar_point(center: tuple[int, int], radius: int, angle: float) -> tuple[int, int]:
        return (
            int(center[0] + math.cos(angle) * radius),
            int(center[1] + math.sin(angle) * radius),
        )

    @staticmethod
    def _ellipse_point(
        center: tuple[int, int],
        axes: tuple[int, int],
        ellipse_angle: float,
        local_angle: float,
    ) -> tuple[int, int]:
        rotation = math.radians(ellipse_angle)
        x = math.cos(local_angle) * axes[0]
        y = math.sin(local_angle) * axes[1]
        return (
            int(center[0] + x * math.cos(rotation) - y * math.sin(rotation)),
            int(center[1] + x * math.sin(rotation) + y * math.cos(rotation)),
        )

    @staticmethod
    def _scale_axes(axes: tuple[int, int], scale: float) -> tuple[int, int]:
        return max(1, int(axes[0] * scale)), max(1, int(axes[1] * scale))

    @staticmethod
    def _scale_color(color: tuple[int, int, int], scale: float) -> tuple[int, int, int]:
        return tuple(min(255, max(0, int(channel * scale))) for channel in color)

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

        soft_bloom = cv2.GaussianBlur(
            glow_layer,
            (0, 0),
            sigmaX=self.config.glow_blur_sigma * 2.2,
            sigmaY=self.config.glow_blur_sigma * 2.2,
        )
        core_bloom = cv2.GaussianBlur(
            glow_layer,
            (0, 0),
            sigmaX=self.config.glow_blur_sigma,
            sigmaY=self.config.glow_blur_sigma,
        )
        cv2.addWeighted(soft_bloom, self.config.glow_strength * 0.34, frame, 1.0, 0, dst=frame)
        cv2.addWeighted(core_bloom, self.config.glow_strength * 0.95, frame, 1.0, 0, dst=frame)

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

    def _draw_hud(
        self,
        frame,
        fps: float,
        pinch_active: bool,
        recording_active: bool,
        combine_portal_active: bool,
    ) -> None:
        """Draw minimal runtime information without covering the hand area."""

        mode_name = {
            1: "Neon Web",
            2: "Skeleton Glow",
            3: "Fingertip Lasers",
            4: "Arcane Portal",
        }[self.mode]

        self._draw_translucent_rect(frame, (12, 12), (300, 126), (8, 12, 20), 0.38)
        self._draw_text(frame, f"FPS: {fps:05.1f}", (20, 34), 0.75, (40, 230, 255), 2)
        self._draw_text(frame, f"Mode: {mode_name}", (20, 66), 0.65, (120, 255, 220), 2)
        pinch_text = "Pinch: ACTIVE" if pinch_active else "Pinch: ready"
        pinch_color = (80, 255, 120) if pinch_active else (180, 190, 200)
        self._draw_text(frame, pinch_text, (20, 98), 0.62, pinch_color, 2)
        if self.mode == 4:
            combine_text = "Combine: ON" if combine_portal_active else "Combine: off"
            combine_color = (0, 190, 255) if combine_portal_active else (175, 185, 200)
            self._draw_text(frame, combine_text, (156, 98), 0.55, combine_color, 2)
        self._draw_mode_selector(frame, (22, 116))
        if recording_active:
            self._draw_recording_badge(frame)

        self._draw_text(
            frame,
            "Pinch/M: switch mode  1/2/3/4: mode  B: combine  S/C: screenshot  R: record  Q: quit",
            (20, frame.shape[0] - 24),
            0.58,
            (210, 220, 230),
            2,
        )

    def _draw_mode_selector(self, frame, origin: tuple[int, int]) -> None:
        colors = {
            1: (20, 255, 190),
            2: (255, 190, 80),
            3: (255, 90, 170),
            4: (0, 180, 255),
        }
        for mode_index in range(1, 5):
            center = (origin[0] + (mode_index - 1) * 34, origin[1])
            color = colors[mode_index]
            radius = 9 if self.mode == mode_index else 6
            cv2.circle(frame, center, radius + 4, (14, 18, 28), -1, cv2.LINE_AA)
            cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
            if self.mode == mode_index:
                cv2.circle(frame, center, radius + 4, color, 1, cv2.LINE_AA)

    def _draw_recording_badge(self, frame) -> None:
        x = frame.shape[1] - 124
        self._draw_translucent_rect(frame, (x, 14), (frame.shape[1] - 18, 48), (8, 8, 18), 0.52)
        cv2.circle(frame, (x + 18, 31), 7, (40, 40, 255), -1, cv2.LINE_AA)
        self._draw_text(frame, "REC", (x + 34, 38), 0.55, (70, 80, 255), 2)

    @staticmethod
    def _draw_translucent_rect(
        frame,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
        color: tuple[int, int, int],
        alpha: float,
    ) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)

    @staticmethod
    def _draw_text(
        frame,
        text: str,
        origin: tuple[int, int],
        font_scale: float,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        """Draw readable HUD text with a subtle dark outline."""

        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (20, 24, 32),
            thickness + 3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

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
