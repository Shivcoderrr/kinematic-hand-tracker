"""Gesture and distance helpers used by the visual layer."""

from __future__ import annotations

import math

from hand_tracker import TrackedHand


FINGER_TIPS = (4, 8, 12, 16, 20)
THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0
MIDDLE_FINGER_MCP = 9


class GestureAnalyzer:
    """Small math helpers that keep interaction logic separate from drawing."""

    @staticmethod
    def distance(point_a: tuple[int, int], point_b: tuple[int, int]) -> float:
        """Euclidean distance between two 2D points."""

        return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])

    def fingertip_points(self, hand: TrackedHand) -> list[tuple[int, int]]:
        """Return the five fingertip points for one hand."""

        return [hand.point(index) for index in FINGER_TIPS]

    def openness_score(self, hand: TrackedHand) -> float:
        """Estimate how open the hand is using fingertip distance from the wrist.

        This is not a full gesture classifier. It is a lightweight signal that
        can drive visuals, such as glow strength or color.
        """

        wrist = hand.point(0)
        distances = [self.distance(wrist, tip) for tip in self.fingertip_points(hand)]
        return sum(distances) / len(distances)

    def pinch_distance(self, hand: TrackedHand) -> float:
        """Distance between thumb tip and index fingertip."""

        return self.distance(hand.point(THUMB_TIP), hand.point(INDEX_TIP))

    def hand_scale(self, hand: TrackedHand) -> float:
        """Approximate hand size from wrist to middle-finger knuckle."""

        return self.distance(hand.point(WRIST), hand.point(MIDDLE_FINGER_MCP))

    def is_pinching(
        self,
        hand: TrackedHand,
        close_ratio: float = 0.26,
        min_close_pixels: float = 18.0,
    ) -> bool:
        """Detect a thumb-index pinch using a hand-size-relative threshold.

        A fixed pixel threshold feels different when the hand is close to the
        camera versus far away. The wrist-to-middle-knuckle distance gives us a
        rough hand scale, so the pinch stays usable at different distances.
        """

        hand_scale = self.hand_scale(hand)
        threshold = max(min_close_pixels, hand_scale * close_ratio)
        return self.pinch_distance(hand) <= threshold
