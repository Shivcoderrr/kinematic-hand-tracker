"""MediaPipe-based hand tracking module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import mediapipe as mp

from config import TrackerConfig


@dataclass(frozen=True)
class Landmark:
    """A single hand landmark converted from normalized MediaPipe coordinates."""

    index: int
    x: int
    y: int
    z: float


@dataclass(frozen=True)
class TrackedHand:
    """A detected hand and its 21 landmarks."""

    handedness: str
    landmarks: tuple[Landmark, ...]

    def point(self, index: int) -> tuple[int, int]:
        """Return a landmark as a simple OpenCV-friendly (x, y) tuple."""

        landmark = self.landmarks[index]
        return landmark.x, landmark.y


class HandTracker:
    """Owns MediaPipe setup and converts raw results into clean Python objects."""

    def __init__(self, config: TrackerConfig) -> None:
        self._config = config
        self._hands_module = mp.solutions.hands
        self._hands = self._hands_module.Hands(
            static_image_mode=False,
            max_num_hands=config.max_num_hands,
            model_complexity=config.model_complexity,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )

    def detect(self, frame_bgr) -> list[TrackedHand]:
        """Detect hands in a BGR OpenCV frame.

        MediaPipe expects RGB frames, while OpenCV captures BGR frames. This
        conversion is the bridge between the two libraries.
        """

        height, width = frame_bgr.shape[:2]
        inference_frame = self._resize_for_inference(frame_bgr)

        frame_rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return []

        labels = self._extract_handedness(results.multi_handedness)

        tracked_hands: list[TrackedHand] = []
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = tuple(
                Landmark(
                    index=index,
                    x=int(landmark.x * width),
                    y=int(landmark.y * height),
                    z=landmark.z,
                )
                for index, landmark in enumerate(hand_landmarks.landmark)
            )

            tracked_hands.append(
                TrackedHand(
                    handedness=labels[hand_index] if hand_index < len(labels) else "Unknown",
                    landmarks=landmarks,
                )
            )

        return tracked_hands

    def close(self) -> None:
        """Release native resources held by MediaPipe."""

        self._hands.close()

    def _resize_for_inference(self, frame_bgr):
        """Use a smaller frame for ML inference to keep real-time FPS stable.

        MediaPipe returns normalized landmark positions, so we can safely run
        detection on a resized copy and still map landmarks back to the original
        display frame by multiplying by the display width and height.
        """

        height, width = frame_bgr.shape[:2]
        if width <= self._config.processing_width:
            return frame_bgr

        scale = self._config.processing_width / width
        processing_size = (self._config.processing_width, int(height * scale))
        return cv2.resize(frame_bgr, processing_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _extract_handedness(raw_handedness: Iterable | None) -> list[str]:
        if not raw_handedness:
            return []

        return [
            handedness.classification[0].label
            for handedness in raw_handedness
            if handedness.classification
        ]
