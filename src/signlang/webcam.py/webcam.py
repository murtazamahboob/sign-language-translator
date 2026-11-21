"""Webcam adapter (optional dependency).

OpenCV is *not* bundled and not required: the translator works on
landmarks from any source. This module only wires a camera to a
``LandmarkProvider`` when both are available:

- ``cv2`` must be installed (``pip install opencv-python``).
- A provider must supply the 21 landmarks per frame — e.g. MediaPipe
  Hands (``mediapipe``), which this project deliberately does not
  bundle, so the core package stays dependency-free and offline.

Without a provider, use :class:`signlang.words` synthetic frames or
feed landmark JSON/JSONL files through the CLI instead. The adapter
never guesses landmarks from raw pixels.
"""

from __future__ import annotations

from typing import Protocol

from .landmarks import Landmarks


class LandmarkProvider(Protocol):
    """Turns one video frame (BGR numpy array) into 21 landmarks or None."""

    def __call__(self, frame) -> Landmarks | None: ...


class WebcamTranslator:
    """Reads camera frames, translates them through a provider + stream."""

    def __init__(self, provider: LandmarkProvider, camera_index: int = 0):
        self.provider = provider
        self.camera_index = camera_index

    def run(self, max_frames: int = 300, show: bool = False) -> str:
        """Translate a live session; return the sentence when it ends."""
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "OpenCV is not installed: pip install opencv-python"
            ) from exc
        from .stream import SignStream

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera {self.camera_index}")
        stream = SignStream()
        try:
            for _ in range(max_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                landmarks = self.provider(frame)
                events = stream.feed(landmarks)
                for event in events:
                    print(f"[{event.t:.1f}s] {event.kind}: {event.text}")
                if show:
                    cv2.imshow("sign-language-translator (q to quit)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if show:
                cv2.destroyAllWindows()
        return stream.finish()
