"""Dynamic signs: words recognised from a sequence of hand poses plus motion.

A word is a base handshape held through a motion template, read off the
wrist trajectory over the frames:

- HELLO: open hand (B), wrist sweeps side to side (wave)
- THANK-YOU: open hand (B), wrist moves down and away from the body
- YES: fist (S), wrist nods up and down
- NO: index+middle together (U), wrist taps down twice
- PLEASE: open hand (B), wrist circles
- GOOD: open hand (B), wrist moves down from chin height
- I-LOVE-YOU: static handshape, thumb + index + pinky extended

Frames are (landmarks, t_seconds) pairs. Motion is measured from the
wrist only, so it works on any landmark source.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .classifier import classify
from .landmarks import Landmarks
from .poses import generate

WORDS: dict[str, dict] = {
    "HELLO": {"pose": "B", "motion": "wave"},
    "THANK-YOU": {"pose": "B", "motion": "down_out"},
    "YES": {"pose": "S", "motion": "nod"},
    "NO": {"pose": "U", "motion": "tap"},
    "PLEASE": {"pose": "B", "motion": "circle"},
    "GOOD": {"pose": "B", "motion": "down"},
    "I-LOVE-YOU": {"pose": "ILY", "motion": "hold"},
}


def _wrist_path(frames: list[tuple[Landmarks, float]]) -> list[tuple[float, float]]:
    return [landmarks[0] for landmarks, _t in frames]


def _oscillations(values: list[float]) -> int:
    """Count direction changes (peaks) in a 1-D series."""
    if len(values) < 3:
        return 0
    count = 0
    prev = values[1] - values[0]
    for i in range(2, len(values)):
        d = values[i] - values[i - 1]
        if d * prev < 0 and abs(d) > 0.005:
            count += 1
        if d != 0:
            prev = d
    return count


def _motion_signature(frames: list[tuple[Landmarks, float]]) -> str:
    """Name the wrist motion in a frame window, or '' if none is clear."""
    path = _wrist_path(frames)
    if len(path) < 4:
        return "hold"
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    x_osc = _oscillations(xs)
    y_osc = _oscillations(ys)
    total = sum(math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
                for i in range(1, len(path)))
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    # Circle first: long path in both axes, ending near where it started.
    if total > 0.25 and x_range > 0.05 and y_range > 0.05 and abs(dx) < 0.12 and abs(dy) < 0.12:
        return "circle"
    if x_osc >= 2 and abs(dx) < 0.10:
        return "wave"
    # Tap: repeated downward pulses that always return to the baseline
    # (y never goes above where it started); nod swings both ways.
    if y_osc >= 2 and min(ys) >= ys[0] - 0.01 and y_range > 0.03:
        return "tap"
    if y_osc >= 2 and abs(dy) < 0.10:
        return "nod"
    if dy > 0.08 and abs(dx) < 0.06:
        return "down"
    if dy > 0.05 and dx > 0.05:
        return "down_out"
    if total < 0.03:
        return "hold"
    return ""


@dataclass(frozen=True)
class WordResult:
    word: str
    confidence: float
    pose_letter: str
    motion: str


def recognise_word(frames: list[tuple[Landmarks, float]]) -> WordResult | None:
    """Recognise one word from a window of frames, or None."""
    if len(frames) < 4:
        return None
    letters = [classify(landmarks).letter for landmarks, _t in frames]
    pose_letter = max(set(letters), key=letters.count)
    if letters.count(pose_letter) < len(letters) * 0.6:
        return None
    motion = _motion_signature(frames)
    for word, spec in WORDS.items():
        if spec["pose"] == pose_letter and spec["motion"] == motion:
            return WordResult(word, 0.90, pose_letter, motion)
    return None


def _move_wrist(landmarks: Landmarks, dx: float, dy: float) -> Landmarks:
    return [(x + dx, y + dy) for x, y in landmarks]


def generate_word_frames(word: str, seed: int = 0, frames: int = 12) -> list[tuple[Landmarks, float]]:
    """Synthetic frame sequence for a word: base pose + wrist motion."""
    word = word.upper()
    if word not in WORDS:
        raise ValueError(f"no word template for {word!r}")
    spec = WORDS[word]
    out: list[tuple[Landmarks, float]] = []
    if spec["pose"] == "ILY":
        base = _ily_landmarks()
    else:
        base = generate(spec["pose"], seed=seed)
    motion = spec["motion"]
    for i in range(frames):
        phase = i / max(frames - 1, 1)
        if motion == "wave":
            dx, dy = 0.08 * math.sin(phase * math.pi * 2 * 1.5), 0.0
        elif motion == "nod":
            dx, dy = 0.0, 0.05 * math.sin(phase * math.pi * 2 * 1.5)
        elif motion == "tap":
            dx, dy = 0.0, 0.05 * abs(math.sin(phase * math.pi * 2))
        elif motion == "circle":
            dx, dy = 0.06 * math.sin(phase * math.pi * 2), 0.06 * math.cos(phase * math.pi * 2)
        elif motion == "down":
            dx, dy = 0.0, 0.12 * phase
        elif motion == "down_out":
            dx, dy = 0.08 * phase, 0.10 * phase
        else:  # hold
            dx, dy = 0.0, 0.0
        out.append((_move_wrist(base, dx, dy), phase * 1.2))
    return out


def _ily_landmarks() -> Landmarks:
    """I-Love-You handshape: thumb, index and pinky extended.

    Built from the Y pose (thumb + pinky) with the index extended too.
    """
    from .landmarks import FINGERS
    from .poses import POSES, _finger_landmarks, _thumb_landmarks, WRIST
    spec = POSES["Y"]
    points: Landmarks = [WRIST]
    reach = dict(spec.reach)
    reach["index"] = 1.0
    for finger in FINGERS:
        joints = _finger_landmarks(finger, reach[finger], 0.0, random.Random(0))
        points.extend(joints)
    points.extend(_thumb_landmarks((0.26, 0.60)))
    ordered: Landmarks = [points[0]] + points[17:21]
    for finger in FINGERS:
        ordered.extend(points[1 + FINGERS.index(finger) * 4: 5 + FINGERS.index(finger) * 4])
    return ordered
