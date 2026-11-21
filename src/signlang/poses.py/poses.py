"""Synthetic hand-pose generator for the ASL fingerspelling alphabet.

Each letter is a parametric pose: per-finger reach (0 = tightly curled,
1 = fully extended), a thumb mode, an orientation and optional spread,
crossing or motion. Landmarks are built from those parameters with
seeded jitter, so the bundled dataset is deterministic and reproducible.

This is a *test and demo* data source, not a claim about real hands: it
lets the rule-based classifier be benchmarked offline without a camera
or a labelled dataset. The classifier itself only ever sees landmarks.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .landmarks import FINGERS, Landmarks

# Finger MCP positions (x, y) in normalised coordinates, palm facing camera.
MCP = {
    "index": (0.42, 0.60),
    "middle": (0.50, 0.58),
    "ring": (0.58, 0.60),
    "pinky": (0.65, 0.63),
}
FINGER_LENGTH = {"index": 0.30, "middle": 0.32, "ring": 0.29, "pinky": 0.23}
# Natural fan: how far each extended fingertip sits from its MCP in x.
FAN = {"index": -0.02, "middle": 0.0, "ring": 0.02, "pinky": 0.04}
WRIST = (0.50, 0.85)

# Thumb-tip targets per mode. These are the same zones the classifier
# in landmarks.py recognises.
THUMB_TIP = {
    "up": (0.39, 0.48),
    "between": (0.47, 0.50),
    "side": (0.26, 0.60),
    "down": (0.36, 0.88),
    "across": (0.54, 0.64),
    "across_low": (0.55, 0.72),
    "tucked": (0.44, 0.69),
    "tucked_wide": (0.49, 0.69),
    "front": (0.50, 0.60),
    "touch_middle": (0.51, 0.60),
}
THUMB_BASE = [(0.34, 0.74), (0.36, 0.68)]  # CMC, MCP


@dataclass(frozen=True)
class PoseSpec:
    reach: dict[str, float]
    thumb: str
    orientation: str = "up"  # up | side | down
    spread: float = 0.0  # extra index/middle tip separation (V, W)
    cross: bool = False  # index tip crosses over middle tip (R)
    motion: str = ""  # e.g. "j-hook", "z-trace" for dynamic letters


def _r(reach: dict[str, float], **over: float) -> dict[str, float]:
    spec = dict(reach)
    spec.update(over)
    return spec


_FIST = {"index": 0.15, "middle": 0.15, "ring": 0.15, "pinky": 0.15}
_OPEN = {"index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0}

# ASL fingerspelling poses, from the standard handshapes.
POSES: dict[str, PoseSpec] = {
    "A": PoseSpec(_FIST, "up"),
    "B": PoseSpec(_OPEN, "across"),
    "C": PoseSpec({f: 0.60 for f in FINGERS}, "across"),
    "D": PoseSpec(_r(_FIST, index=1.0), "touch_middle"),
    "E": PoseSpec({f: 0.35 for f in FINGERS}, "across_low"),
    "F": PoseSpec(_r(_OPEN, index=0.60), "touch_index"),
    "G": PoseSpec(_r(_FIST, index=1.0), "across_low", orientation="side"),
    "H": PoseSpec(_r(_FIST, index=1.0, middle=1.0), "across", orientation="side"),
    "I": PoseSpec(_r(_FIST, pinky=1.0), "across"),
    "J": PoseSpec(_r(_FIST, pinky=1.0), "across", motion="j-hook"),
    "K": PoseSpec(_r(_FIST, index=1.0, middle=1.0), "up", spread=0.02),
    "L": PoseSpec(_r(_FIST, index=1.0), "side"),
    "M": PoseSpec(_FIST, "tucked_wide"),
    "N": PoseSpec(_FIST, "tucked"),
    "O": PoseSpec({f: 0.48 for f in FINGERS}, "touch_index"),
    "P": PoseSpec(_r(_FIST, index=1.0, middle=1.0), "up", orientation="down", spread=0.02),
    "Q": PoseSpec(_r(_FIST, index=1.0), "down", orientation="down"),
    "R": PoseSpec(_r(_FIST, index=1.0, middle=1.0), "across", cross=True),
    "S": PoseSpec(_FIST, "front"),
    "T": PoseSpec(_FIST, "between"),
    "U": PoseSpec(_r(_FIST, index=1.0, middle=1.0), "across"),
    "V": PoseSpec(_r(_FIST, index=1.0, middle=1.0), "across", spread=0.10),
    "W": PoseSpec(_r(_FIST, index=1.0, middle=1.0, ring=1.0), "across", spread=0.08),
    "X": PoseSpec(_r(_FIST, index=0.55), "across_low"),
    "Y": PoseSpec(_r(_FIST, pinky=1.0), "side"),
    "Z": PoseSpec(_r(_FIST, index=1.0), "across_low", motion="z-trace"),
}

ALPHABET = sorted(POSES)


def _finger_landmarks(finger: str, reach: float, spread: float, rng: random.Random) -> list[tuple[float, float]]:
    """MCP, PIP, DIP, tip for one finger at the given reach."""
    mcp = MCP[finger]
    length = FINGER_LENGTH[finger]
    # Extended tip: straight up from the MCP with the natural fan.
    ext_tip = (mcp[0] + FAN[finger] + spread, mcp[1] - length)
    # Curled tip: folded down toward the palm, slightly toward the middle.
    curl_tip = (mcp[0] + (0.50 - mcp[0]) * 0.25, mcp[1] + 0.06)
    tip = (
        curl_tip[0] + (ext_tip[0] - curl_tip[0]) * reach,
        curl_tip[1] + (ext_tip[1] - curl_tip[1]) * reach,
    )
    # Joints lie along the MCP->tip line with a bend that grows as the
    # finger curls (a curled finger's joints arc toward the palm).
    bend = (1.0 - reach) * 0.05
    joints = [mcp]
    for t in (0.35, 0.70):
        joints.append((
            mcp[0] + (tip[0] - mcp[0]) * t + bend * t,
            mcp[1] + (tip[1] - mcp[1]) * t + bend * 0.5,
        ))
    joints.append(tip)
    return joints


def _thumb_landmarks(tip_target: tuple[float, float]) -> list[tuple[float, float]]:
    joints = list(THUMB_BASE)
    # IP sits two-thirds of the way from MCP to tip; tip is the target.
    mcp = THUMB_BASE[1]
    joints.append((mcp[0] + (tip_target[0] - mcp[0]) * 0.66,
                   mcp[1] + (tip_target[1] - mcp[1]) * 0.66))
    joints.append(tip_target)
    return joints


def _rotate(point: tuple[float, float], angle_deg: float) -> tuple[float, float]:
    """Rotate a point around the wrist. 0 = up, -90 = pointing right."""
    angle = math.radians(angle_deg)
    x, y = point[0] - WRIST[0], point[1] - WRIST[1]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return (WRIST[0] + x * cos_a - y * sin_a,
            WRIST[1] + x * sin_a + y * cos_a)


def generate(letter: str, seed: int = 0, jitter: float = 0.0) -> Landmarks:
    """Deterministic landmarks for one letter.

    `jitter` adds seeded Gaussian noise (in normalised units) to every
    landmark, simulating detector noise. Same seed -> same landmarks.
    """
    letter = letter.upper()
    if letter not in POSES:
        raise ValueError(f"no pose for letter {letter!r}")
    spec = POSES[letter]
    rng = random.Random(seed)
    points: Landmarks = [WRIST]
    # Spread fans index and middle apart (index left, middle right);
    # crossing moves the index tip over the middle tip (R).
    spread = {"index": -spec.spread, "middle": spec.spread,
              "ring": 0.0, "pinky": 0.0}
    if spec.cross:
        spread["index"] = 0.10
        spread["middle"] = -0.10
    finger_tips: dict[str, tuple[float, float]] = {}
    for finger in FINGERS:
        joints = _finger_landmarks(finger, spec.reach[finger], spread[finger], rng)
        finger_tips[finger] = joints[-1]
        points.extend(joints)
    # Thumb target may depend on the index fingertip (F, O: thumb meets it).
    if spec.thumb == "touch_index":
        tip_target = finger_tips["index"]
    else:
        tip_target = THUMB_TIP[spec.thumb]
    points.extend(_thumb_landmarks(tip_target))
    # Reorder: our build order is wrist, fingers, thumb, but the model
    # order is wrist, thumb (1-4), fingers (5-20).
    ordered: Landmarks = [points[0]] + points[17:21]
    for finger in FINGERS:
        ordered.extend(points[1 + FINGERS.index(finger) * 4: 5 + FINGERS.index(finger) * 4])
    angle = {"up": 0.0, "side": 90.0, "down": 180.0}[spec.orientation]
    if angle:
        ordered = [_rotate(p, angle) for p in ordered]
    if jitter:
        ordered = [(x + rng.gauss(0, jitter), y + rng.gauss(0, jitter))
                   for x, y in ordered]
    return ordered


def generate_alphabet(seed: int = 0, jitter: float = 0.0) -> dict[str, Landmarks]:
    return {letter: generate(letter, seed=seed, jitter=jitter) for letter in ALPHABET}


def motion_of(letter: str) -> str:
    return POSES[letter.upper()].motion
