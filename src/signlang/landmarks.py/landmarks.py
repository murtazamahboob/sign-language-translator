"""Hand landmarks: the 21-point hand model and geometric features.

Landmark order follows the MediaPipe Hands convention so the classifier
works on landmarks from any compatible detector:

    0 wrist
    1-4   thumb  (CMC, MCP, IP, tip)
    5-8   index  (MCP, PIP, DIP, tip)
    9-12  middle (MCP, PIP, DIP, tip)
    13-16 ring   (MCP, PIP, DIP, tip)
    17-20 pinky  (MCP, PIP, DIP, tip)

Coordinates are normalised to the image frame (x right, y down, 0..1).
Only the standard library is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Landmark = tuple[float, float]
Landmarks = list[Landmark]

FINGERS = ("index", "middle", "ring", "pinky")
# MCP, PIP, DIP, tip indices per finger.
FINGER_JOINTS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
THUMB_JOINTS = (1, 2, 3, 4)
WRIST = 0

# Nominal fingertip-to-MCP distance for a fully extended finger, in
# normalised units. Used to turn a raw distance into a 0..1 reach.
NOMINAL_LENGTH = {"index": 0.30, "middle": 0.32, "ring": 0.29, "pinky": 0.23}


def dist(a: Landmark, b: Landmark) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def validate_landmarks(landmarks) -> Landmarks:
    """Return a clean list of 21 (x, y) float pairs or raise ValueError."""
    if landmarks is None:
        raise ValueError("no landmarks supplied")
    points = [(float(x), float(y)) for x, y in landmarks]
    if len(points) != 21:
        raise ValueError(f"expected 21 landmarks, got {len(points)}")
    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("landmark coordinates must be finite numbers")
    return points


def finger_reach(landmarks: Landmarks, finger: str) -> float:
    """How extended a finger is: tip-to-MCP distance over nominal length.

    ~1.0 fully extended, ~0.6 half-curled (C, O), ~0.2 tightly curled
    (fist). Orientation-invariant because it uses distance, not direction.
    """
    mcp, _pip, _dip, tip = FINGER_JOINTS[finger]
    return dist(landmarks[mcp], landmarks[tip]) / NOMINAL_LENGTH[finger]


def finger_direction(landmarks: Landmarks, finger: str) -> Landmark:
    """Unit vector from a finger's MCP to its tip."""
    mcp, _pip, _dip, tip = FINGER_JOINTS[finger]
    dx = landmarks[tip][0] - landmarks[mcp][0]
    dy = landmarks[tip][1] - landmarks[mcp][1]
    length = math.hypot(dx, dy) or 1.0
    return (dx / length, dy / length)


def palm_centre(landmarks: Landmarks) -> Landmark:
    xs = [landmarks[i][0] for i in (0, 5, 9, 13, 17)]
    ys = [landmarks[i][1] for i in (0, 5, 9, 13, 17)]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def hand_scale(landmarks: Landmarks) -> float:
    """Wrist-to-middle-MCP distance: the hand's own size unit."""
    return dist(landmarks[0], landmarks[9]) or 1.0


@dataclass(frozen=True)
class HandFeatures:
    reach: dict[str, float]
    thumb_tip: Landmark
    thumb_mode: str
    direction: str  # "up" | "side" | "down": where extended fingers point
    index_middle_spread: float  # tip x gap, index minus middle
    index_middle_crossed: bool
    extended: tuple[str, ...]

    def is_extended(self, finger: str) -> bool:
        return self.reach[finger] >= 0.78

    def is_curled(self, finger: str) -> bool:
        return self.reach[finger] <= 0.45


def classify_thumb_mode(landmarks: Landmarks) -> str:
    """Where the thumb tip sits, from landmark geometry only.

    Modes: up, between, side, down, across, across_low, tucked,
    tucked_wide, front, touch_index, touch_middle.
    """
    tip = landmarks[4]
    index_tip = landmarks[8]
    middle_pip = landmarks[10]
    # Touch modes only count when the index finger is actually reaching
    # out to meet the thumb (D, F, O). In a fist the curled index tip
    # sits next to the thumb by accident and must not trigger them.
    index_reach = finger_reach(landmarks, "index")
    if index_reach > 0.30 and dist(tip, index_tip) < 0.08:
        return "touch_index"
    if index_reach > 0.30 and dist(tip, middle_pip) < 0.08:
        return "touch_middle"
    x, y = tip
    if x < 0.33:
        return "side"
    if y > 0.82:
        return "down"
    if y < 0.55 and 0.34 <= x < 0.44:
        return "up"
    if y < 0.55 and 0.44 <= x <= 0.51:
        return "between"
    if 0.60 <= y < 0.68 and x > 0.50:
        return "across"
    if 0.64 <= y < 0.74 and 0.47 <= x <= 0.52:
        return "tucked_wide"
    if 0.64 <= y < 0.74 and 0.42 <= x < 0.47:
        return "tucked"
    if 0.68 <= y < 0.76 and x > 0.53:
        return "across_low"
    if 0.56 <= y < 0.64 and 0.46 <= x <= 0.54:
        return "front"
    # Nearest fallback: closest named zone centre.
    zones = {
        "up": (0.39, 0.48),
        "between": (0.47, 0.50),
        "side": (0.26, 0.60),
        "down": (0.36, 0.88),
        "across": (0.54, 0.64),
        "across_low": (0.55, 0.72),
        "tucked": (0.44, 0.69),
        "tucked_wide": (0.49, 0.69),
        "front": (0.50, 0.60),
    }
    return min(zones, key=lambda name: dist(tip, zones[name]))


def _rotate(point: Landmark, angle_deg: float, centre: Landmark) -> Landmark:
    angle = math.radians(angle_deg)
    x, y = point[0] - centre[0], point[1] - centre[1]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return (centre[0] + x * cos_a - y * sin_a,
            centre[1] + x * sin_a + y * cos_a)


def _canonicalise(points: Landmarks) -> Landmarks:
    """Translate/scale the hand into a canonical frame: wrist at
    (0.50, 0.85), wrist-to-middle-MCP distance 0.27.

    Real detectors frame the hand differently and rotated poses move
    landmarks outside the unit square; thumb zones and reaches are
    only meaningful in this canonical frame.
    """
    wrist = points[0]
    scale = 0.27 / hand_scale(points)
    return [(0.50 + (x - wrist[0]) * scale, 0.85 + (y - wrist[1]) * scale)
            for x, y in points]


def extract_features(landmarks: Landmarks) -> HandFeatures:
    points = validate_landmarks(landmarks)
    extended = tuple(f for f in FINGERS
                     if finger_reach(_canonicalise(points), f) >= 0.78)
    # Orientation from the extended fingers' mean direction, then rotate
    # the hand back to the upright frame so thumb zones, spread and
    # crossing are read in hand-local coordinates (P/Q point down,
    # G/H point sideways, but their thumb means the same as upright).
    direction = "up"
    if extended:
        dx = sum(finger_direction(points, f)[0] for f in extended) / len(extended)
        dy = sum(finger_direction(points, f)[1] for f in extended) / len(extended)
        if dy > 0.45:
            direction = "down"
        elif dx > 0.45:
            direction = "side"
    upright = _canonicalise(points)
    if direction == "down":
        upright = [_rotate(p, 180.0, (0.50, 0.85)) for p in upright]
    elif direction == "side":
        upright = [_rotate(p, -90.0, (0.50, 0.85)) for p in upright]
    reach = {f: finger_reach(upright, f) for f in FINGERS}
    thumb_mode = classify_thumb_mode(upright)
    spread = upright[12][0] - upright[8][0]  # middle tip minus index tip
    crossed = upright[8][0] > upright[12][0] + 0.01
    return HandFeatures(
        reach=reach,
        thumb_tip=upright[4],
        thumb_mode=thumb_mode,
        direction=direction,
        index_middle_spread=spread,
        index_middle_crossed=crossed,
        extended=extended,
    )
