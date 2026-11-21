"""Rule-based ASL fingerspelling classifier over hand landmarks.

The decision tree reads only the geometric features extracted in
landmarks.py: per-finger reach, thumb mode, hand orientation and the
index/middle relationship. No model, no training data, no API key.
Letters that differ only by motion (J = I + hook, Z = index + trace)
need a motion hint; without one they read as their static base letter.
"""

from __future__ import annotations

from dataclasses import dataclass

from .landmarks import HandFeatures, Landmarks, extract_features


@dataclass(frozen=True)
class Classification:
    letter: str
    confidence: float  # 0..1, from how cleanly the features matched
    features: HandFeatures
    reason: str


def classify(landmarks: Landmarks, motion: str = "") -> Classification:
    features = extract_features(landmarks)
    reach = features.reach
    thumb = features.thumb_mode
    extended = features.extended
    n_ext = len(extended)

    def done(letter: str, reason: str, confidence: float = 0.95) -> Classification:
        return Classification(letter, confidence, features, reason)

    # Dynamic letters first: same handshape as a static letter + motion.
    if motion == "j-hook" and extended == ("pinky",):
        return done("J", "pinky extended with a J-hook motion")
    if motion == "z-trace" and extended == ("index",):
        return done("Z", "index extended with a Z-trace motion")

    # Thumb meets the index fingertip: F (three fingers up) or O (curled).
    if thumb == "touch_index":
        if n_ext >= 3:
            return done("F", "index meets thumb, other fingers extended")
        return done("O", "fingertips meet in a circle", 0.90)

    # Sideways handshapes: G (index only) and H (index + middle).
    if features.direction == "side":
        if extended == ("index", "middle"):
            return done("H", "index and middle pointing sideways")
        if extended == ("index",):
            return done("G", "index pointing sideways")
        return done("?", f"unrecognised sideways handshape ({extended})", 0.30)

    # Downward handshapes: P (K-shape down) and Q (G-shape down).
    if features.direction == "down":
        if set(extended) == {"index", "middle"}:
            return done("P", "index and middle pointing down, thumb up")
        if extended == ("index",):
            return done("Q", "index pointing down, thumb down")
        return done("?", f"unrecognised downward handshape ({extended})", 0.30)

    # Four fingers extended: B, unless the hand is half-curled (C).
    if n_ext == 4:
        return done("B", "all fingers extended, thumb across palm")
    if all(0.45 <= reach[f] <= 0.70 for f in ("index", "middle", "ring", "pinky")):
        return done("C", "all fingers half-curled into a C")

    # Three fingers extended: W.
    if set(extended) == {"index", "middle", "ring"}:
        return done("W", "index, middle and ring extended")

    # I-Love-You handshape (a word sign, not a letter): thumb, index
    # and pinky extended.
    if set(extended) == {"index", "pinky"} and thumb == "side":
        return done("ILY", "thumb, index and pinky extended")

    # Two fingers extended: K, R, U, V.
    if set(extended) == {"index", "middle"}:
        if features.index_middle_crossed:
            return done("R", "index crossed over middle")
        if thumb == "up":
            return done("K", "index and middle up, thumb between them")
        if features.index_middle_spread > 0.20:
            return done("V", "index and middle spread apart")
        return done("U", "index and middle together")

    # One finger extended.
    if n_ext == 1:
        only = extended[0]
        if only == "index":
            if thumb == "touch_middle":
                return done("D", "index up, thumb touching middle finger")
            if thumb == "side":
                return done("L", "index up, thumb out to the side")
            if 0.45 <= reach["index"] <= 0.70:
                return done("X", "index hooked")
            return done("?", "index up but thumb position unclear", 0.40)
        if only == "pinky":
            if thumb == "side":
                return done("Y", "thumb and pinky out")
            return done("I", "pinky up, thumb across palm")
        return done("?", f"only {only} extended", 0.30)

    # Hooked index (X): index half-extended, everything else curled.
    # It never reaches the "extended" threshold, so it needs its own
    # branch before the fist family below.
    if (n_ext == 0 and 0.40 <= reach["index"] <= 0.70
            and all(reach[f] <= 0.30 for f in ("middle", "ring", "pinky"))):
        return done("X", "index hooked")

    # No finger extended: fist family, told apart by the thumb.
    if n_ext == 0:
        mean_reach = sum(reach.values()) / 4
        if thumb == "across_low" and mean_reach > 0.15:
            return done("E", "fingers curled tight, thumb low across")
        if thumb == "up":
            return done("A", "fist with thumb up alongside")
        if thumb == "front":
            return done("S", "fist with thumb across the front")
        if thumb == "between":
            return done("T", "fist with thumb poking up between fingers")
        if thumb == "tucked_wide":
            return done("M", "fist with thumb tucked under three fingers")
        if thumb == "tucked":
            return done("N", "fist with thumb tucked under two fingers")
        if thumb in ("across", "across_low"):
            return done("S", "fist with thumb across the fingers", 0.80)
        return done("?", f"fist with unclear thumb ({thumb})", 0.30)

    return done("?", f"unrecognised handshape ({extended}, thumb {thumb})", 0.30)


def classify_letter(landmarks: Landmarks, motion: str = "") -> str:
    return classify(landmarks, motion).letter
