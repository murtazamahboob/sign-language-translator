import math

import pytest

from signlang.landmarks import (
    extract_features,
    finger_reach,
    validate_landmarks,
)
from signlang.poses import ALPHABET, generate, generate_alphabet


def test_validate_rejects_bad_input():
    with pytest.raises(ValueError):
        validate_landmarks(None)
    with pytest.raises(ValueError):
        validate_landmarks([(0.1, 0.2)] * 20)
    with pytest.raises(ValueError):
        validate_landmarks([(float("nan"), 0.1)] * 21)


def test_alphabet_has_26_letters():
    assert len(ALPHABET) == 26
    assert ALPHABET[0] == "A" and ALPHABET[-1] == "Z"


def test_generate_is_deterministic():
    assert generate("B", seed=3) == generate("B", seed=3)
    assert generate("B", seed=3, jitter=0.01) != generate("B", seed=4, jitter=0.01)


def test_generate_has_21_landmarks_in_range():
    for letter, points in generate_alphabet().items():
        assert len(points) == 21, letter
        for x, y in points:
            assert math.isfinite(x) and math.isfinite(y), letter


def test_reach_extended_vs_curled():
    open_hand = generate("B")
    fist = generate("A")
    assert finger_reach(open_hand, "index") > 0.9
    assert finger_reach(fist, "index") < 0.3
    assert finger_reach(generate("C"), "index") == pytest.approx(0.6, abs=0.15)


def test_thumb_modes_differ_across_fist_family():
    modes = {letter: extract_features(generate(letter)).thumb_mode
             for letter in "ASTMN"}
    assert len(set(modes.values())) == 5, modes


def test_orientation_detected():
    assert extract_features(generate("G")).direction == "side"
    assert extract_features(generate("H")).direction == "side"
    assert extract_features(generate("P")).direction == "down"
    assert extract_features(generate("Q")).direction == "down"
    assert extract_features(generate("B")).direction == "up"


def test_jitter_keeps_landmarks_finite():
    points = generate("V", seed=1, jitter=0.01)
    assert len(points) == 21
    assert all(math.isfinite(x) and math.isfinite(y) for x, y in points)
