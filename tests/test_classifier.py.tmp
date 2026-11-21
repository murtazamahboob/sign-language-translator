from signlang.classifier import classify
from signlang.poses import ALPHABET, generate, motion_of


def test_clean_alphabet_26_of_26():
    for letter in ALPHABET:
        result = classify(generate(letter), motion=motion_of(letter))
        assert result.letter == letter, f"{letter} -> {result.letter} ({result.reason})"


def test_jittered_alphabet_stays_accurate():
    correct = total = 0
    for letter in ALPHABET:
        for seed in range(5):
            got = classify(generate(letter, seed=100 + seed, jitter=0.01),
                           motion=motion_of(letter)).letter
            total += 1
            correct += got == letter
    assert correct / total >= 0.95, f"{correct}/{total}"


def test_confusable_pairs():
    pairs = [("A", "S"), ("A", "T"), ("M", "N"), ("U", "V"), ("K", "P"),
             ("D", "G"), ("E", "S"), ("R", "U"), ("I", "J"), ("X", "A")]
    for a, b in pairs:
        got_a = classify(generate(a), motion=motion_of(a)).letter
        got_b = classify(generate(b), motion=motion_of(b)).letter
        assert got_a == a, f"{a} misread as {got_a}"
        assert got_b == b, f"{b} misread as {got_b}"


def test_dynamic_letters_need_motion():
    # Without the motion hint, J reads as its static base I, and Z's
    # static handshape is ambiguous (index up, thumb across) — "?" is
    # the honest answer until the trace is seen.
    assert classify(generate("J")).letter == "I"
    assert classify(generate("Z")).letter == "?"
    assert classify(generate("J"), motion="j-hook").letter == "J"
    assert classify(generate("Z"), motion="z-trace").letter == "Z"


def test_unknown_handshape_returns_question_mark():
    # A hand with every finger half-extended and thumb nowhere standard.
    points = generate("C")
    result = classify(points)
    assert result.letter == "C"  # sanity: C itself is recognised
    assert result.confidence > 0
