from signlang.poses import generate
from signlang.stream import SignStream
from signlang.words import generate_word_frames


def _feed_letters(stream, text, hold=3, letter_gap=0.4, word_gap=2.0):
    t = 0.0
    for letter in text:
        for _ in range(hold):
            stream.feed(generate(letter), t=t)
            t += 0.1
        # Short pause between letters (no word break), long pause after
        # the word (a real word break).
        t += letter_gap
        stream.feed(None, t=t)
    t += word_gap
    stream.feed(None, t=t)
    return stream.finish()


def test_fingerspells_a_word():
    assert _feed_letters(SignStream(), "CAT") == "CAT"


def test_two_words_joined_with_space():
    stream = SignStream()
    _feed_letters(stream, "CAT")
    assert _feed_letters(stream, "DOG") == "CAT DOG"


def test_flicker_below_hold_is_ignored():
    stream = SignStream(hold_frames=3)
    t = 0.0
    for letter in ["C", "A", "C", "A", "C"]:
        stream.feed(generate(letter), t=t)
        t += 0.1
    stream.feed(None, t=5.0)
    assert stream.finish() == ""


def test_word_sign_beats_fingerspelling():
    stream = SignStream()
    for landmarks, t in generate_word_frames("HELLO"):
        stream.feed(landmarks, t=t)
    assert stream.finish() == "HELLO"
    assert stream.stats.words == 1


def test_gap_ends_word_and_sentence_joins():
    stream = SignStream()
    assert _feed_letters(stream, "HI") == "HI"


def test_no_hand_frames_do_not_crash():
    stream = SignStream()
    for _ in range(5):
        stream.feed(None, t=0.0)
    assert stream.finish() == ""
    assert stream.stats.frames == 5
    assert stream.stats.hand_coverage == 0.0


def test_latency_is_measured():
    stream = SignStream()
    for landmarks, t in generate_word_frames("YES"):
        stream.feed(landmarks, t=t)
    assert stream.stats.frames_with_hand > 0
    assert stream.stats.mean_latency_ms < 50
