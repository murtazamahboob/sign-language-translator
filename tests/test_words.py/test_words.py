from signlang.words import WORDS, generate_word_frames, recognise_word


def test_all_words_recognised():
    for word in WORDS:
        frames = generate_word_frames(word)
        result = recognise_word(frames)
        assert result is not None, word
        assert result.word == word, f"{word} -> {result}"


def test_word_templates_cover_seven_signs():
    assert len(WORDS) == 7
    assert set(WORDS) == {"HELLO", "THANK-YOU", "YES", "NO", "PLEASE",
                          "GOOD", "I-LOVE-YOU"}


def test_too_few_frames_recognises_nothing():
    frames = generate_word_frames("HELLO")[:3]
    assert recognise_word(frames) is None


def test_mixed_poses_recognise_nothing():
    hello = generate_word_frames("HELLO")
    yes = generate_word_frames("YES")
    mixed = hello[:6] + yes[:6]
    assert recognise_word(mixed) is None


def test_unknown_word_template_rejected():
    import pytest
    with pytest.raises(ValueError):
        generate_word_frames("GOODBYE")
