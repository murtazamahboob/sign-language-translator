"""Real-time streaming translator: frames in, transcript out.

A frame is landmarks (or None when no hand is visible) with a
timestamp. A letter is committed only after it is held for
``hold_frames`` consecutive frames, so flicker between two similar
handshapes never reaches the transcript. A gap with no hand (or a
long pause) ends the current word; words join into a sentence.

The engine also watches for whole-word signs: when a window of frames
matches a word template (pose + wrist motion), the word is emitted
instead of the letters underneath it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .classifier import classify
from .landmarks import Landmarks
from .words import WORDS, recognise_word


@dataclass
class TranscriptEvent:
    kind: str  # "letter" | "word" | "space"
    text: str
    t: float


@dataclass
class StreamStats:
    frames: int = 0
    frames_with_hand: int = 0
    letters: int = 0
    words: int = 0
    total_latency_ms: float = 0.0

    @property
    def mean_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.frames_with_hand, 1)

    @property
    def hand_coverage(self) -> float:
        return self.frames_with_hand / max(self.frames, 1)


class SignStream:
    """Debounced, streaming sign-language translator."""

    def __init__(self, hold_frames: int = 3, word_window: int = 12,
                 gap_seconds: float = 1.5):
        self.hold_frames = hold_frames
        self.word_window = word_window
        self.gap_seconds = gap_seconds
        self._pending: str | None = None
        self._pending_count = 0
        self._last_hand_t: float | None = None
        self._window: list[tuple[Landmarks, float, int]] = []
        self._frame_no = 0
        self._letter_frames: list[tuple[int, str]] = []
        self._suppress = False  # set after a word sign, cleared when the hand leaves
        self.words: list[str] = []
        self._current_word: list[str] = []
        self.events: list[TranscriptEvent] = []
        self.stats = StreamStats()

    @property
    def sentence(self) -> str:
        parts = list(self.words)
        if self._current_word:
            parts.append("".join(self._current_word))
        return " ".join(parts)

    def _emit_word(self, word: str, t: float) -> None:
        # Letters committed from frames inside this word's window were
        # the word's handshape, not fingerspelling: take them back.
        cutoff = self._window[0][2] if self._window else 0
        kept = [(n, ch) for n, ch in self._letter_frames if n < cutoff]
        removed = len(self._letter_frames) - len(kept)
        self._letter_frames = kept
        if removed:
            self._current_word = self._current_word[: len(self._current_word) - removed]
            self.events = [e for e in self.events
                           if not (e.kind == "letter" and e.t >= self._window[0][1])]
            self.stats.letters -= removed
        self._flush_letters()
        self.words.append(word)
        self.events.append(TranscriptEvent("word", word, t))
        self.stats.words += 1
        # The rest of this hand's frames are the same sign: do not
        # re-commit its letters or re-fire the word until the hand leaves.
        self._suppress = True

    def _flush_letters(self) -> None:
        if self._current_word:
            self.words.append("".join(self._current_word))
            self._current_word = []

    def _commit_letter(self, letter: str, t: float, frame_no: int) -> None:
        if letter == "?":
            return
        self._current_word.append(letter)
        self._letter_frames.append((frame_no, letter))
        self.events.append(TranscriptEvent("letter", letter, t))
        self.stats.letters += 1

    def feed(self, landmarks: Landmarks | None, t: float | None = None) -> list[TranscriptEvent]:
        """Feed one frame; return any events it produced."""
        t = time.monotonic() if t is None else t
        start = time.perf_counter()
        self.stats.frames += 1
        self._frame_no += 1
        produced: list[TranscriptEvent] = []
        if landmarks is None:
            self._pending = None
            self._pending_count = 0
            self._window.clear()
            self._suppress = False
            if self._last_hand_t is not None and t - self._last_hand_t >= self.gap_seconds:
                self._flush_letters()
                self._last_hand_t = None
            return produced
        self.stats.frames_with_hand += 1
        self._last_hand_t = t
        if self._suppress:
            # Still holding the handshape of a word just emitted.
            self.stats.total_latency_ms += (time.perf_counter() - start) * 1000
            return produced
        self._window.append((landmarks, t, self._frame_no))
        if len(self._window) > self.word_window:
            self._window.pop(0)

        # Whole-word signs take priority over fingerspelling.
        if len(self._window) >= 8:
            result = recognise_word([(lm, tt) for lm, tt, _n in self._window])
            if result is not None:
                self._emit_word(result.word, t)
                produced.append(self.events[-1])
                self._window.clear()
                self._pending = None
                self._pending_count = 0
                self.stats.total_latency_ms += (time.perf_counter() - start) * 1000
                return produced

        letter = classify(landmarks).letter
        if letter == self._pending:
            self._pending_count += 1
        else:
            self._pending = letter
            self._pending_count = 1
        if self._pending_count >= self.hold_frames and letter != "?":
            before = len(self.events)
            self._commit_letter(letter, t, self._frame_no)
            produced.extend(self.events[before:])
            self._pending = None
            self._pending_count = 0
        self.stats.total_latency_ms += (time.perf_counter() - start) * 1000
        return produced

    def finish(self) -> str:
        """End the stream and return the full sentence."""
        self._flush_letters()
        return self.sentence
