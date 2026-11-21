"""Demo: fingerspell the alphabet, recognise word signs, stream a sentence.

Writes outputs/report.md and outputs/summary.json next to the project.
Run from the project root:  python examples/demo.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from signlang.classifier import classify  # noqa: E402
from signlang.poses import ALPHABET, generate, generate_alphabet, motion_of  # noqa: E402
from signlang.stream import SignStream  # noqa: E402
from signlang.words import WORDS, generate_word_frames  # noqa: E402


def main() -> None:
    print("Real-Time Sign Language Translator — demo\n")
    correct = 0
    rows = []
    for letter, landmarks in generate_alphabet().items():
        got = classify(landmarks, motion=motion_of(letter)).letter
        correct += got == letter
        rows.append((letter, got, got == letter))
    print(f"Alphabet: {correct}/{len(ALPHABET)} correct")
    for letter, got, ok in rows:
        print(f"  {letter} -> {got}" + ("" if ok else "  MISMATCH"))

    print("\nWord signs through the live stream:")
    word_results = {}
    for word in WORDS:
        stream = SignStream()
        for landmarks, t in generate_word_frames(word):
            stream.feed(landmarks, t=t)
        sentence = stream.finish()
        word_results[word] = sentence
        print(f"  {word:<12} -> {sentence!r}")

    print("\nStreaming a fingerspelled sentence: 'CAT DOG'")
    stream = SignStream()
    t = 0.0
    for word in ("CAT", "DOG"):
        for letter in word:
            for _ in range(3):
                stream.feed(generate(letter), t=t)
                t += 0.1
            # Short pause between letters, long pause between words.
            t += 0.4
            stream.feed(None, t=t)
        t += 2.0
        stream.feed(None, t=t)
    sentence = stream.finish()
    print(f"  -> {sentence!r}  ({stream.stats.frames} frames, "
          f"mean latency {stream.stats.mean_latency_ms:.2f} ms/frame)")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    report = ["# Sign Language Translator — demo report\n",
              f"Alphabet: {correct}/{len(ALPHABET)} correct\n",
              "| Letter | Recognised | Correct |",
              "|---|---|---|"]
    report += [f"| {a} | {g} | {'yes' if ok else 'no'} |" for a, g, ok in rows]
    report += ["\n## Word signs\n"]
    report += [f"- {w}: {s!r}" for w, s in word_results.items()]
    report += [f"\nStreamed sentence: {sentence!r}",
               f"Mean latency: {stream.stats.mean_latency_ms:.2f} ms/frame"]
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({
        "alphabet_correct": correct, "alphabet_total": len(ALPHABET),
        "words": word_results, "streamed_sentence": sentence,
        "mean_latency_ms": round(stream.stats.mean_latency_ms, 3),
    }, indent=2), encoding="utf-8")
    print("\nWrote outputs/report.md and outputs/summary.json")


if __name__ == "__main__":
    main()
