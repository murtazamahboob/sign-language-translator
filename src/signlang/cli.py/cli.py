"""Command-line interface for the sign-language translator."""

from __future__ import annotations

import argparse
import json
import sys

from .classifier import classify
from .poses import ALPHABET, generate, generate_alphabet, motion_of
from .stream import SignStream
from .words import WORDS, generate_word_frames, recognise_word


def _load_landmarks(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cmd_alphabet(_args: argparse.Namespace) -> int:
    print("ASL fingerspelling alphabet (static handshapes; J and Z add motion):")
    for letter in ALPHABET:
        motion = motion_of(letter)
        print(f"  {letter}" + (f"  (+ {motion})" if motion else ""))
    print(f"\n{len(ALPHABET)} letters. Words: {', '.join(WORDS)}")
    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    landmarks = _load_landmarks(args.file)
    result = classify(landmarks, motion=args.motion or "")
    out = {"letter": result.letter, "confidence": result.confidence,
           "reason": result.reason, "thumb": result.features.thumb_mode,
           "direction": result.features.direction,
           "reach": result.features.reach}
    print(json.dumps(out, indent=2) if args.json else
          f"{result.letter}  ({result.confidence:.2f}) — {result.reason}")
    return 0


def cmd_spell(args: argparse.Namespace) -> int:
    """Translate a JSONL file: one frame per line, {"landmarks": [...], "t": 0.0}."""
    stream = SignStream()
    with open(args.file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            landmarks = frame.get("landmarks") or None
            stream.feed(landmarks, t=float(frame.get("t", 0.0)))
    print(stream.finish())
    return 0


def cmd_demo(_args: argparse.Namespace) -> int:
    print("Real-Time Sign Language Translator — demo\n")
    print("Fingerspelling the alphabet (clean synthetic poses):")
    correct = 0
    for letter, landmarks in generate_alphabet().items():
        got = classify(landmarks, motion=motion_of(letter)).letter
        correct += got == letter
        print(f"  {letter} -> {got}" + ("" if got == letter else "  MISMATCH"))
    print(f"\nAlphabet: {correct}/{len(ALPHABET)} correct\n")
    print("Words (synthetic frame sequences through the live stream):")
    for word in WORDS:
        stream = SignStream()
        for landmarks, t in generate_word_frames(word):
            stream.feed(landmarks, t=t)
        sentence = stream.finish()
        mark = "OK " if sentence == word else "MISS"
        print(f"  {mark} {word:<12} -> {sentence!r}")
    print("\nFingerspelling 'HELLO' is a word sign, so spell it as H-E-L-L-O:")
    stream = SignStream()
    t = 0.0
    for letter in "HELLO":
        for _ in range(3):
            stream.feed(generate(letter), t=t)
            t += 0.1
        stream.feed(None, t=t)
        t += 1.2
    print(f"  -> {stream.finish()!r}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    import random

    from .poses import generate

    seeds = range(args.seeds)
    total = correct = 0
    per_letter: dict[str, int] = {}
    for letter in ALPHABET:
        hits = 0
        for seed in seeds:
            landmarks = generate(letter, seed=100 + seed, jitter=args.jitter)
            if classify(landmarks, motion=motion_of(letter)).letter == letter:
                hits += 1
        per_letter[letter] = hits
        total += len(seeds)
        correct += hits
    print(f"Alphabet benchmark: {args.seeds} jittered samples per letter "
          f"(jitter={args.jitter})")
    worst = sorted(per_letter.items(), key=lambda kv: kv[1])[:5]
    print(f"Overall: {correct}/{total} = {correct / total:.1%}")
    print("Weakest letters: " + ", ".join(f"{l} {h}/{args.seeds}" for l, h in worst))
    word_hits = 0
    for word in WORDS:
        frames = generate_word_frames(word)
        result = recognise_word(frames)
        word_hits += result is not None and result.word == word
    print(f"Words: {word_hits}/{len(WORDS)} recognised from frame windows")
    rng = random.Random(7)
    stream = SignStream()
    t = 0.0
    for letter in "CAT":
        for _ in range(3):
            stream.feed(generate(letter, seed=rng.randint(0, 999)), t=t)
            t += 0.1
    t += 1.2
    stream.feed(None, t=t)
    print(f"Streamed 'CAT' -> {stream.finish()!r}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signlang",
        description="Real-time ASL sign-language translator (offline, landmark-based).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("alphabet", help="list the alphabet and the word signs")
    p = sub.add_parser("classify", help="classify one landmark JSON file")
    p.add_argument("file")
    p.add_argument("--motion", default="")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_classify)
    p = sub.add_parser("spell", help="translate a JSONL frame file into a sentence")
    p.add_argument("file")
    p.set_defaults(func=cmd_spell)
    sub.add_parser("demo", help="run the alphabet + words demo").set_defaults(func=cmd_demo)
    p = sub.add_parser("benchmark", help="accuracy on jittered synthetic poses")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--jitter", type=float, default=0.01)
    p.set_defaults(func=cmd_benchmark)
    sub.choices["alphabet"].set_defaults(func=cmd_alphabet)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
