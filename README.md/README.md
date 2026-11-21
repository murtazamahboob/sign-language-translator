# Real-Time Sign Language Translator — ASL Handshapes to Text, Live

**Offline, deterministic, standard-library only: a rule-based ASL translator that reads 21 hand landmarks and turns fingerspelling and word signs into a live transcript — no model, no API key, no camera required to try it.**

Fingerspelling apps usually need a trained model and a labelled dataset. This one doesn't: the classifier is a decision tree over hand geometry — how far each finger is extended, where the thumb sits, which way the hand points, whether the index crosses the middle. That makes every decision explainable ("R, because the index is crossed over the middle") and the whole thing runnable anywhere Python runs.

## Results (measured, not claimed)

Benchmark on the bundled synthetic alphabet (5 jittered samples per letter, jitter σ=0.01):

| Check | Result |
|---|---|
| Alphabet, clean poses | **26/26 correct (100%)** |
| Alphabet, jittered poses | **126/130 = 96.9%** (weakest: C, M, O, X at 4/5) |
| Word signs (7) | **7/7 recognised** from frame windows |
| Streamed "CAT DOG" | **"CAT DOG"** — letters debounced, words split on pauses |
| Mean latency | **0.04–0.05 ms per frame** (far under a 16 ms real-time budget) |

## Features

- **Full ASL fingerspelling alphabet (A–Z)** — including the confusable pairs the tree is built to separate: A/S/T (thumb position), M/N (thumb tucked under two vs three fingers), U/V (finger spread), K/P (orientation), R (index crossed over middle), D/G (orientation), E (tight curl + low thumb), X (hooked index), C (half-curled), O (fingertips meet).
- **Dynamic letters J and Z** — same handshape as I / pointing index, plus a hook / Z-trace motion hint.
- **7 word signs** — HELLO (wave), THANK-YOU (down-and-out), YES (nod), NO (tap), PLEASE (circle), GOOD (down from chin), I-LOVE-YOU (static handshape). Recognised from the wrist trajectory over a frame window, and they take priority over the fingerspelling underneath them.
- **Real-time streaming engine** — a letter is committed only after it is held for 3 frames, so flicker never reaches the transcript; a 1.5 s pause ends a word; word signs suppress their own letters so HELLO never comes out as "HELLO B".
- **Orientation-aware** — the hand is canonicalised (framing) and rotated back upright before the thumb is read, so sideways (G, H) and downward (P, Q) handshapes classify the same as upright ones.
- **Synthetic pose generator** — a parametric 21-landmark hand model (seeded, deterministic) supplies the whole alphabet plus word-sign frame sequences, so the classifier is benchmarked offline with no camera or labelled data. The generator is a *test fixture*, clearly labelled as such — the classifier only ever sees landmarks.
- **Webcam adapter (optional)** — `webcam.py` wires a camera to any landmark provider (e.g. MediaPipe Hands) when OpenCV is installed. The core package has no dependencies and never guesses landmarks from raw pixels.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

signlang alphabet                       # list the alphabet + word signs
signlang demo                           # alphabet + words + streamed sentence
signlang benchmark --seeds 5            # jittered accuracy + word recognition

# Classify one hand (landmarks JSON, 21 [x, y] pairs)
signlang classify examples/b.json --json

# Translate a frame stream (JSONL: {"landmarks": [...], "t": 0.0})
signlang spell frames.jsonl
```

## Python API

```python
from signlang import SignStream, classify, generate, generate_word_frames

classify(generate("R")).letter            # 'R'
classify(generate("J"), motion="j-hook").letter  # 'J'

stream = SignStream()
for landmarks, t in generate_word_frames("HELLO"):
    stream.feed(landmarks, t=t)
stream.finish()  # 'HELLO'
```

## How it works

1. **Landmarks** (`landmarks.py`) — the 21-point hand model (MediaPipe Hands order). Features: per-finger reach (tip-to-MCP distance over nominal length), thumb mode (up / across / tucked / front / touch_index / touch_middle / side / down / between), hand orientation, index/middle spread and crossing. Landmarks are canonicalised first (wrist to a fixed point, hand scaled to a fixed size) so framing and distance don't change the answer.
2. **Poses** (`poses.py`) — parametric handshape specs per letter plus a deterministic generator with seeded jitter. Used for tests, the demo and the benchmark only.
3. **Classifier** (`classifier.py`) — a decision tree over those features. Motion-gated letters (J, Z) first, then thumb-meets-index (F, O), orientation (G/H sideways, P/Q down), then finger counts (B, C, W, K/R/U/V, D/L/X/I/Y, the fist family A/S/T/M/N/E).
4. **Words** (`words.py`) — a word is a base handshape plus a wrist-motion template (wave, nod, tap, circle, down, down-out, hold), matched over a 12-frame window.
5. **Stream** (`stream.py`) — debouncing, word priority (with committed letters taken back), pause-based word boundaries, latency stats.

## Honest limits

- The bundled alphabet is **synthetic**: poses are generated from handshape specs, not photographed hands. Accuracy numbers measure the classifier against that generator, not against real signers — a real deployment needs a landmark detector (MediaPipe Hands) and validation on real footage, which the webcam adapter is built to accept.
- Static landmarks alone can't tell some signs apart (J vs I, Z vs a pointing hand) — those need the motion hint, which the stream supplies from the wrist trajectory.
- Only 7 word signs are bundled; the template table in `words.py` is where new ones go.
