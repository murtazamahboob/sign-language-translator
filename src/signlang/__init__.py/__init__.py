"""Real-Time Sign Language Translator — translate ASL handshapes to text.

Offline, deterministic, standard-library only. The classifier reads
21 hand landmarks (MediaPipe Hands order); a bundled synthetic pose
generator supplies the alphabet for testing and demos, and an
optional OpenCV adapter feeds a live camera through any landmark
provider (see webcam.py).
"""

from .classifier import Classification, classify, classify_letter
from .landmarks import HandFeatures, extract_features, validate_landmarks
from .poses import ALPHABET, POSES, generate, generate_alphabet
from .stream import SignStream, StreamStats, TranscriptEvent
from .words import WORDS, WordResult, generate_word_frames, recognise_word

__version__ = "1.0.0"
__all__ = [
    "ALPHABET",
    "POSES",
    "WORDS",
    "Classification",
    "HandFeatures",
    "SignStream",
    "StreamStats",
    "TranscriptEvent",
    "WordResult",
    "classify",
    "classify_letter",
    "extract_features",
    "generate",
    "generate_alphabet",
    "generate_word_frames",
    "recognise_word",
    "validate_landmarks",
]
