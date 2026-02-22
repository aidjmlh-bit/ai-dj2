"""BPM estimation from WAV audio files."""

import os
import essentia.standard as es


def get_bpm(filepath: str) -> float:
    """Estimate the global BPM of a WAV audio file.

    Uses Essentia's RhythmExtractor2013 with the multifeature method,
    which combines several beat-tracking algorithms for higher accuracy
    than onset-envelope approaches, especially on electronic music.

    Args:
        filepath: Absolute or relative path to a ``.wav`` audio file.

    Returns:
        Estimated BPM as a float rounded to 2 decimal places,
        guaranteed to be in the range [60.0, 200.0].

    Raises:
        FileNotFoundError: If no file exists at *filepath*.
        ValueError: If the file cannot be decoded as audio, if the audio
            contains no samples, or if the estimated BPM falls outside
            the valid range [60, 200].

    Example:
        >>> bpm = get_bpm("track.wav")
        >>> print(f"Estimated BPM: {bpm}")
        Estimated BPM: 128.0
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath!r}")

    try:
        audio = es.MonoLoader(filename=filepath)()
    except Exception as exc:
        raise ValueError(
            f"Failed to decode audio file {filepath!r}: {exc}"
        ) from exc

    if len(audio) == 0:
        raise ValueError(f"Audio file contains no samples: {filepath!r}")

    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, _, _, _, _ = rhythm_extractor(audio)

    bpm = round(float(bpm), 2)

    if not (60.0 <= bpm <= 200.0):
        raise ValueError(
            f"Estimated BPM {bpm} is outside the valid range [60, 200]. "
            "The track may be silent, extremely slow, or extremely fast."
        )

    return bpm
