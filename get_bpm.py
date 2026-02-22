"""BPM estimation from WAV audio files."""

import os
import numpy as np
import librosa


def get_bpm(filepath: str) -> float:
    """Estimate the global BPM of a WAV audio file.

    Uses librosa's onset-strength-based beat tracking to estimate tempo.
    The onset envelope is computed with a median aggregator for robustness
    against transient noise, then fed into a dynamic-programming beat tracker.

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
        # Load as mono at the file's native sample rate so beat tracking
        # operates on the full stereo mix without resampling artifacts.
        y, sr = librosa.load(filepath, mono=True, sr=None)
    except Exception as exc:
        raise ValueError(
            f"Failed to decode audio file {filepath!r}: {exc}"
        ) from exc

    if y.size == 0:
        raise ValueError(f"Audio file contains no samples: {filepath!r}")

    # Onset strength envelope with median aggregation suppresses spurious
    # peaks from noise, giving a cleaner signal to the beat tracker.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)

    # Dynamic-programming beat tracker. start_bpm=120 is a weak prior;
    # the tracker will deviate substantially when the evidence is clear.
    tempo, _ = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        start_bpm=120.0,
        units="frames",
    )

    # librosa >=0.10 may return a 1-element ndarray instead of a scalar.
    bpm = round(float(np.atleast_1d(tempo)[0]), 2)

    if not (60.0 <= bpm <= 200.0):
        raise ValueError(
            f"Estimated BPM {bpm} is outside the valid range [60, 200]. "
            "The track may be silent, extremely slow, or extremely fast."
        )

    return bpm