"""Verse detection from WAV audio files using chroma self-similarity."""

from __future__ import annotations

import os
import sys

import numpy as np
import librosa
import soundfile as sf

# Allow importing get_bpm from the project root and get_chorus from sections/.
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root)   # for get_bpm
sys.path.insert(0, _here)   # for get_chorus (same directory)

from get_bpm import get_bpm
from get_chorus import find_chorus


def find_verse(filepath: str) -> list[tuple[float, float]]:
    """Find every verse instance in a WAV file and return their timestamps.

    Algorithm:
        1. Split the song into fixed-length bars using the estimated BPM.
        2. Compute a mean chroma (CQT) vector and RMS per bar.
        3. Call find_chorus() to locate chorus bars and measure chorus energy.
        4. For each chorus start, look at the 8 bars immediately before it —
           these pre-chorus windows anchor the verse template.
        5. Average those pre-chorus chroma vectors into a single verse template.
        6. A bar is a verse candidate if it is:
             • harmonically similar to the verse template (cosine > threshold)
             • NOT already labelled as chorus
             • lower energy than the median chorus RMS
        7. Merge consecutive verse-candidate bars into segments; drop segments
           shorter than ``min_verse_bars``.
        8. Keep only patterns that repeat ≥ 2 times as separate segments.

    Args:
        filepath: Absolute or relative path to a ``.wav`` audio file.

    Returns:
        List of ``(start_sec, end_sec)`` tuples, one per detected verse
        instance, in chronological order.  Empty list if no verse is found.

    Raises:
        FileNotFoundError: If no file exists at *filepath*.
        ValueError: If the audio cannot be decoded or is too short to analyse.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath!r}")

    # ------------------------------------------------------------------ #
    # 1. Load audio and derive bar length from BPM                        #
    # ------------------------------------------------------------------ #
    try:
        y, sr = librosa.load(filepath, mono=True, sr=None)
    except Exception as exc:
        raise ValueError(f"Failed to decode audio file {filepath!r}: {exc}") from exc

    if y.size == 0:
        raise ValueError(f"Audio file contains no samples: {filepath!r}")

    bpm = get_bpm(filepath)
    bar_duration = 4.0 * (60.0 / bpm)       # seconds per bar (assumes 4/4)
    bar_samples = int(bar_duration * sr)

    n_bars = len(y) // bar_samples
    if n_bars < 4:
        raise ValueError(
            f"Audio is too short to analyse ({n_bars} full bars detected). "
            "At least 4 bars are required."
        )

    # ------------------------------------------------------------------ #
    # 2. Chroma + RMS per bar                                             #
    # ------------------------------------------------------------------ #
    chroma_vecs = np.zeros((n_bars, 12), dtype=np.float32)
    rms_vals = np.zeros(n_bars, dtype=np.float32)

    for i in range(n_bars):
        bar = y[i * bar_samples : (i + 1) * bar_samples]
        chroma = librosa.feature.chroma_cqt(y=bar, sr=sr)  # (12, T)
        chroma_vecs[i] = chroma.mean(axis=1)
        rms_vals[i] = np.sqrt(np.mean(bar ** 2))

    norms = np.linalg.norm(chroma_vecs, axis=1, keepdims=True)
    X = chroma_vecs / (norms + 1e-8)        # L2-normalised, shape (N, 12)

    # ------------------------------------------------------------------ #
    # 3. Find chorus → anchor points and energy ceiling                   #
    # ------------------------------------------------------------------ #
    chorus_timestamps = find_chorus(filepath)

    if not chorus_timestamps:
        # Without a chorus reference we can't locate the verse reliably.
        return []

    # Convert chorus timestamps to bar indices.
    chorus_bar_starts = [int(ts[0] / bar_duration) for ts in chorus_timestamps]
    chorus_bar_ends   = [min(n_bars - 1, int(ts[1] / bar_duration))
                         for ts in chorus_timestamps]

    # Build a boolean mask of all chorus bars so we can exclude them later.
    chorus_bar_mask = np.zeros(n_bars, dtype=bool)
    for s, e in zip(chorus_bar_starts, chorus_bar_ends):
        chorus_bar_mask[s : e + 1] = True

    # Median RMS of all chorus bars → energy ceiling for verse candidates.
    chorus_rms = np.median(rms_vals[chorus_bar_mask])

    # ------------------------------------------------------------------ #
    # 4. Build verse template from pre-chorus windows                     #
    # ------------------------------------------------------------------ #
    # The block immediately before each chorus is almost always the verse
    # (or pre-chorus).  Averaging these windows across all chorus instances
    # gives a robust harmonic fingerprint of the verse.
    lookback = 8        # bars to look back before each chorus start
    pre_chorus_vecs: list[np.ndarray] = []

    for c_start in chorus_bar_starts:
        window_start = max(0, c_start - lookback)
        if c_start > window_start:
            pre_chorus_vecs.extend(X[window_start:c_start])

    if not pre_chorus_vecs:
        return []

    verse_template = np.mean(pre_chorus_vecs, axis=0).astype(np.float32)
    verse_template /= (np.linalg.norm(verse_template) + 1e-8)

    # ------------------------------------------------------------------ #
    # 5. Identify verse-candidate bars                                    #
    # ------------------------------------------------------------------ #
    threshold = 0.85
    sims_to_verse = X @ verse_template          # (N,) cosine similarities

    verse_mask = (
        (sims_to_verse > threshold) &           # harmonically matches verse
        (~chorus_bar_mask) &                    # not already a chorus bar
        (rms_vals < chorus_rms)                 # lower energy than chorus
    )

    # ------------------------------------------------------------------ #
    # 6. Group consecutive verse bars into segments                       #
    # ------------------------------------------------------------------ #
    min_verse_bars = 4      # ~8 s at 120 BPM; drops isolated stray bars

    segments: list[tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for i, is_verse in enumerate(verse_mask):
        if is_verse and not in_seg:
            seg_start = i
            in_seg = True
        elif not is_verse and in_seg:
            if i - seg_start >= min_verse_bars:
                segments.append((seg_start, i - 1))
            in_seg = False
    if in_seg and n_bars - seg_start >= min_verse_bars:
        segments.append((seg_start, n_bars - 1))

    # A verse must repeat — drop the whole result if only one instance found.
    if len(segments) < 2:
        return []

    # ------------------------------------------------------------------ #
    # 7. Convert bar indices → seconds                                    #
    # ------------------------------------------------------------------ #
    timestamps = [
        (s * bar_duration, (e + 1) * bar_duration)
        for s, e in segments
    ]

    # ------------------------------------------------------------------ #
    # 8. Save first verse instance as a wav snippet                       #
    # ------------------------------------------------------------------ #
    s0, e0 = segments[0]
    verse_audio = y[s0 * bar_samples : (e0 + 1) * bar_samples]
    snippet_path = os.path.join(os.path.dirname(os.path.abspath(filepath)),
                                "verse_snippet.wav")
    sf.write(snippet_path, verse_audio, sr)

    return timestamps


# ---------------------------------------------------------------------- #
# CLI entry point                                                         #
# ---------------------------------------------------------------------- #

def _fmt(seconds: float) -> str:
    """Format seconds as M:SS.ss."""
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m}:{s:05.2f}"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_verse.py <path/to/track.wav>")
        sys.exit(1)

    wav_path = sys.argv[1]

    try:
        results = find_verse(wav_path)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}")
        sys.exit(1)

    if not results:
        print("No verse detected. The chorus may not have been found, or the "
              "verse pattern doesn't repeat.")
        sys.exit(0)

    snippet_path = os.path.join(os.path.dirname(os.path.abspath(wav_path)),
                                "verse_snippet.wav")
    print(f"Verse instances found: {len(results)}")
    for idx, (start, end) in enumerate(results, 1):
        tag = "  <- saved as verse_snippet.wav" if idx == 1 else ""
        print(f"  [{idx}]  {_fmt(start)} -> {_fmt(end)}  ({end - start:.1f} s){tag}")
