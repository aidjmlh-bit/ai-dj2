"""Chorus detection from WAV audio files using chroma self-similarity."""

from __future__ import annotations

import os
import sys

import numpy as np
import librosa
import soundfile as sf

# Allow running as a script from any working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bpm import get_bpm


def find_chorus(filepath: str) -> list[tuple[float, float]]:
    """Find every chorus instance in a WAV file and return their timestamps.

    Algorithm:
        1. Split the song into fixed-length bars using the estimated BPM.
        2. Compute a mean chroma (CQT) vector per bar as a harmonic fingerprint.
        3. Build an N×N cosine self-similarity matrix over all bars.
        4. Score each bar by combining how many other bars it resembles
           (repeat_count) with its RMS energy.  High score → likely chorus.
        5. Use the top-scoring bar as a chorus template and find every other
           bar with cosine similarity above the threshold.
        6. Merge consecutive matching bars into segments; drop segments that
           are shorter than ``min_chorus_bars``.

    Args:
        filepath: Absolute or relative path to a ``.wav`` audio file.

    Returns:
        List of ``(start_sec, end_sec)`` tuples, one per detected chorus
        instance, in chronological order.

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
        # chroma_cqt is more pitch-stable than chroma_stft for short windows.
        chroma = librosa.feature.chroma_cqt(y=bar, sr=sr)  # (12, T)
        chroma_vecs[i] = chroma.mean(axis=1)
        rms_vals[i] = np.sqrt(np.mean(bar ** 2))

    # ------------------------------------------------------------------ #
    # 3. Cosine self-similarity matrix (pure numpy)                       #
    # ------------------------------------------------------------------ #
    norms = np.linalg.norm(chroma_vecs, axis=1, keepdims=True)
    X = chroma_vecs / (norms + 1e-8)        # L2-normalised, shape (N, 12)
    sim_matrix = X @ X.T                    # (N, N) cosine similarities

    # ------------------------------------------------------------------ #
    # 4. Score bars: repeat count weighted with energy                    #
    # ------------------------------------------------------------------ #
    threshold = 0.85
    sim_binary = sim_matrix > threshold
    np.fill_diagonal(sim_binary, False)     # don't count self-match
    repeat_count = sim_binary.sum(axis=1).astype(np.float32)

    # Graceful normalisation — avoid divide-by-zero on silent tracks.
    repeat_score = repeat_count / (repeat_count.max() + 1e-8)
    energy_score = rms_vals / (rms_vals.max() + 1e-8)

    # Repetition is the stronger chorus signal; energy breaks ties.
    combined_score = 0.6 * repeat_score + 0.4 * energy_score
    template_idx = int(np.argmax(combined_score))

    # ------------------------------------------------------------------ #
    # 5. Identify all bars matching the chorus template                   #
    # ------------------------------------------------------------------ #
    sims_to_template = X @ X[template_idx]  # (N,) cosine sim vs template

    # Chroma alone is too permissive — a song in a consistent key will have
    # high cosine similarity across ALL bars.  Add an energy gate so that
    # only loud, harmonically-matching bars are labelled chorus.
    chorus_mask = (sims_to_template > threshold) & (rms_vals > np.median(rms_vals))

    # ------------------------------------------------------------------ #
    # 6. Group consecutive chorus bars into segments                      #
    # ------------------------------------------------------------------ #
    min_chorus_bars = 4     # ~8 s at 120 BPM; filters isolated stray bars

    segments: list[tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for i, is_chorus in enumerate(chorus_mask):
        if is_chorus and not in_seg:
            seg_start = i
            in_seg = True
        elif not is_chorus and in_seg:
            if i - seg_start >= min_chorus_bars:
                segments.append((seg_start, i - 1))
            in_seg = False
    if in_seg and n_bars - seg_start >= min_chorus_bars:
        segments.append((seg_start, n_bars - 1))

    # ------------------------------------------------------------------ #
    # 7. Convert bar indices → seconds                                    #
    # ------------------------------------------------------------------ #
    timestamps = [
        (s * bar_duration, (e + 1) * bar_duration)
        for s, e in segments
    ]

    # ------------------------------------------------------------------ #
    # 8. Save first chorus instance as a wav snippet                      #
    # ------------------------------------------------------------------ #
    if segments:
        s0, e0 = segments[0]
        chorus_audio = y[s0 * bar_samples : (e0 + 1) * bar_samples]
        snippet_path = os.path.join(os.path.dirname(os.path.abspath(filepath)),
                                    "chorus_snippet.wav")
        sf.write(snippet_path, chorus_audio, sr)

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
        print("Usage: python get_chorus.py <path/to/track.wav>")
        sys.exit(1)

    wav_path = sys.argv[1]

    try:
        results = find_chorus(wav_path)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}")
        sys.exit(1)

    if not results:
        print("No chorus detected. Try lowering the similarity threshold.")
        sys.exit(0)

    snippet_path = os.path.join(os.path.dirname(os.path.abspath(wav_path)),
                                "chorus_snippet.wav")
    print(f"Chorus instances found: {len(results)}")
    for idx, (start, end) in enumerate(results, 1):
        tag = "  <- saved as chorus_snippet.wav" if idx == 1 else ""
        print(f"  [{idx}]  {_fmt(start)} -> {_fmt(end)}  ({end - start:.1f} s){tag}")
