"""BPM matching — speed up the slower of two songs to match the faster one."""

from __future__ import annotations

import os
import sys

import numpy as np
import librosa
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from get_bpm import get_bpm

# Maximum allowed BPM difference before stretching is rejected.
MAX_BPM_DIFF = 20   # absolute BPM


def match_bpm(filepath_a: str, filepath_b: str) -> str:
    """Speed up the slower of two songs to match the faster one's BPM.

    Uses librosa's phase vocoder (``librosa.effects.time_stretch``) which
    changes tempo without altering pitch.  The stretch is only applied when
    the two BPMs are within 20 BPM of each other; larger gaps introduce
    audible artefacts and are rejected.

    Args:
        filepath_a: Path to the first WAV file.
        filepath_b: Path to the second WAV file.

    Returns:
        Absolute path to the stretched output WAV file.

    Raises:
        FileNotFoundError: If either file does not exist.
        ValueError: If either file cannot be decoded, or if the BPM
            difference exceeds 20 BPM.
    """
    for fp in (filepath_a, filepath_b):
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Audio file not found: {fp!r}")

    # ------------------------------------------------------------------ #
    # 1. Detect BPMs                                                      #
    # ------------------------------------------------------------------ #
    bpm_a = get_bpm(filepath_a)
    bpm_b = get_bpm(filepath_b)

    faster_bpm = max(bpm_a, bpm_b)
    slower_bpm = min(bpm_a, bpm_b)

    # ------------------------------------------------------------------ #
    # 2. Enforce 20 BPM limit                                            #
    # ------------------------------------------------------------------ #
    if faster_bpm - slower_bpm > MAX_BPM_DIFF:
        raise ValueError(
            f"BPM difference is {faster_bpm - slower_bpm:.1f} "
            f"({bpm_a} vs {bpm_b}), which exceeds the {MAX_BPM_DIFF} BPM limit. "
            "The songs are too far apart to stretch cleanly."
        )

    # ------------------------------------------------------------------ #
    # 3. Decide which song to stretch (always speed up the slower one)    #
    # ------------------------------------------------------------------ #
    if bpm_a <= bpm_b:
        path_to_stretch = filepath_a
        original_bpm = bpm_a
        target_bpm = bpm_b
    else:
        path_to_stretch = filepath_b
        original_bpm = bpm_b
        target_bpm = bpm_a

    rate = target_bpm / original_bpm    # always > 1.0

    # ------------------------------------------------------------------ #
    # 4. Load audio (stereo-aware) and time-stretch                       #
    # ------------------------------------------------------------------ #
    try:
        # mono=False preserves stereo channels; sr=None keeps native rate.
        y, sr = librosa.load(path_to_stretch, mono=False, sr=None)
    except Exception as exc:
        raise ValueError(
            f"Failed to decode audio file {path_to_stretch!r}: {exc}"
        ) from exc

    # librosa.effects.time_stretch requires a 1-D array.
    # For stereo (shape 2 × N) we stretch each channel separately.
    if y.ndim == 1:
        y_stretched = librosa.effects.time_stretch(y, rate=rate)
    else:
        y_stretched = np.stack(
            [librosa.effects.time_stretch(ch, rate=rate) for ch in y]
        )

    # ------------------------------------------------------------------ #
    # 5. Save output                                                      #
    # ------------------------------------------------------------------ #
    stem = os.path.splitext(os.path.basename(path_to_stretch))[0]
    out_name = f"{stem}_matched_{int(round(target_bpm))}bpm.wav"
    out_path = os.path.join(os.path.dirname(os.path.abspath(path_to_stretch)),
                            out_name)

    # soundfile expects shape (samples,) for mono or (samples, channels) for
    # stereo, so we transpose the (channels, samples) array from librosa.
    if y_stretched.ndim == 1:
        sf.write(out_path, y_stretched, sr)
    else:
        sf.write(out_path, y_stretched.T, sr)

    return out_path


# ---------------------------------------------------------------------- #
# CLI entry point                                                         #
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python 2_match_bpm.py <track_a.wav> <track_b.wav>")
        sys.exit(1)

    path_a, path_b = sys.argv[1], sys.argv[2]

    try:
        result = match_bpm(path_a, path_b)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}")
        sys.exit(1)

    bpm_a = get_bpm(path_a)
    bpm_b = get_bpm(path_b)
    slower_path = path_a if bpm_a <= bpm_b else path_b
    original = min(bpm_a, bpm_b)
    target = max(bpm_a, bpm_b)

    print(f"Stretching : {os.path.basename(slower_path)}")
    print(f"  Original BPM : {original:.2f}")
    print(f"  Target BPM   : {target:.2f}")
    print(f"  Stretch rate : {target / original:.4f}x")
    print(f"  Saved to     : {result}")
