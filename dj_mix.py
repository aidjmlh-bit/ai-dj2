"""dj_mix.py — Unified DJ mix entry point.

Given two WAV files, automatically selects the best mix strategy and produces
a single output WAV in output/mixes/:

  1. LOOP MIX   — BPM within ±10 AND  keys compatible
  2. TIGHT      — BPM within ±5  OR  (keys compatible AND BPM within ±15)
  3. LOOSE      — fallback (always succeeds)

Usage:
    python dj_mix.py <song1.wav> <song2.wav> [output_dir]

Output filename:
    {song1}_{song2}_{mode}_t{M}m{SS}s.wav
    e.g.  Location_Trance_loop_t1m45s.wav
"""

from __future__ import annotations

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "sections"))

from get_bpm import get_bpm
from sections.get_chorus import find_chorus
from sections.get_verse  import find_verse
from many_transitions import (
    BPM_TIGHT_THRESHOLD,
    BPM_LOOSE_THRESHOLD,
    get_key,
    keys_compatible,
    make_transition,
)
from loop_mix import build_loop_mix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_ts(seconds: float) -> str:
    """Format seconds as tXmYYs for use in filenames."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"t{m}m{s:02d}s"


def _fmt_display(seconds: float) -> str:
    """Format seconds as M:SS for console display."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def _song_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _transition_timestamp(mode: str, song1_path: str, bpm1: float) -> float:
    """Return approximate transition-start time (seconds) for filename labelling."""
    try:
        chorus1 = find_chorus(song1_path)
        if not chorus1:
            return 0.0

        if mode == "loop":
            # Composite section starts at end of Song 1 Chorus 1
            return chorus1[0][1]

        if mode == "tight":
            # Transition window opens at Song 1 Chorus 1 start
            return chorus1[0][0]

        # Loose — transition anchored to Song 1 Verse 2 start; fallback to chorus end
        try:
            verse1 = find_verse(song1_path)
            if verse1 and len(verse1) >= 2:
                return verse1[1][0]
        except Exception:
            pass
        return chorus1[0][1]

    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main routing logic
# ---------------------------------------------------------------------------

def dj_mix(song1_path: str, song2_path: str, output_dir: str = "output") -> str:
    """Analyse, pick strategy, build mix, rename output file.

    Returns:
        Path to the final mix WAV.
    """
    for p in (song1_path, song2_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Audio file not found: {p!r}")

    # ── Analyse ─────────────────────────────────────────────────────────────
    print("Analysing songs…")
    bpm1 = get_bpm(song1_path)
    bpm2 = get_bpm(song2_path)
    key1 = get_key(song1_path)
    key2 = get_key(song2_path)

    bpm_diff  = abs(bpm1 - bpm2)
    bpm_loop  = bpm_diff <= 10
    bpm_ok    = bpm_diff <= BPM_TIGHT_THRESHOLD
    bpm_loose = bpm_diff <= BPM_LOOSE_THRESHOLD
    key_ok    = keys_compatible(key1, key2)

    def _cam(c: tuple[int, str]) -> str:
        return f"{c[0]}{c[1]}"

    print(
        f"\n  Song 1: {bpm1:.1f} BPM  key {_cam(key1)}\n"
        f"  Song 2: {bpm2:.1f} BPM  key {_cam(key2)}\n"
        f"  BPM diff: {bpm_diff:.1f}  |  Keys compatible: {key_ok}"
    )

    # ── Pick mode ────────────────────────────────────────────────────────────
    loop_eligible  = bpm_loop and key_ok
    tight_eligible = bpm_ok or (key_ok and bpm_loose)

    if loop_eligible:
        mode = "loop"
    elif tight_eligible:
        mode = "tight"
    else:
        mode = "loose"

    # ── Transition timestamp (for filename) ──────────────────────────────────
    print(f"\nComputing transition timestamp for filename…")
    trans_sec = _transition_timestamp(mode, song1_path, bpm1)

    print(
        f"\n{'─'*52}\n"
        f"  Mode       : {mode.upper()}\n"
        f"  Transition : {_fmt_display(trans_sec)}\n"
        f"{'─'*52}\n"
    )

    # ── Build mix ────────────────────────────────────────────────────────────
    if mode == "loop":
        builder_path = build_loop_mix(song1_path, song2_path, output_dir=output_dir)
    else:
        builder_path = make_transition(song1_path, song2_path, output_dir=output_dir)

    # ── Rename to canonical format ───────────────────────────────────────────
    s1     = _song_stem(song1_path)
    s2     = _song_stem(song2_path)
    ts_str = _fmt_ts(trans_sec)
    new_name = f"{s1}_{s2}_{mode}_{ts_str}.wav"
    new_path = os.path.join(os.path.dirname(builder_path), new_name)

    os.replace(builder_path, new_path)
    print(f"\nFinal mix → {new_path}")
    return new_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python dj_mix.py <song1.wav> <song2.wav> [output_dir]")
        sys.exit(1)

    song1    = sys.argv[1]
    song2    = sys.argv[2]
    out_dir  = sys.argv[3] if len(sys.argv) == 4 else "output"

    try:
        result = dj_mix(song1, song2, output_dir=out_dir)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}")
        sys.exit(1)
