"""Smart DJ transition — selects tight (chorus→chorus) or loose (chorus→verse→chorus)
based on BPM closeness and harmonic (key) compatibility.

Tight transition  [BPM within ±5 AND keys compatible]:
    Song 1 plays as a continuous strip from Verse 1 start → transition point.
    Phase A (1 phrase = 8 bars = 32 beats at Song 1 BPM):
        Song 1 stems: lows fade 1→0, mids + highs held at full
        Song 2 stems: lows fade 0→1
    Hard cut at phrase boundary: Song 2 (all stems, full) takes over.

Loose transition  [otherwise]:
    Song 1 plays as a continuous strip from Verse 1 start → transition point.
    Phase A (1 phrase): Song 1 lows fade 1→0, Song 2 lows fade 0→1; Song 1 mids+highs held.
    Phase B (1 phrase): Song 1 mids+highs fade 1→0, Song 2 mids+highs fade 0→1.
    Song 2 continues from (Chorus 1 + 2 phrases) onward.

BPM rule: always speed up Song 2 to Song 1's BPM if Song 2 is slower; never slow down.

Key compatibility uses the Camelot wheel — any one of:
    1. Same number + same letter      (identical key)
    2. Same letter + number ±1        (adjacent on ring, wraps 12→1)
    3. Same number + opposite letter  (relative major / minor)
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import essentia.standard as es
import librosa
import numpy as np
import soundfile as sf

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "sections"))

from get_bpm import get_bpm
from get_chorus import find_chorus
from get_verse import find_verse


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BPM_TIGHT_THRESHOLD = 5   # |bpm1 - bpm2| ≤ this → eligible for tight transition

# Enharmonic normalisation: flatten → sharp equivalent
_ENHARMONICS: dict[str, str] = {
    "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#",
    "Ab": "G#", "Bb": "A#", "Cb": "B",
}

# (key_in_sharps, scale) → Camelot (number 1–12, letter "A" | "B")
# "B" = major ring, "A" = minor ring
_CAMELOT: dict[tuple[str, str], tuple[int, str]] = {
    # ── major keys (B ring) ──────────────────────────────────────────────
    ("B",  "major"): (1,  "B"),
    ("F#", "major"): (2,  "B"),
    ("C#", "major"): (3,  "B"),
    ("G#", "major"): (4,  "B"),
    ("D#", "major"): (5,  "B"),
    ("A#", "major"): (6,  "B"),
    ("F",  "major"): (7,  "B"),
    ("C",  "major"): (8,  "B"),
    ("G",  "major"): (9,  "B"),
    ("D",  "major"): (10, "B"),
    ("A",  "major"): (11, "B"),
    ("E",  "major"): (12, "B"),
    # ── minor keys (A ring) ──────────────────────────────────────────────
    ("G#", "minor"): (1,  "A"),
    ("D#", "minor"): (2,  "A"),
    ("A#", "minor"): (3,  "A"),
    ("F",  "minor"): (4,  "A"),
    ("C",  "minor"): (5,  "A"),
    ("G",  "minor"): (6,  "A"),
    ("D",  "minor"): (7,  "A"),
    ("A",  "minor"): (8,  "A"),
    ("E",  "minor"): (9,  "A"),
    ("B",  "minor"): (10, "A"),
    ("F#", "minor"): (11, "A"),
    ("C#", "minor"): (12, "A"),
}


# ---------------------------------------------------------------------------
# Key analysis
# ---------------------------------------------------------------------------

def get_key(filepath: str) -> tuple[int, str]:
    """Return the Camelot (number, letter) for a WAV file using Essentia.

    Args:
        filepath: Path to a WAV audio file.

    Returns:
        (number, letter) e.g. (8, "B") for C major.

    Raises:
        FileNotFoundError: File does not exist.
        ValueError: Key returned by Essentia is not in the Camelot table.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath!r}")

    audio = es.MonoLoader(filename=filepath)()
    key_name, scale, _ = es.KeyExtractor()(audio)

    # Normalise enharmonic equivalents (e.g. "Db" → "C#")
    key_name = _ENHARMONICS.get(key_name, key_name)

    camelot = _CAMELOT.get((key_name, scale))
    if camelot is None:
        raise ValueError(
            f"Unknown key from Essentia: {key_name!r} {scale!r}. "
            "Check _ENHARMONICS and _CAMELOT tables."
        )
    return camelot


def keys_compatible(c1: tuple[int, str], c2: tuple[int, str]) -> bool:
    """Return True if two Camelot positions are harmonically compatible.

    Rules (any one is sufficient):
        1. Same number + same letter      → identical key
        2. Same letter + number ±1        → adjacent on the same ring (wraps 12↔1)
        3. Same number + opposite letter  → relative major / minor pair
    """
    n1, l1 = c1
    n2, l2 = c2

    if n1 == n2 and l1 == l2:          # rule 1
        return True

    if l1 == l2:
        diff = abs(n1 - n2)
        if diff == 1 or diff == 11:     # rule 2 (diff==11 catches 1↔12 wrap)
            return True

    if n1 == n2 and l1 != l2:          # rule 3
        return True

    return False


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _ensure_stereo(y: np.ndarray) -> np.ndarray:
    """Convert mono (N,) to stereo (2, N) by duplication; leave stereo unchanged."""
    return np.stack([y, y]) if y.ndim == 1 else y


def _snap_to_phrase(start_sec: float, phrase_sec: float) -> float:
    """Return the smallest phrase boundary >= start_sec."""
    return math.ceil(start_sec / phrase_sec) * phrase_sec


def _sec_to_samp(sec: float, sr: int) -> int:
    return int(round(sec * sr))


def _split_stems(
    filepath: str, demucs_out_dir: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Run DEMUCS and return unnormalized (low, mid, high, sr) stereo arrays.

    Stems are NOT normalised so that low + mid + high ≈ original audio.
    Each stem has shape (2, N), dtype float32.
    """
    os.makedirs(demucs_out_dir, exist_ok=True)
    subprocess.run(
        ["python", "-m", "demucs", "--out", demucs_out_dir, filepath],
        check=True,
    )

    song_name = os.path.splitext(os.path.basename(filepath))[0]
    stem_dir  = os.path.join(demucs_out_dir, "htdemucs", song_name)

    bass,  sr = librosa.load(os.path.join(stem_dir, "bass.wav"),   sr=None, mono=False)
    drums, _  = librosa.load(os.path.join(stem_dir, "drums.wav"),  sr=None, mono=False)
    vox,   _  = librosa.load(os.path.join(stem_dir, "vocals.wav"), sr=None, mono=False)
    other, _  = librosa.load(os.path.join(stem_dir, "other.wav"),  sr=None, mono=False)

    low  = _ensure_stereo(bass)
    mid  = _ensure_stereo(vox + other)
    high = _ensure_stereo(drums)

    return low, mid, high, int(sr)


def _stretch_stem(stem: np.ndarray, rate: float) -> np.ndarray:
    """Time-stretch each channel independently; no-op when rate == 1.0."""
    if rate == 1.0:
        return stem
    return np.stack(
        [librosa.effects.time_stretch(ch, rate=rate) for ch in stem]
    )


def _resample_stems(
    stems: list[np.ndarray], sr_from: int, sr_to: int
) -> list[np.ndarray]:
    """Resample a list of stereo stem arrays from sr_from to sr_to."""
    if sr_from == sr_to:
        return stems
    return [
        np.stack([librosa.resample(ch, orig_sr=sr_from, target_sr=sr_to) for ch in s])
        for s in stems
    ]


def _fmt(sec: float) -> str:
    m = int(sec) // 60
    s = sec - m * 60
    return f"{m}:{s:05.2f}"


# ---------------------------------------------------------------------------
# Transition builders
# ---------------------------------------------------------------------------

def _build_tight_transition(
    y1: np.ndarray,
    low1: np.ndarray, mid1: np.ndarray, high1: np.ndarray,
    low2: np.ndarray, mid2: np.ndarray, high2: np.ndarray,
    s1_v1_start: int,
    trans_start: int,
    s2_start: int,
    phrase_samples: int,
    s2_end_sample: int,
) -> np.ndarray:
    """Chorus→chorus: 1-phrase low-swap then hard cut to Song 2.

    Layout:
        Song 1 continuous [s1_v1_start → trans_start)
        Phase A            [trans_start, trans_start + phrase_samples)
        Song 2 full        [s2_start + phrase_samples → end)
    """
    fade_out = np.linspace(1.0, 0.0, phrase_samples, dtype=np.float32)
    fade_in  = np.linspace(0.0, 1.0, phrase_samples, dtype=np.float32)

    def _sl(stem: np.ndarray, start: int) -> np.ndarray:
        return stem[:, start : start + phrase_samples]

    s1_pre = y1[:, s1_v1_start : trans_start]

    phase_a = (
        _sl(low1,  trans_start) * fade_out
        + _sl(mid1,  trans_start)               # mids held at full
        + _sl(high1, trans_start)               # highs held at full
        + _sl(low2,  s2_start)  * fade_in
    )

    s2_full  = low2 + mid2 + high2
    s2_after = s2_full[:, s2_start + phrase_samples : s2_end_sample]

    return np.concatenate([s1_pre, phase_a, s2_after], axis=1)


def _build_tight_fallback(
    y1: np.ndarray,
    low1: np.ndarray, mid1: np.ndarray, high1: np.ndarray,
    low2: np.ndarray, mid2: np.ndarray, high2: np.ndarray,
    s1_v1_start: int,
    trans_start: int,
    s2_start: int,
    phrase_samples: int,
    s2_end_sample: int,
) -> np.ndarray:
    """Short-chorus fallback: play Song 1 fully through chorus, then 1-phrase blend → Song 2.

    Used when Song 1 Chorus 1 is shorter than 2 phrases (no room for an
    internal transition). ``trans_start`` should equal ``s1_c1_end``.

    Phase A (1 phrase):
        Song 1 lows: fade 1→0  (mids+highs already off — taken out at chorus end)
        Song 2 mids+highs: fade 0→1  (Song 2 lows held off)
    After Phase A: Song 2 full (all stems, lows come in).
    """
    fade_out = np.linspace(1.0, 0.0, phrase_samples, dtype=np.float32)
    fade_in  = np.linspace(0.0, 1.0, phrase_samples, dtype=np.float32)

    def _sl(stem: np.ndarray, start: int) -> np.ndarray:
        return stem[:, start : start + phrase_samples]

    s1_pre = y1[:, s1_v1_start : trans_start]

    # Phase A: Song 1 lows fade out; Song 2 mids+highs fade in; Song 2 lows off
    phase_a = (
        _sl(low1,  trans_start) * fade_out
        + _sl(mid2,  s2_start)  * fade_in
        + _sl(high2, s2_start)  * fade_in
    )

    s2_full  = low2 + mid2 + high2
    s2_after = s2_full[:, s2_start + phrase_samples : s2_end_sample]

    return np.concatenate([s1_pre, phase_a, s2_after], axis=1)


def _build_loose_transition(
    y1: np.ndarray,
    low1: np.ndarray, mid1: np.ndarray, high1: np.ndarray,
    low2: np.ndarray, mid2: np.ndarray, high2: np.ndarray,
    s1_v1_start: int,
    trans_start: int,
    s2_start: int,
    phrase_samples: int,
    s2_end_sample: int,
) -> np.ndarray:
    """Chorus→verse→chorus: 2-phrase gradual frequency-band swap.

    Layout:
        Song 1 continuous [s1_v1_start → trans_start)
        Phase A            [trans_start, trans_start + phrase_samples)
        Phase B            [trans_start + phrase_samples, trans_start + 2*phrase_samples)
        Song 2 full        [s2_start + 2*phrase_samples → end)
    """
    fade_out = np.linspace(1.0, 0.0, phrase_samples, dtype=np.float32)
    fade_in  = np.linspace(0.0, 1.0, phrase_samples, dtype=np.float32)

    def _sl(stem: np.ndarray, start: int) -> np.ndarray:
        return stem[:, start : start + phrase_samples]

    s1_pre = y1[:, s1_v1_start : trans_start]

    # Phase A: lows swap; Song 1 mids+highs held at full
    phase_a = (
        _sl(low1,  trans_start) * fade_out
        + _sl(mid1,  trans_start)
        + _sl(high1, trans_start)
        + _sl(low2,  s2_start)  * fade_in
    )

    # Phase B: mids+highs swap; Song 2 lows already at full
    phB_s1 = trans_start + phrase_samples
    phB_s2 = s2_start    + phrase_samples
    phase_b = (
        _sl(mid1,  phB_s1) * fade_out
        + _sl(high1, phB_s1) * fade_out
        + _sl(low2,  phB_s2)
        + _sl(mid2,  phB_s2) * fade_in
        + _sl(high2, phB_s2) * fade_in
    )

    s2_full  = low2 + mid2 + high2
    s2_after = s2_full[:, s2_start + 2 * phrase_samples : s2_end_sample]

    return np.concatenate([s1_pre, phase_a, phase_b, s2_after], axis=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_transition(
    song1_path: str,
    song2_path: str,
    output_dir: str = "output",
) -> str:
    """Detect BPM + key, select tight or loose transition, build and save mix.

    Args:
        song1_path: Path to the outgoing song WAV.
        song2_path: Path to the incoming song WAV.
        output_dir: Root directory for all output files.

    Returns:
        Path to the saved mix WAV.

    Raises:
        FileNotFoundError: If either WAV does not exist.
        ValueError: If required sections cannot be detected.
    """
    for p in (song1_path, song2_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Audio file not found: {p!r}")

    # ------------------------------------------------------------------ #
    # 1. Analyse songs                                                     #
    # ------------------------------------------------------------------ #
    print("Analysing Song 1…")
    bpm1       = get_bpm(song1_path)
    key1       = get_key(song1_path)
    chorus1_ts = find_chorus(song1_path)
    verse1_ts  = find_verse(song1_path)

    print("Analysing Song 2…")
    bpm2       = get_bpm(song2_path)
    key2       = get_key(song2_path)
    chorus2_ts = find_chorus(song2_path)
    verse2_ts  = find_verse(song2_path)   # optional — used for reference save only

    # ------------------------------------------------------------------ #
    # 2. Decide transition type                                            #
    # ------------------------------------------------------------------ #
    bpm_ok  = abs(bpm1 - bpm2) <= BPM_TIGHT_THRESHOLD
    key_ok  = keys_compatible(key1, key2)
    tight   = bpm_ok and key_ok

    def _cam(c: tuple[int, str]) -> str:
        return f"{c[0]}{c[1]}"

    print(
        f"\nSong 1: {bpm1:.1f} BPM  key {_cam(key1)}\n"
        f"Song 2: {bpm2:.1f} BPM  key {_cam(key2)}\n"
        f"BPM within ±{BPM_TIGHT_THRESHOLD}: {bpm_ok}  |  "
        f"Keys compatible: {key_ok}\n"
        f"→ {'TIGHT  (chorus → chorus, hard cut)' if tight else 'LOOSE  (chorus → verse → chorus, 2-phrase fade)'}"
    )

    # ------------------------------------------------------------------ #
    # 3. Validate required sections                                        #
    # ------------------------------------------------------------------ #
    if not chorus1_ts:
        raise ValueError("Song 1: no chorus detected.")
    if not chorus2_ts:
        raise ValueError("Song 2: no chorus detected.")
    if not verse1_ts:
        raise ValueError("Song 1: no verse detected (need at least Verse 1).")
    # Loose path needs Verse 2 as its transition anchor; tight needs only Verse 1.
    if not tight and len(verse1_ts) < 2:
        raise ValueError(
            f"Song 1: loose transition needs ≥2 verse instances (found {len(verse1_ts)})."
        )
    if len(verse2_ts) < 2:
        raise ValueError(
            f"Song 2: need ≥2 verse instances to cap Song 2 content "
            f"(found {len(verse2_ts)})."
        )

    s1_v1 = verse1_ts[0]
    s1_c1 = chorus1_ts[0]
    s2_c1 = chorus2_ts[0]

    # ------------------------------------------------------------------ #
    # 4. Phrase geometry (at Song 1's BPM)                                #
    # ------------------------------------------------------------------ #
    bar_duration    = 4.0 * (60.0 / bpm1)
    phrase_duration = 8.0 * bar_duration          # 8 bars = 32 beats

    if tight:
        chorus_duration  = s1_c1[1] - s1_c1[0]
        n_chorus_phrases = int(chorus_duration / phrase_duration)
        if n_chorus_phrases >= 2:
            # Transition at the second-to-last phrase of Song 1 Chorus 1
            transition_start_sec = s1_c1[0] + (n_chorus_phrases - 2) * phrase_duration
            use_fallback = False
        else:
            # Chorus too short — transition at end of Song 1 Chorus 1
            transition_start_sec = s1_c1[1]
            use_fallback = True
        phrases_needed = 1
    else:
        s1_v2 = verse1_ts[1]
        transition_start_sec = _snap_to_phrase(s1_v2[0], phrase_duration)
        use_fallback         = False
        phrases_needed       = 2

    phase_end_sec = transition_start_sec + phrases_needed * phrase_duration

    print(
        f"\nPhrase       : {phrase_duration:.2f}s  "
        f"({int(phrase_duration / (60.0 / bpm1))} beats)\n"
        f"Trans start  : {_fmt(transition_start_sec)}  ({transition_start_sec:.2f}s) — "
        f"{'short-chorus fallback: Song 2 mids+highs fade in' if (tight and use_fallback) else 'Song 2 lows fade in'}\n"
        f"Trans end    : {_fmt(phase_end_sec)}  ({phase_end_sec:.2f}s) — Song 2 fully dominant"
    )

    # ------------------------------------------------------------------ #
    # 4. Load Song 1 original audio (stereo)                              #
    # ------------------------------------------------------------------ #
    print("\nLoading Song 1 audio…")
    y1, sr1 = librosa.load(song1_path, mono=False, sr=None)
    y1 = _ensure_stereo(y1)

    # ------------------------------------------------------------------ #
    # 5. Stem separation                                                   #
    # ------------------------------------------------------------------ #
    stems_root = os.path.join(output_dir, "stems")

    print("Running DEMUCS on Song 1…")
    low1, mid1, high1, _ = _split_stems(
        song1_path, os.path.join(stems_root, "song1")
    )

    print("Running DEMUCS on Song 2…")
    low2_raw, mid2_raw, high2_raw, sr2_stems = _split_stems(
        song2_path, os.path.join(stems_root, "song2")
    )

    # ------------------------------------------------------------------ #
    # 6. BPM matching — speed up only                                      #
    # ------------------------------------------------------------------ #
    if bpm2 < bpm1:
        stretch_rate = bpm1 / bpm2
        print(f"Stretching Song 2: {bpm2:.1f} → {bpm1:.1f} BPM  (×{stretch_rate:.4f})")
    else:
        stretch_rate = 1.0
        print(f"Song 2 BPM ({bpm2:.1f}) ≥ Song 1 ({bpm1:.1f}); no stretching.")

    low2  = _stretch_stem(low2_raw,  stretch_rate)
    mid2  = _stretch_stem(mid2_raw,  stretch_rate)
    high2 = _stretch_stem(high2_raw, stretch_rate)

    if sr2_stems != sr1:
        print(f"Resampling Song 2 stems {sr2_stems} Hz → {sr1} Hz…")
        low2, mid2, high2 = _resample_stems([low2, mid2, high2], sr2_stems, sr1)

    # ------------------------------------------------------------------ #
    # 7. Convert timestamps → sample indices                              #
    # ------------------------------------------------------------------ #
    phrase_samples = _sec_to_samp(phrase_duration, sr1)
    trans_start    = _sec_to_samp(transition_start_sec, sr1)
    s1_v1_start    = _sec_to_samp(s1_v1[0], sr1)
    s1_v1_end      = _sec_to_samp(s1_v1[1], sr1)
    s1_c1_start    = _sec_to_samp(s1_c1[0], sr1)
    s1_c1_end      = _sec_to_samp(s1_c1[1], sr1)

    s2_start  = _sec_to_samp(s2_c1[0] / stretch_rate, sr1)
    s2_v2_end = _sec_to_samp(verse2_ts[1][1] / stretch_rate, sr1)

    # ------------------------------------------------------------------ #
    # 8. Guard: check stems are long enough                               #
    # ------------------------------------------------------------------ #
    req_s1 = trans_start + phrases_needed * phrase_samples
    req_s2 = s2_start    + phrases_needed * phrase_samples

    if low1.shape[1] < req_s1:
        raise ValueError(
            f"Song 1 stems too short for the transition window "
            f"(need {req_s1} samples, have {low1.shape[1]})."
        )
    if low2.shape[1] < req_s2:
        raise ValueError(
            f"Song 2 stems too short for the transition window "
            f"(need {req_s2} samples, have {low2.shape[1]})."
        )

    # ------------------------------------------------------------------ #
    # 9. Build mix                                                         #
    # ------------------------------------------------------------------ #
    builder_kwargs = dict(
        y1=y1,
        low1=low1, mid1=mid1, high1=high1,
        low2=low2, mid2=mid2, high2=high2,
        s1_v1_start=s1_v1_start,
        trans_start=trans_start,
        s2_start=s2_start,
        phrase_samples=phrase_samples,
        s2_end_sample=s2_v2_end,
    )

    if tight and use_fallback:
        mix = _build_tight_fallback(**builder_kwargs)
    elif tight:
        mix = _build_tight_transition(**builder_kwargs)
    else:
        mix = _build_loose_transition(**builder_kwargs)

    # Peak normalise to 0.9
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix * (0.9 / peak)

    # ------------------------------------------------------------------ #
    # 10. Save outputs                                                     #
    # ------------------------------------------------------------------ #
    song1_name = os.path.splitext(os.path.basename(song1_path))[0]
    song2_name = os.path.splitext(os.path.basename(song2_path))[0]

    def _save(path: str, audio: np.ndarray, sr: int) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, audio.T, sr)
        print(f"  Saved: {path}")

    # Final mix
    mixes_dir = os.path.join(output_dir, "mixes")
    mix_path  = os.path.join(mixes_dir, f"{song1_name}_{song2_name}_mix.wav")
    _save(mix_path, mix, sr1)

    # Song 1 reference sections
    out1 = os.path.join(output_dir, "song_1")
    _save(os.path.join(out1, "verse1.wav"),  y1[:, s1_v1_start : s1_v1_end], sr1)
    _save(os.path.join(out1, "chorus1.wav"), y1[:, s1_c1_start : s1_c1_end], sr1)
    if not tight:
        _save(os.path.join(out1, "verse2.wav"), y1[:, trans_start:], sr1)

    # Song 2 reference sections (stretched)
    s2_full = low2 + mid2 + high2
    out2    = os.path.join(output_dir, "song_2")
    s2_c1_end = _sec_to_samp(s2_c1[1] / stretch_rate, sr1)
    _save(
        os.path.join(out2, "chorus1.wav"),
        s2_full[:, s2_start : s2_c1_end],
        sr1,
    )
    if verse2_ts:
        s2_v1_start = _sec_to_samp(verse2_ts[0][0] / stretch_rate, sr1)
        s2_v1_end   = min(
            _sec_to_samp(verse2_ts[0][1] / stretch_rate, sr1),
            s2_full.shape[1],
        )
        _save(os.path.join(out2, "verse1.wav"), s2_full[:, s2_v1_start : s2_v1_end], sr1)

    print(
        f"\n{'Tight' if tight else 'Loose'} transition complete.\n"
        f"Mix saved to: {mix_path}"
    )
    return mix_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python many_transitions.py <song1.wav> <song2.wav> [output_dir]")
        sys.exit(1)

    song1 = sys.argv[1]
    song2 = sys.argv[2]
    out   = sys.argv[3] if len(sys.argv) == 4 else "output"

    try:
        result = make_transition(song1, song2, output_dir=out)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}")
        sys.exit(1)
