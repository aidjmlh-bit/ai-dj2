"""loop_mix.py — Score Song 2 vocal fit over Song 1 chorus beat, then build a loop mix.

Mix structure:
    Song 1 plays from Verse 1 start → end of Chorus 1 (original audio).
    Song 1 chorus instrumental (bass + drums + other, NO vocals) is looped for
        (Song 2 stretched chorus duration + phrases_needed × phrase_duration).
    Song 2 chorus vocals (time-stretched to Song 1 BPM) are layered over the loop
        for exactly the stretched chorus duration, then silence.
    Transition phases (tight = 1 phrase, loose = 2 phrases) fade from Song 1 loop
        into Song 2 verse (full stems).
    Song 2 verse continues to end of its second verse instance.

Tight vs loose: key compatibility (Camelot wheel) — BPMs always match post-stretch.

Fit scoring (informational, printed before mix is built):
    accent    — syllable/accent alignment between Song 2 vocals and Song 1 groove
    timing    — microtiming: how consistently vocal onsets hit beat subdivisions
    contour   — pitch-movement vs beat accent coincidence
    voc_ref   — Song 2 vocal similarity to Song 1 vocal reference
    final     — weighted combination (0.4/0.25/0.15/0.20)
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import librosa
import numpy as np
import soundfile as sf
from scipy.stats import pearsonr

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "sections"))

from get_bpm import get_bpm
from get_chorus import find_chorus
from get_verse import find_verse
from many_transitions import (
    BPM_TIGHT_THRESHOLD,
    _ensure_stereo,
    _fmt,
    _resample_stems,
    _sec_to_samp,
    _snap_to_phrase,
    _stretch_stem,
    get_key,
    keys_compatible,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOP_LENGTH      = 512       # frames for onset / pitch analysis
_SIGMA_MS       = 50.0      # microtiming σ threshold (ms)
_MU_MS          = 30.0      # microtiming mean-drift threshold (ms)
_XFADE_SAMP     = 512       # crossfade window at loop boundaries (samples)
_TRANS_FADE_SEC = 5         # Song 1 fade-out duration (seconds) when Song 2 verse starts


# ---------------------------------------------------------------------------
# Stem loading — 4 separate stems
# ---------------------------------------------------------------------------

def _split_stems_4(
    filepath: str, demucs_out_dir: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Run DEMUCS and return (bass, drums, vox, other, sr) as stereo (2, N) arrays.

    Unlike _split_stems in many_transitions, stems are NOT pre-combined so the
    caller can freely form instrumental = bass + drums + other (no vocals).
    """
    song_name = os.path.splitext(os.path.basename(filepath))[0]
    stem_dir  = os.path.join(demucs_out_dir, "htdemucs", song_name)
    _STEM_FILES = ("bass.wav", "drums.wav", "vocals.wav", "other.wav")

    # Check the requested output dir *and* every sibling slot directory (song1/,
    # song2/, …) so we never re-run DEMUCS when stems were produced under a
    # different slot in a previous run.
    stems_parent = os.path.dirname(demucs_out_dir)
    candidates   = [demucs_out_dir]
    if os.path.isdir(stems_parent):
        candidates += [
            os.path.join(stems_parent, d)
            for d in os.listdir(stems_parent)
            if os.path.isdir(os.path.join(stems_parent, d))
            and os.path.join(stems_parent, d) != demucs_out_dir
        ]

    stem_dir = None
    for cand in candidates:
        cand_stem_dir = os.path.join(cand, "htdemucs", song_name)
        if all(os.path.exists(os.path.join(cand_stem_dir, f)) for f in _STEM_FILES):
            stem_dir = cand_stem_dir
            print(f"  Stems already exist for '{song_name}' in '{cand}'; skipping DEMUCS.")
            break

    if stem_dir is None:
        stem_dir = os.path.join(demucs_out_dir, "htdemucs", song_name)
        os.makedirs(demucs_out_dir, exist_ok=True)
        subprocess.run(
            ["python", "-m", "demucs", "--out", demucs_out_dir, filepath],
            check=True,
        )

    bass,  sr = librosa.load(os.path.join(stem_dir, "bass.wav"),   sr=None, mono=False)
    drums, _  = librosa.load(os.path.join(stem_dir, "drums.wav"),  sr=None, mono=False)
    vox,   _  = librosa.load(os.path.join(stem_dir, "vocals.wav"), sr=None, mono=False)
    other, _  = librosa.load(os.path.join(stem_dir, "other.wav"),  sr=None, mono=False)

    return (
        _ensure_stereo(bass),
        _ensure_stereo(drums),
        _ensure_stereo(vox),
        _ensure_stereo(other),
        int(sr),
    )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _accent_curve(y_mono: np.ndarray, sr: int, hop: int = HOP_LENGTH) -> np.ndarray:
    """Onset strength envelope from a mono instrumental signal."""
    env = librosa.onset.onset_strength(y=y_mono, sr=sr, hop_length=hop)
    if env.max() > 0:
        env = env / env.max()
    return env.astype(np.float32)


def _vocal_rhythm_curve(y_vox_mono: np.ndarray, sr: int, hop: int = HOP_LENGTH) -> np.ndarray:
    """Onset strength envelope from a mono vocal stem (syllable emphasis proxy)."""
    env = librosa.onset.onset_strength(y=y_vox_mono, sr=sr, hop_length=hop)
    if env.max() > 0:
        env = env / env.max()
    return env.astype(np.float32)


def _pitch_contour(
    y_vox_mono: np.ndarray, sr: int, hop: int = HOP_LENGTH
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate F0 via pYIN, fill short gaps, return (f0_semitones, voiced_flag, d_f0).

    d_f0 = |delta F0 per frame| (pitch velocity, used for contour-accent metric).
    """
    f0, voiced_flag, _ = librosa.pyin(
        y_vox_mono,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=hop,
    )

    # Convert Hz → semitones (log scale); NaN where unvoiced
    with np.errstate(divide="ignore", invalid="ignore"):
        f0_st = np.where(voiced_flag, 12.0 * np.log2(np.where(f0 > 0, f0, 1.0) / 440.0), np.nan)

    # Fill short gaps (linear interpolation across up to 4 unvoiced frames)
    nans = np.isnan(f0_st)
    if nans.any() and (~nans).any():
        idx = np.arange(len(f0_st))
        f0_st = np.interp(idx, idx[~nans], f0_st[~nans])

    # Pitch velocity (absolute frame-to-frame difference)
    d_f0 = np.abs(np.diff(f0_st, prepend=f0_st[0])).astype(np.float32)
    if d_f0.max() > 0:
        d_f0 = d_f0 / d_f0.max()

    return f0_st.astype(np.float32), voiced_flag, d_f0


def _beat_emphasis_template(
    y_instr_mono: np.ndarray, sr: int, bpm: float, hop: int = HOP_LENGTH
) -> np.ndarray:
    """Build a per-frame expected emphasis curve from Song 1 groove.

    For each beat subdivision frame, average the accent curve values across
    all bars to produce a repeating emphasis template, then tile it to match
    the signal length.
    """
    accent  = librosa.onset.onset_strength(y=y_instr_mono, sr=sr, hop_length=hop)
    _, beats = librosa.beat.beat_track(y=y_instr_mono, sr=sr, hop_length=hop, bpm=bpm)

    n_frames     = len(accent)
    frames_per_beat = max(1, int(round(60.0 / bpm * sr / hop)))
    frames_per_bar  = 4 * frames_per_beat

    # Average accent over each subdivision offset within the bar
    template = np.zeros(frames_per_bar, dtype=np.float32)
    counts   = np.zeros(frames_per_bar, dtype=np.float32)
    for i in range(n_frames):
        slot = i % frames_per_bar
        template[slot] += accent[i]
        counts[slot]   += 1
    with np.errstate(invalid="ignore"):
        template = np.where(counts > 0, template / counts, 0.0)
    if template.max() > 0:
        template /= template.max()

    # Tile to full signal length
    n_reps   = math.ceil(n_frames / frames_per_bar) + 1
    full     = np.tile(template, n_reps)[:n_frames]
    return full.astype(np.float32)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r clipped to [0, 1] (negative correlation = 0 fit)."""
    min_len = min(len(a), len(b))
    if min_len < 2:
        return 0.0
    r, _ = pearsonr(a[:min_len], b[:min_len])
    return float(np.clip(r, 0.0, 1.0))


def score_vocal_fit(
    s1_instr_mono: np.ndarray,
    s1_vox_mono:   np.ndarray,
    s2_vox_mono:   np.ndarray,
    sr:            int,
    bpm:           float,
    hop:           int = HOP_LENGTH,
    use_dtw:       bool = False,
) -> dict[str, float]:
    """Compute and print a vocal fit score dict.

    Keys: accent, timing, contour, voc_ref, final.
    All scores in [0, 1]; higher = better fit.

    Args:
        s1_instr_mono: Song 1 chorus instrumental (mono, no vocals).
        s1_vox_mono:   Song 1 chorus vocals (mono) — gold reference.
        s2_vox_mono:   Song 2 chorus vocals (mono, already time-stretched to Song 1 BPM).
        sr:            Sample rate (common).
        bpm:           Song 1 BPM (target grid).
        hop:           Analysis hop length in samples.
        use_dtw:       If True, use DTW-aligned correlation for voc_ref (slower).
    """
    # ── Signals ──────────────────────────────────────────────────────────
    emphasis   = _beat_emphasis_template(s1_instr_mono, sr, bpm, hop)
    accent1    = _accent_curve(s1_instr_mono, sr, hop)
    v2_rhythm  = _vocal_rhythm_curve(s2_vox_mono, sr, hop)
    v1_rhythm  = _vocal_rhythm_curve(s1_vox_mono, sr, hop)
    _, _, d_f2 = _pitch_contour(s2_vox_mono, sr, hop)

    # ── Metric 1: syllable / accent alignment ────────────────────────────
    score_accent = _safe_corr(v2_rhythm, emphasis)

    # ── Metric 2: microtiming ────────────────────────────────────────────
    # Detect vocal onset times (seconds)
    onset_frames = librosa.onset.onset_detect(
        y=s2_vox_mono, sr=sr, hop_length=hop, units="frames"
    )
    onset_times  = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)

    beat_period  = 60.0 / bpm          # seconds per beat
    subdiv       = beat_period / 2.0   # eighth-note grid

    if len(onset_times) >= 2:
        # Offset of each vocal onset from nearest subdivision
        offsets = np.array([
            t % subdiv - subdiv / 2 for t in onset_times
        ])                             # centred on [-subdiv/2, +subdiv/2]
        sigma_s = _SIGMA_MS / 1000.0
        mu_s    = _MU_MS    / 1000.0
        score_timing = (
            math.exp(-(np.std(offsets) ** 2) / (sigma_s ** 2))
            * math.exp(-(np.mean(offsets) ** 2) / (mu_s ** 2))
        )
    else:
        score_timing = 0.0

    # ── Metric 3: pitch-movement vs beat accents ─────────────────────────
    score_contour = _safe_corr(d_f2, emphasis)

    # ── Step 5: Song 1 vocal reference ───────────────────────────────────
    if use_dtw:
        # DTW-aligned correlation (import only when needed)
        from librosa.sequence import dtw as _dtw
        min_len = min(len(v2_rhythm), len(v1_rhythm))
        D, wp   = _dtw(v2_rhythm[:min_len].reshape(1, -1),
                       v1_rhythm[:min_len].reshape(1, -1))
        v2_aligned = v2_rhythm[wp[:, 0]]
        v1_aligned = v1_rhythm[wp[:, 1]]
        score_voc_ref = _safe_corr(v2_aligned, v1_aligned)
    else:
        score_voc_ref = _safe_corr(v2_rhythm, v1_rhythm)

    # ── Final weighted score ─────────────────────────────────────────────
    final = (
        0.40 * score_accent
        + 0.25 * score_timing
        + 0.15 * score_contour
        + 0.20 * score_voc_ref
    )

    scores = {
        "accent":  round(score_accent,   3),
        "timing":  round(score_timing,   3),
        "contour": round(score_contour,  3),
        "voc_ref": round(score_voc_ref,  3),
        "final":   round(final,          3),
    }

    print(
        f"\n── Vocal Fit Scores ────────────────────────────────\n"
        f"  Syllable/Accent alignment : {scores['accent']:.3f}\n"
        f"  Microtiming consistency   : {scores['timing']:.3f}\n"
        f"  Pitch-movement vs accents : {scores['contour']:.3f}\n"
        f"  Song 1 vocal reference    : {scores['voc_ref']:.3f}\n"
        f"  ── Final (weighted)       : {scores['final']:.3f}  "
        f"({'good fit' if final >= 0.5 else 'weak fit — proceed with caution'})\n"
        f"────────────────────────────────────────────────────\n"
    )
    return scores


# ---------------------------------------------------------------------------
# Loop helpers
# ---------------------------------------------------------------------------

def _loop_to_duration(
    stem: np.ndarray,
    bar_samp: int,
    target_samp: int,
) -> np.ndarray:
    """Tile stem (2, N) to exactly target_samp, snapped to bar boundaries.

    Applies a short crossfade (_XFADE_SAMP ramps) at each loop point to
    prevent phase-discontinuity clicks.
    """
    seg_len = stem.shape[1]
    # Snap to nearest complete bar to guarantee a clean loop point
    bars    = max(1, seg_len // bar_samp)
    seg_len = bars * bar_samp
    seg     = stem[:, :seg_len]

    n_reps  = math.ceil(target_samp / seg_len) + 1
    # Allocate with extra margin for crossfade
    out     = np.tile(seg, (1, n_reps))[:, : target_samp + _XFADE_SAMP].copy()

    xf       = _XFADE_SAMP
    ramp_out = np.linspace(1.0, 0.0, xf, dtype=np.float32)
    ramp_in  = np.linspace(0.0, 1.0, xf, dtype=np.float32)

    for k in range(1, n_reps):
        idx = k * seg_len
        if idx >= xf and idx + xf <= out.shape[1]:
            out[:, idx - xf : idx] *= ramp_out
            out[:, idx      : idx + xf] *= ramp_in

    return out[:, :target_samp]


# ---------------------------------------------------------------------------
# Transition builder (loop → Song 2 verse)
# ---------------------------------------------------------------------------

def _build_loop_transition(
    loop_low:         np.ndarray,   # Song 1 loop — bass
    loop_mid:         np.ndarray,   # Song 1 loop — other
    loop_high:        np.ndarray,   # Song 1 loop — drums
    s2v_vox:          np.ndarray,   # Song 2 verse vocals — held at full
    s2v_bass:         np.ndarray,   # Song 2 verse bass   — full from bar 1
    s2v_other:        np.ndarray,   # Song 2 verse other  — full from bar 1
    s2v_drums:        np.ndarray,   # Song 2 verse drums  — full from bar 1
    loop_trans_start: int,
    fade_samp:        int,
) -> np.ndarray:
    """Song 2 verse starts at full; Song 1 loop fades to silence over fade_samp samples.

    After fade_samp samples Song 1 is completely gone — hard cut.
    Song 2 verse stems (vox + instrumental) play at full throughout.
    """
    d        = loop_trans_start
    fade_out = np.linspace(1.0, 0.0, fade_samp, dtype=np.float32)

    def _sl_s2v(stem: np.ndarray) -> np.ndarray:
        chunk = stem[:, :fade_samp]
        if chunk.shape[1] < fade_samp:
            pad   = np.zeros((2, fade_samp - chunk.shape[1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=1)
        return chunk

    return (
        loop_low[:,  d : d + fade_samp] * fade_out
        + loop_mid[:,  d : d + fade_samp] * fade_out
        + loop_high[:, d : d + fade_samp] * fade_out
        + _sl_s2v(s2v_vox)
        + _sl_s2v(s2v_bass)
        + _sl_s2v(s2v_other)
        + _sl_s2v(s2v_drums)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_loop_mix(
    song1_path: str,
    song2_path: str,
    output_dir: str = "output",
) -> str:
    """Score vocal fit then build a loop-mix WAV.

    Returns:
        Path to the saved mix WAV.
    """
    for p in (song1_path, song2_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Audio file not found: {p!r}")

    # ------------------------------------------------------------------ #
    # 1. Analyse                                                           #
    # ------------------------------------------------------------------ #
    print("Analysing Song 1…")
    bpm1        = get_bpm(song1_path)
    key1        = get_key(song1_path)
    chorus1_ts  = find_chorus(song1_path)
    verse1_ts   = find_verse(song1_path)

    print("Analysing Song 2…")
    bpm2        = get_bpm(song2_path)
    key2        = get_key(song2_path)
    chorus2_ts  = find_chorus(song2_path)
    verse2_ts   = find_verse(song2_path)

    # ------------------------------------------------------------------ #
    # 2. Validate                                                          #
    # ------------------------------------------------------------------ #
    if not chorus1_ts:
        raise ValueError("Song 1: no chorus detected.")
    if not verse1_ts:
        raise ValueError("Song 1: no verse detected.")
    if not chorus2_ts:
        raise ValueError("Song 2: no chorus detected.")
    if len(verse2_ts) < 2:
        raise ValueError(
            f"Song 2: need ≥2 verse instances to cap Song 2 content "
            f"(found {len(verse2_ts)})."
        )

    # Find the verse in Song 2 that starts AFTER its Chorus 1 end
    s2_c1_end_orig = chorus2_ts[0][1]
    verse_after_ch = next(
        (ts for ts in verse2_ts if ts[0] > s2_c1_end_orig), None
    )
    if verse_after_ch is None:
        raise ValueError(
            "Song 2: no verse instance found after Chorus 1 end "
            f"({_fmt(s2_c1_end_orig)}). Cannot build loop mix."
        )

    # ------------------------------------------------------------------ #
    # 3. Tight vs loose (key compatibility; BPMs will match after stretch) #
    # ------------------------------------------------------------------ #
    key_ok = keys_compatible(key1, key2)
    tight  = key_ok
    phrases_needed = 1 if tight else 2

    def _cam(c: tuple[int, str]) -> str:
        return f"{c[0]}{c[1]}"

    print(
        f"\nSong 1: {bpm1:.1f} BPM  key {_cam(key1)}\n"
        f"Song 2: {bpm2:.1f} BPM  key {_cam(key2)}\n"
        f"Keys compatible: {key_ok}  → "
        f"{'TIGHT (1-phrase transition)' if tight else 'LOOSE (2-phrase transition)'}"
    )

    # ------------------------------------------------------------------ #
    # 4. Phrase geometry                                                   #
    # ------------------------------------------------------------------ #
    bar_dur    = 4.0 * (60.0 / bpm1)
    phrase_dur = 8.0 * bar_dur

    # ------------------------------------------------------------------ #
    # 5. Load Song 1 original audio                                        #
    # ------------------------------------------------------------------ #
    print("\nLoading Song 1 audio…")
    y1, sr1 = librosa.load(song1_path, mono=False, sr=None)
    y1 = _ensure_stereo(y1)

    # ------------------------------------------------------------------ #
    # 6. DEMUCS stem separation                                           #
    # ------------------------------------------------------------------ #
    stems_root = os.path.join(output_dir, "stems")

    print("Running DEMUCS on Song 1…")
    bass1, drums1, vox1, other1, _ = _split_stems_4(
        song1_path, os.path.join(stems_root, "song1")
    )

    print("Running DEMUCS on Song 2…")
    bass2r, drums2r, vox2r, other2r, sr2 = _split_stems_4(
        song2_path, os.path.join(stems_root, "song2")
    )

    # ------------------------------------------------------------------ #
    # 7. BPM matching — speed up Song 2 only                             #
    # ------------------------------------------------------------------ #
    if bpm2 < bpm1:
        stretch_rate = bpm1 / bpm2
        print(f"Stretching Song 2: {bpm2:.1f} → {bpm1:.1f} BPM  (×{stretch_rate:.4f})")
    else:
        stretch_rate = 1.0
        print(f"Song 2 BPM ({bpm2:.1f}) ≥ Song 1 ({bpm1:.1f}); no stretching.")

    bass2  = _stretch_stem(bass2r,  stretch_rate)
    drums2 = _stretch_stem(drums2r, stretch_rate)
    vox2   = _stretch_stem(vox2r,   stretch_rate)
    other2 = _stretch_stem(other2r, stretch_rate)

    if sr2 != sr1:
        print(f"Resampling Song 2 stems {sr2} Hz → {sr1} Hz…")
        bass2, drums2, vox2, other2 = _resample_stems(
            [bass2, drums2, vox2, other2], sr2, sr1
        )

    # ------------------------------------------------------------------ #
    # 8. Sample-index conversions                                         #
    # ------------------------------------------------------------------ #
    bar_samp    = _sec_to_samp(bar_dur,    sr1)
    phrase_samp = _sec_to_samp(phrase_dur, sr1)

    s1_v1_start = _sec_to_samp(verse1_ts[0][0],  sr1)
    s1_c1_start = _sec_to_samp(chorus1_ts[0][0], sr1)
    s1_c1_end   = _sec_to_samp(chorus1_ts[0][1], sr1)

    # Song 2 in stretched domain
    s2_c1_start     = _sec_to_samp(chorus2_ts[0][0] / stretch_rate, sr1)
    s2_c1_end       = _sec_to_samp(chorus2_ts[0][1] / stretch_rate, sr1)
    d2_chorus_samp  = s2_c1_end - s2_c1_start

    s2_verse_ach_start = _sec_to_samp(verse_after_ch[0] / stretch_rate, sr1)
    s2_v2_end          = _sec_to_samp(verse2_ts[1][1]   / stretch_rate, sr1)

    print(
        f"\nPhrase       : {phrase_dur:.2f}s\n"
        f"S1 chorus    : {_fmt(chorus1_ts[0][0])} → {_fmt(chorus1_ts[0][1])}\n"
        f"S2 chorus    : {_fmt(chorus2_ts[0][0])} → {_fmt(chorus2_ts[0][1])}"
        f"  (stretched: {d2_chorus_samp / sr1:.2f}s)\n"
        f"S2 verse(ach): {_fmt(verse_after_ch[0])} → {_fmt(verse_after_ch[1])}\n"
        f"S2 verse2 end: {_fmt(verse2_ts[1][1])}"
    )

    # ------------------------------------------------------------------ #
    # 9. Score vocal fit (informational)                                  #
    # ------------------------------------------------------------------ #
    print("\nScoring vocal fit…")
    s1_instr_mono = (bass1 + drums1 + other1)[0, s1_c1_start : s1_c1_end]
    s1_vox_mono   = vox1[0, s1_c1_start : s1_c1_end]
    s2_vox_mono   = vox2[0, s2_c1_start : s2_c1_end]
    score_vocal_fit(s1_instr_mono, s1_vox_mono, s2_vox_mono, sr1, bpm1)

    # ------------------------------------------------------------------ #
    # 10. Build Part 1 — Song 1 up to end of Chorus 1                    #
    # ------------------------------------------------------------------ #
    s1_pre = y1[:, s1_v1_start : s1_c1_end]

    # ------------------------------------------------------------------ #
    # 11. Build loop instrumental (Song 1 chorus, no vocals)              #
    # ------------------------------------------------------------------ #
    trans_fade_samp   = int(_TRANS_FADE_SEC * sr1)
    total_loop_samp   = d2_chorus_samp + trans_fade_samp
    s1_chorus_seg     = (bass1 + drums1 + other1)[:, s1_c1_start : s1_c1_end]

    loop_all  = _loop_to_duration(s1_chorus_seg, bar_samp, total_loop_samp)
    loop_bass = _loop_to_duration(bass1[:,  s1_c1_start:s1_c1_end], bar_samp, total_loop_samp)
    loop_mid  = _loop_to_duration(other1[:, s1_c1_start:s1_c1_end], bar_samp, total_loop_samp)
    loop_high = _loop_to_duration(drums1[:, s1_c1_start:s1_c1_end], bar_samp, total_loop_samp)

    # ------------------------------------------------------------------ #
    # 12. Build vocal overlay (Song 2 chorus vocals)                      #
    # ------------------------------------------------------------------ #
    vox2_chorus = vox2[:, s2_c1_start : s2_c1_end]  # shape (2, d2_chorus_samp)

    # Clip in case of rounding differences
    actual_vox_len = min(vox2_chorus.shape[1], total_loop_samp)
    vox_overlay    = np.zeros((2, total_loop_samp), dtype=np.float32)
    vox_overlay[:, :actual_vox_len] = vox2_chorus[:, :actual_vox_len]

    # Composite = loop instrumental + vocal overlay
    part2 = loop_all + vox_overlay

    # ------------------------------------------------------------------ #
    # 13. Clamp Song 2 end to actual stem length                         #
    # ------------------------------------------------------------------ #
    s2_stem_len = min(bass2.shape[1], drums2.shape[1], vox2.shape[1], other2.shape[1])
    s2_v2_end   = min(s2_v2_end, s2_stem_len)

    if s2_verse_ach_start + trans_fade_samp > s2_stem_len:
        print(
            f"[loop_mix] Song 2 verse shorter than {_TRANS_FADE_SEC}s fade window "
            f"({s2_stem_len - s2_verse_ach_start} samples available); "
            "transition tail will be zero-padded."
        )

    # ------------------------------------------------------------------ #
    # 14. Build transition (loop beat → Song 2 verse)                    #
    # ------------------------------------------------------------------ #
    s2v_vox   = vox2[:,   s2_verse_ach_start:]
    s2v_bass  = bass2[:,  s2_verse_ach_start:]
    s2v_other = other2[:, s2_verse_ach_start:]
    s2v_drums = drums2[:, s2_verse_ach_start:]

    trans = _build_loop_transition(
        loop_bass, loop_mid, loop_high,
        s2v_vox, s2v_bass, s2v_other, s2v_drums,
        loop_trans_start=d2_chorus_samp,
        fade_samp=trans_fade_samp,
    )

    # ------------------------------------------------------------------ #
    # 15. Build Song 2 verse tail                                         #
    # ------------------------------------------------------------------ #
    s2_verse_full     = bass2 + drums2 + vox2 + other2
    verse_tail_start  = s2_verse_ach_start + trans_fade_samp
    s2_tail           = s2_verse_full[:, verse_tail_start : s2_v2_end]

    # ------------------------------------------------------------------ #
    # 16. Assemble and normalise                                           #
    # ------------------------------------------------------------------ #
    pieces = [s1_pre, part2[:, :d2_chorus_samp], trans]
    if s2_tail.shape[1] > 0:
        pieces.append(s2_tail)
    mix = np.concatenate(pieces, axis=1)

    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix * (0.9 / peak)

    # ------------------------------------------------------------------ #
    # 17. Save                                                            #
    # ------------------------------------------------------------------ #
    song1_name = os.path.splitext(os.path.basename(song1_path))[0]
    song2_name = os.path.splitext(os.path.basename(song2_path))[0]

    def _save(path: str, audio: np.ndarray, sr: int) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, audio.T, sr)
        print(f"  Saved: {path}")

    mixes_dir = os.path.join(output_dir, "mixes")
    mix_path  = os.path.join(mixes_dir, f"{song1_name}_{song2_name}_loop_mix.wav")
    _save(mix_path, mix, sr1)

    print(
        f"\n{'Tight' if tight else 'Loose'} loop mix complete.\n"
        f"Mix saved to: {mix_path}"
    )
    return mix_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python loop_mix.py <song1.wav> <song2.wav> [output_dir]")
        sys.exit(1)

    song1 = sys.argv[1]
    song2 = sys.argv[2]
    out   = sys.argv[3] if len(sys.argv) == 4 else "output"

    try:
        result = build_loop_mix(song1, song2, output_dir=out)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}")
        sys.exit(1)
