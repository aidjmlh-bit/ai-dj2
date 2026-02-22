import pyrubberband as pyrb
import numpy as np
import librosa
import soundfile as sf
from scipy.io.wavfile import write as wav_write
import os

def full_mix(filepath1, filepath2, transition_start, output_path, seconds_per_bpm=5):
    """
    Full pipeline:
    1. Detect BPMs for both songs
    2. Stretch song2 transition portion to match song1's BPM
    3. Gradually slow song2 back to its natural BPM after transition
    4. Combine song1 + transition + rest of song2 into one file

    Args:
        filepath1:        first song
        filepath2:        second song
        transition_start: second where transition begins in song2
        output_path:      where to save final mix
        seconds_per_bpm:  controls how long the slowdown takes per BPM difference
    """

    # --- 1. detect BPMs ---
    y1_mono, sr1 = librosa.load(filepath1, sr=None)
    y2_mono, sr2 = librosa.load(filepath2, sr=None)

    bpm1 = float(librosa.beat.beat_track(y=y1_mono, sr=sr1, start_bpm=128)[0])
    bpm2 = float(librosa.beat.beat_track(y=y2_mono, sr=sr2, start_bpm=128)[0])
    if bpm1 < 100:
        bpm1 *= 2
    if bpm2 < 100:
        bpm2 *= 2
    print(f"Song 1 BPM: {bpm1:.1f}")
    print(f"Song 2 BPM: {bpm2:.1f}")

    bpm_diff         = abs(bpm1 - bpm2)
    seconds_per_bpm  = 5 + (bpm_diff * 0.3)
    slowdown_duration = bpm_diff * seconds_per_bpm
    curve_tension    = 1.0 / (1.0 + bpm_diff * 0.1)
    print(f"BPM difference: {bpm_diff:.1f} — slowdown duration: {slowdown_duration:.1f}s")

    # --- 2. load stereo ---
    y1, sr1 = librosa.load(filepath1, sr=None, mono=False)
    y2, sr2 = librosa.load(filepath2, sr=None, mono=False)

    if sr1 != sr2:
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
    sr = sr1

    transition_sample = int(transition_start * sr)

    # split song2 into transition and post portions
    transition = y2[..., :transition_sample]
    post       = y2[..., transition_sample:]

    # --- 3. stretch transition portion to match song1's BPM ---
    rate = bpm1 / bpm2
    print(f"Stretching transition by rate {rate:.4f} to match {bpm1:.1f} BPM")
    if transition.ndim == 1:
        transition_stretched = librosa.effects.time_stretch(transition, rate=rate)
    else:
        transition_stretched = np.stack([
            librosa.effects.time_stretch(ch, rate=rate) for ch in transition
        ])

    # --- 4. gradual slowdown on post portion ---
    slowdown_samples = min(int(slowdown_duration * sr), post.shape[-1])

    start_rate = bpm1 / bpm2
    end_rate   = 1.0
    n_points   = 1000
    t          = np.linspace(0, 1, n_points)
    rates      = start_rate + (end_rate - start_rate) * (1 - np.cos(t * np.pi * curve_tension)) / 2

    input_samples  = np.linspace(0, slowdown_samples, n_points)
    inv_rates      = 1.0 / rates
    output_samples = np.cumsum(inv_rates) / np.sum(inv_rates) * slowdown_samples
    time_map       = [(int(i), int(o)) for i, o in zip(input_samples, output_samples)]

    slowdown_region = post[..., :slowdown_samples]
    tail            = post[..., slowdown_samples:]

    print("Applying gradual slowdown...")
    if slowdown_region.ndim == 1:
        slowed = pyrb.timemap_stretch(slowdown_region, sr, time_map)
    else:
        slowed = np.stack([
            pyrb.timemap_stretch(ch, sr, time_map) for ch in slowdown_region
        ])

    # --- 5. combine everything ---
    print("Combining song1 + transition + slowed post + tail...")
    post_processed = np.concatenate([slowed, tail], axis=-1)
    output = np.concatenate([y1, transition_stretched, post_processed], axis=-1)
    output = output / np.max(np.abs(output))

    if output.ndim == 1:
        sf.write(output_path, output, sr)
    else:
        sf.write(output_path, output.T, sr)

    print(f"Saved to: {output_path}")
    return output_path


# Usage
full_mix(
    '/Users/siddarvind/Downloads/fameisagun.wav',
    '/Users/siddarvind/Downloads/thinkingaboutyou.wav',
    transition_start=41.0,          # seconds into song2 where transition begins
    output_path='/Users/siddarvind/Downloads/full_mix.wav',
    seconds_per_bpm=5
)
