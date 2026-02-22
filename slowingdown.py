import os
import numpy as np
import librosa
import soundfile as sf
import pyrubberband as pyrb

def gradual_slowdown(
    filepath1,
    filepath2,
    transition_end_sample,
    seconds_per_bpm=5.0,
    out_path=None,
    peak_normalize=True,
    n_points=2000
):
    """
    Uses get_bpm(filepath) (you already have it) to:
      1) Stretch all of song2 to match song1 BPM
      2) Keep it matched until transition_end_sample
      3) Then gradually return tempo back toward original song2 BPM using timemap_stretch
      4) Output = song1 + processed song2
    """

    bpm1 = float(get_bpm(filepath1))
    bpm2 = float(get_bpm(filepath2))
    if bpm1 <= 0 or bpm2 <= 0:
        raise ValueError(f"Invalid BPMs: bpm1={bpm1}, bpm2={bpm2}")

    print(f"Song1 BPM: {bpm1:.2f}")
    print(f"Song2 BPM: {bpm2:.2f}")

    bpm_diff = abs(bpm1 - bpm2)
    return_duration_s = bpm_diff * float(seconds_per_bpm)
    print(f"BPM difference: {bpm_diff:.2f} → gradual return length: {return_duration_s:.2f}s")

    # Load audio
    y1, sr1 = librosa.load(filepath1, mono=False, sr=None)
    y2, sr2 = librosa.load(filepath2, mono=False, sr=None)

    # Force (channels, samples)
    y1c = y1[np.newaxis, :] if y1.ndim == 1 else y1
    y2c = y2[np.newaxis, :] if y2.ndim == 1 else y2

    # Resample song2 to song1 sr
    if sr1 != sr2:
        y2c = np.stack([librosa.resample(ch, orig_sr=sr2, target_sr=sr1) for ch in y2c])
    sr = sr1

    # 1) Match BPM for entire song2 (Rubber Band)
    match_ratio = bpm1 / bpm2
    print(f"Stretching song2 to match by ratio {match_ratio:.6f}")
    y2_match = np.stack([pyrb.time_stretch(ch, sr, match_ratio) for ch in y2c])

    # 2) Split at transition point (in matched domain)
    transition_end_sample = int(max(0, min(transition_end_sample, y2_match.shape[-1])))
    pre  = y2_match[..., :transition_end_sample]
    post = y2_match[..., transition_end_sample:]

    if post.shape[-1] == 0:
        output = np.concatenate([y1c, y2_match], axis=-1)
        return _write_output(output, sr, filepath2, out_path, peak_normalize)

    # Decide how many samples to apply gradual return to
    slowdown_samples = min(int(return_duration_s * sr), post.shape[-1])
    slowdown_region = post[..., :slowdown_samples]
    tail            = post[..., slowdown_samples:]

    if slowdown_samples < 8:
        # too short to bother timemapping
        song2_processed = np.concatenate([pre, slowdown_region, tail], axis=-1)
        output = np.concatenate([y1c, song2_processed], axis=-1)
        return _write_output(output, sr, filepath2, out_path, peak_normalize)

    # 3) Gradual return: matched (~bpm1) -> original (~bpm2)
    # Since slowdown_region is already matched, ramp 1.0 -> bpm2/bpm1
    start_rate = 1.0
    end_rate   = bpm2 / bpm1

    n_points = int(max(32, min(n_points, slowdown_samples)))  # keep sane
    t = np.linspace(0.0, 1.0, n_points)
    rates = start_rate + (end_rate - start_rate) * (1.0 - np.cos(np.pi * t)) / 2.0

    # Build time_map in *SAMPLES* for pyrubberband validation:
    # time_map: [(in_sample, out_sample), ...]
    # Use inverse rate to accumulate output position
    in_pos = np.linspace(0.0, float(slowdown_samples), n_points)
    inv_rates = 1.0 / rates

    # expected output length in samples
    out_end = float(slowdown_samples) * float(np.mean(inv_rates))

    out_pos = np.cumsum(inv_rates)
    out_pos = out_pos / out_pos[-1] * out_end

    # Force exact endpoints required by pyrubberband
    in_pos[0], out_pos[0] = 0.0, 0.0
    in_pos[-1] = float(slowdown_samples)          # must equal len(y)
    # out_pos[-1] can be any >=0, but keep it as computed

    # Ensure strictly increasing input positions (avoid duplicates)
    eps = 1e-6
    for i in range(1, len(in_pos)):
        if in_pos[i] <= in_pos[i - 1]:
            in_pos[i] = in_pos[i - 1] + eps

    time_map = list(zip(in_pos.tolist(), out_pos.tolist()))

    # Apply timemap per channel
    stretched = np.stack([pyrb.timemap_stretch(ch, sr, time_map) for ch in slowdown_region])

    # 4) Reassemble and concat with song1
    song2_processed = np.concatenate([pre, stretched, tail], axis=-1)
    output = np.concatenate([y1c, song2_processed], axis=-1)

    return _write_output(output, sr, filepath2, out_path, peak_normalize)

def _write_output(output, sr, filepath2, out_path, peak_normalize):
    if peak_normalize:
        peak = np.max(np.abs(output))
        if peak > 0:
            output = output / peak

    if out_path is None:
        stem = os.path.splitext(os.path.basename(filepath2))[0]
        out_path = os.path.join(os.path.dirname(os.path.abspath(filepath2)), f"{stem}_slowed_back.wav")

    # output is (channels, samples) => write (samples, channels)
    sf.write(out_path, output.T, sr)
    print(f"Saved to: {out_path}")

    return out_path

gradual_slowdown(
    '/Users/siddarvind/Downloads/fameisagun.wav',
    '/Users/siddarvind/Downloads/thinkingaboutyou.wav',
    transition_end_sample=transition_end_sample,
    seconds_per_bpm=5
)
