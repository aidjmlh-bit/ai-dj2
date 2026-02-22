# ai-dj2
back n better #hackher413


Detect chorus / verse
Pick transition boundary (bar-aligned)
Match BPMs
Apply transition

(audio_env) alishasrivastava@vl965-172-31-115-227 ai-dj2 % python dj_mix.py "/Users/alishasrivastava/ai-dj2/output/song_1/08 In The Rain.wav" "/Users/alishasrivastava/ai-dj2/output/song_2/04 Aquamarine.wav"
Analysing songs…

  Song 1: 119.0 BPM  key 6A
  Song 2: 111.8 BPM  key 5A
  BPM diff: 7.2  |  Keys compatible: True

Computing transition timestamp for filename…
/opt/homebrew/Caskroom/miniconda/base/envs/audio_env/lib/python3.8/site-packages/librosa/core/spectrum.py:266: UserWarning: n_fft=1024 is too large for input signal of length=696
  warnings.warn(

────────────────────────────────────────────────────
  Mode       : LOOP
  Transition : 1:12
────────────────────────────────────────────────────

Analysing Song 1…
Analysing Song 2…
/opt/homebrew/Caskroom/miniconda/base/envs/audio_env/lib/python3.8/site-packages/librosa/core/spectrum.py:266: UserWarning: n_fft=1024 is too large for input signal of length=740
  warnings.warn(

Song 1: 119.0 BPM  key 6A
Song 2: 111.8 BPM  key 5A
Keys compatible: True  → TIGHT (1-phrase transition)

Loading Song 1 audio…
Running DEMUCS on Song 1…
  Stems already exist for '08 In The Rain' in 'output/stems/song1'; skipping DEMUCS.
Running DEMUCS on Song 2…
  Stems already exist for '04 Aquamarine' in 'output/stems/song2'; skipping DEMUCS.
Stretching Song 2: 111.8 → 119.0 BPM  (×1.0645)

Phrase       : 16.14s
S1 chorus    : 1:04.56 → 1:12.63
S2 chorus    : 0:55.84 → 1:10.87  (stretched: 14.12s)
S2 verse(ach): 1:13.02 → 1:25.91
S2 verse2 end: 1:25.91

Scoring vocal fit…

── Vocal Fit Scores ────────────────────────────────
  Syllable/Accent alignment : 0.044
  Microtiming consistency   : 0.203
  Pitch-movement vs accents : 0.020
  Song 1 vocal reference    : 0.029
  ── Final (weighted)       : 0.077  (weak fit — proceed with caution)
────────────────────────────────────────────────────

  Saved: output/mixes/08 In The Rain_04 Aquamarine_loop_mix.wav

Tight loop mix complete.
Mix saved to: output/mixes/08 In The Rain_04 Aquamarine_loop_mix.wav

Final mix → output/mixes/08 In The Rain_04 Aquamarine_loop_t1m12s.wav
