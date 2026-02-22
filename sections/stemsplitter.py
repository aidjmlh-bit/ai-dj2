import subprocess
import os
import librosa
import numpy as np
from scipy.io.wavfile import write as wav_write

def demucs_hml(filepath, output_dir='separated'):
    # run demucs
    subprocess.run([
        'python', '-m', 'demucs',
        '--out', output_dir,
        filepath
    ])

    song_name = os.path.splitext(os.path.basename(filepath))[0]
    stem_dir  = os.path.join(output_dir, 'htdemucs', song_name)

    # load stems
    bass,  sr = librosa.load(os.path.join(stem_dir, 'bass.wav'),  sr=None, mono=False)
    drums, _  = librosa.load(os.path.join(stem_dir, 'drums.wav'), sr=None, mono=False)
    vox,   _  = librosa.load(os.path.join(stem_dir, 'vocals.wav'),sr=None, mono=False)
    other, _  = librosa.load(os.path.join(stem_dir, 'other.wav'), sr=None, mono=False)

    # combine into high/mid/low
    low  = bass
    mid  = vox + other
    high = drums

    # normalize
    low  = low  / np.max(np.abs(low))
    mid  = mid  / np.max(np.abs(mid))
    high = high / np.max(np.abs(high))

    wav_write('low.wav',  sr, low.T.astype(np.float32))
    wav_write('mid.wav',  sr, mid.T.astype(np.float32))
    wav_write('high.wav', sr, high.T.astype(np.float32))
    
    # save individual stems too
    vox_norm = vox / np.max(np.abs(vox))
    wav_write('vocals.wav', sr, vox_norm.T.astype(np.float32))
    wav_write('drums.wav',  sr, (drums / np.max(np.abs(drums))).T.astype(np.float32))
    wav_write('bass.wav',   sr, (bass  / np.max(np.abs(bass))).T.astype(np.float32))
    wav_write('other.wav',  sr, (other / np.max(np.abs(other))).T.astype(np.float32))

    print("Saved low.wav, mid.wav, high.wav, vocals.wav, drums.wav, bass.wav, other.wav")
    print("Saved low.wav, mid.wav, high.wav")

# Usage
demucs_hml('/path/songgoeshere.wav')
