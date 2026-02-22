# server.py
# Flask backend for DJ AI.
#
# Files used (all in ai-dj2/ root):
#   bpm.py              — BPM detection (librosa)
#   get_chorus.py       — chorus timestamp detection
#   get_verse.py        — verse timestamp detection
#   stemsplitter.py     — demucs stem separation (low/mid/high)
#   many_transitions.py — smart tight/loose mix engine
#
# NOTE: make sure many_transitions.py has these two lines fixed:
#   line 32: remove "import essentia.standard as es"
#   line 41: change "from get_bpm import get_bpm" → "from bpm import get_bpm"
#
# SETUP:
#   pip install flask flask-cors
#
# RUN:
#   cd C:\Users\rheam\OneDrive\Documents\ai-dj2
#   python server.py

import os, sys, traceback, threading
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS


# ── Path setup ─────────────────────────────────────────────────────
# Everything lives in the root of ai-dj2/
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sections"))

# ── BPM (librosa — no Essentia) ────────────────────────────────────
from bpm import get_bpm

try:
    from get_chorus import find_chorus
    HAS_CHORUS = True
except Exception as e:
    print(f"  [warn] get_chorus: {e}")
    HAS_CHORUS = False

try:
    from get_verse import find_verse
    HAS_VERSE = True
except Exception as e:
    print(f"  [warn] get_verse: {e}")
    HAS_VERSE = False

# ── Stem splitter ──────────────────────────────────────────────────
try:
    from stemsplitter import demucs_hml
    HAS_STEMS = True
except Exception as e:
    print(f"  [warn] stemsplitter: {e}")
    HAS_STEMS = False

# ── Mix engine ─────────────────────────────────────────────────────
try:
    from many_transitions import make_transition
    HAS_MIX = True
except Exception as e:
    print(f"  [warn] many_transitions: {e}")
    HAS_MIX = False

# ── Flask ──────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

UPLOAD_DIR = os.path.join(ROOT, "uploads")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mix job tracker: job_id → { status, output_file, error, stage }
mix_jobs = {}


@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "modules": {
            "bpm":              True,
            "chorus":           HAS_CHORUS,
            "verse":            HAS_VERSE,
            "stemsplitter":     HAS_STEMS,
            "many_transitions": HAS_MIX,
        }
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Upload a WAV → returns BPM, chorus timestamps, verse timestamps.
    Key/Camelot is handled inside many_transitions.py at mix time.
    Expects: multipart/form-data with 'file' field.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files['file']
    if not f.filename.lower().endswith('.wav'):
        return jsonify({"error": "Only .wav files supported"}), 400

    filepath = os.path.join(UPLOAD_DIR, f.filename)
    f.save(filepath)

    result = {"filename": f.filename}

    # BPM via bpm.py (librosa)
    try:
        result["bpm"] = round(get_bpm(filepath), 2)
    except Exception as e:
        result["bpm"]       = None
        result["bpm_error"] = str(e)

    # Chorus timestamps via get_chorus.py
    if HAS_CHORUS:
        try:
            result["choruses"] = [[round(s, 2), round(e, 2)] for s, e in find_chorus(filepath)]
        except Exception as e:
            result["choruses"]     = []
            result["chorus_error"] = str(e)
    else:
        result["choruses"] = []

    # Verse timestamps via get_verse.py
    if HAS_VERSE:
        try:
            result["verses"] = [[round(s, 2), round(e, 2)] for s, e in find_verse(filepath)]
        except Exception as e:
            result["verses"]      = []
            result["verse_error"] = str(e)
    else:
        result["verses"] = []

    return jsonify(result)


# ── /mix/start ─────────────────────────────────────────────────────
@app.route('/mix/start', methods=['POST'])
def mix_start():
    """
    Start a mix job in a background thread (demucs takes time).
    Body: { file_a, file_b }
    Returns: { job_id }
    Poll /mix/status/<job_id> to check progress.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    file_a = os.path.join(UPLOAD_DIR, data.get("file_a", ""))
    file_b = os.path.join(UPLOAD_DIR, data.get("file_b", ""))

    if not os.path.exists(file_a):
        return jsonify({"error": f"File not found: {data.get('file_a')}"}), 400
    if not os.path.exists(file_b):
        return jsonify({"error": f"File not found: {data.get('file_b')}"}), 400
    if not HAS_MIX:
        return jsonify({"error": "many_transitions.py not loaded"}), 500

    job_id   = f"{Path(file_a).stem}__{Path(file_b).stem}"
    out_name = f"mix__{job_id}.wav"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    work_dir = os.path.join(OUTPUT_DIR, "work", job_id)

    mix_jobs[job_id] = {
        "status":      "running",
        "output_file": None,
        "error":       None,
        "stage":       "Starting..."
    }

    def run_mix():
        try:
            mix_jobs[job_id]["stage"] = "Analyzing BPM and key..."
            mix_jobs[job_id]["stage"] = "Running DEMUCS on Song 1 (stem separation)..."
            mix_jobs[job_id]["stage"] = "Running DEMUCS on Song 2 (stem separation)..."

            # make_transition handles everything internally:
            # BPM, key detection, chorus/verse detection, demucs, tight/loose decision
            mix_path = make_transition(file_a, file_b, output_dir=work_dir)

            import shutil
            shutil.copy(mix_path, out_path)

            mix_jobs[job_id]["status"]      = "done"
            mix_jobs[job_id]["output_file"] = out_name
            mix_jobs[job_id]["stage"]       = "Complete"
            print(f"\n[server] Mix done → {out_name}")

        except Exception as e:
            traceback.print_exc()
            mix_jobs[job_id]["status"] = "error"
            mix_jobs[job_id]["error"]  = str(e)
            mix_jobs[job_id]["stage"]  = "Failed"

    threading.Thread(target=run_mix, daemon=True).start()
    return jsonify({"job_id": job_id})


# ── /mix/status/<job_id> ───────────────────────────────────────────
@app.route('/mix/status/<job_id>')
def mix_status(job_id):
    """Poll this every few seconds to check if the mix is ready."""
    job = mix_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)


@app.route('/stream/<folder>/<filename>')
def stream(folder, filename):
    """Stream a WAV file for browser playback."""
    base = UPLOAD_DIR if folder == "uploads" else OUTPUT_DIR
    path = os.path.join(base, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, mimetype='audio/wav', conditional=True)


@app.route('/download/<filename>')
def download(filename):
    """Download a finished mix file."""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  DJ AI — SERVER")
    print("="*50)
    print(f"  bpm.py:              ✅")
    print(f"  get_chorus.py:       {'✅' if HAS_CHORUS else '❌'}")
    print(f"  get_verse.py:        {'✅' if HAS_VERSE else '❌'}")
    print(f"  stemsplitter.py:     {'✅' if HAS_STEMS else '❌'}")
    print(f"  many_transitions.py: {'✅' if HAS_MIX else '❌'}")
    print(f"\n  Uploads : {UPLOAD_DIR}")
    print(f"  Outputs : {OUTPUT_DIR}")
    print(f"\n  Open dj_ai_ui.html in your browser")
    print(f"  Running → http://localhost:5000")
    print("="*50 + "\n")

    app.run(debug=True, port=5000, host='0.0.0.0')