# server.py
# Flask backend — connects all AI DJ modules to the browser UI.
#
# Mix priority:
#   1. many_transitions.py  (tight/loose auto-detection, best quality)
#   2. section_transition.py (fallback with manual strategy)
#   3. eq_transition.py      (last resort)
#
# SETUP:
#   pip install flask flask-cors
#
# RUN:
#   cd C:\Users\rheam\OneDrive\Documents\ai-dj2
#   python server.py

import os, sys, json, traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "get_key"))
sys.path.insert(0, os.path.join(ROOT, "sections"))
sys.path.insert(0, os.path.join(ROOT, "EQtransition"))

from bpm     import get_bpm
from get_key import detect_key
from camelot import camelot_compatibility, get_transition_advice

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

try:
    from many_transitions import make_transition
    HAS_MANY = True
except Exception as e:
    print(f"  [warn] many_transitions: {e}")
    HAS_MANY = False

try:
    from section_transition import mix_by_sections
    HAS_SECTION = True
except Exception as e:
    print(f"  [warn] section_transition: {e}")
    HAS_SECTION = False

try:
    from eq_transition import render_mix
    HAS_EQ = True
except Exception as e:
    print(f"  [warn] eq_transition: {e}")
    HAS_EQ = False

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = os.path.join(ROOT, "uploads")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "modules": {
            "bpm":              True,
            "key":              True,
            "chorus":           HAS_CHORUS,
            "verse":            HAS_VERSE,
            "many_transitions": HAS_MANY,
            "section_mix":      HAS_SECTION,
            "eq_mix":           HAS_EQ,
        }
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Upload a WAV file → returns BPM, key, camelot code, chorus/verse timestamps.
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

    # BPM
    try:
        result["bpm"] = round(get_bpm(filepath), 2)
    except Exception as e:
        result["bpm"] = None
        result["bpm_error"] = str(e)

    # Key + Camelot
    try:
        kd = detect_key(filepath)
        result["key_name"]    = kd.get("best_key") or kd.get("key_name", "?")
        result["camelot_code"]= kd.get("camelot_code", "?")
        result["confidence"]  = round(kd.get("confidence", 0), 3)
    except Exception as e:
        result["key_name"]     = "?"
        result["camelot_code"] = "?"
        result["key_error"]    = str(e)

    # Chorus timestamps
    if HAS_CHORUS:
        try:
            result["choruses"] = [[round(s,2), round(e,2)] for s,e in find_chorus(filepath)]
        except Exception as e:
            result["choruses"]     = []
            result["chorus_error"] = str(e)
    else:
        result["choruses"] = []

    # Verse timestamps
    if HAS_VERSE:
        try:
            result["verses"] = [[round(s,2), round(e,2)] for s,e in find_verse(filepath)]
        except Exception as e:
            result["verses"]      = []
            result["verse_error"] = str(e)
    else:
        result["verses"] = []

    return jsonify(result)


@app.route('/compatibility', methods=['POST'])
def compatibility():
    """
    Check how well two songs will mix.
    Body: { camelot_a, camelot_b, bpm_a, bpm_b }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    try:
        key_score = camelot_compatibility(data["camelot_a"], data["camelot_b"])
        advice    = get_transition_advice(data["camelot_a"], data["camelot_b"])

        bpm_a = float(data.get("bpm_a") or 120)
        bpm_b = float(data.get("bpm_b") or 120)
        # Account for half/double time
        delta = min(abs(bpm_a - bpm_b), abs(bpm_a - bpm_b*2), abs(bpm_a*2 - bpm_b))

        bpm_score = max(0.0, 1.0 - delta / 20.0)
        combined  = round(0.6 * key_score + 0.4 * bpm_score, 3)

        # Predict which transition many_transitions.py will choose
        bpm_tight  = delta <= 5
        key_compat = key_score >= 0.7
        predicted  = "TIGHT" if (bpm_tight and key_compat) else "LOOSE"

        return jsonify({
            "key_score":  round(key_score, 3),
            "bpm_score":  round(bpm_score, 3),
            "combined":   combined,
            "bpm_delta":  round(delta, 2),
            "advice":     advice,
            "predicted_transition": predicted,
            "recommended": (
                "crossfade"    if key_score >= 0.8 and delta <= 6 else
                "low_cut_echo" if key_score < 0.6 else
                "crossfade"
            )
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/mix', methods=['POST'])
def mix():
    """
    Create a mix between two uploaded songs.

    Uses many_transitions.py first (auto tight/loose detection).
    Falls back to section_transition.py, then eq_transition.py.

    Body: { file_a, file_b, strategy, bars }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    file_a   = os.path.join(UPLOAD_DIR, data.get("file_a", ""))
    file_b   = os.path.join(UPLOAD_DIR, data.get("file_b", ""))
    strategy = data.get("strategy", "chorus_chorus")
    bars     = int(data.get("bars", 8))

    if not os.path.exists(file_a):
        return jsonify({"error": f"File not found: {data.get('file_a')}"}), 400
    if not os.path.exists(file_b):
        return jsonify({"error": f"File not found: {data.get('file_b')}"}), 400

    a_stem   = Path(file_a).stem
    b_stem   = Path(file_b).stem
    out_name = f"mix__{a_stem}__{b_stem}.wav"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    out_dir  = os.path.join(OUTPUT_DIR, "many_transitions_work")

    engine_used = None

    try:
        # ── Engine 1: many_transitions (smart auto tight/loose) ──
        if HAS_MANY:
            print(f"\n[server] Using many_transitions: {a_stem} → {b_stem}")
            mix_path   = make_transition(file_a, file_b, output_dir=out_dir)
            # Copy result to standard output location
            import shutil
            shutil.copy(mix_path, out_path)
            engine_used = "many_transitions"

        # ── Engine 2: section_transition (manual strategy) ──
        elif HAS_SECTION:
            print(f"\n[server] Using section_transition ({strategy}): {a_stem} → {b_stem}")
            mix_by_sections(file_a, file_b, out_path, preference=strategy, bars=bars)
            engine_used = "section_transition"

        # ── Engine 3: eq_transition (last resort) ──
        elif HAS_EQ:
            print(f"\n[server] Using eq_transition: {a_stem} → {b_stem}")
            render_mix(file_a, file_b, out_path)
            engine_used = "eq_transition"

        else:
            return jsonify({"error": "No mix engine available"}), 500

        return jsonify({
            "success":     True,
            "output_file": out_name,
            "engine":      engine_used,
            "strategy":    strategy,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "engine_attempted": engine_used}), 500


@app.route('/stream/<folder>/<filename>')
def stream(folder, filename):
    """Stream a WAV for browser playback."""
    base = UPLOAD_DIR if folder == "uploads" else OUTPUT_DIR
    path = os.path.join(base, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, mimetype='audio/wav', conditional=True)


@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    print("\n" + "="*54)
    print("  DJ AI — FLASK SERVER")
    print("="*54)
    print(f"  Uploads : {UPLOAD_DIR}")
    print(f"  Outputs : {OUTPUT_DIR}")
    print(f"\n  Analysis modules:")
    print(f"    bpm + key:         ✅")
    print(f"    chorus detect:     {'✅' if HAS_CHORUS else '❌'}")
    print(f"    verse detect:      {'✅' if HAS_VERSE else '❌'}")
    print(f"\n  Mix engines (priority):")
    print(f"    1. many_transitions:   {'✅  ← PRIMARY' if HAS_MANY else '❌'}")
    print(f"    2. section_transition: {'✅' if HAS_SECTION else '❌'}")
    print(f"    3. eq_transition:      {'✅' if HAS_EQ else '❌'}")
    print(f"\n  Open dj_ai_ui.html in your browser")
    print(f"  Running at → http://localhost:5000")
    print("="*54 + "\n")

    app.run(debug=True, port=5000, host='0.0.0.0')