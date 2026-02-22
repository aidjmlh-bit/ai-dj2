"""Generate a browser-based DJ visualizer for a mix WAV.

Usage:
    python visualizer/generate_viz.py <path/to/mix.wav> [output.html]

If no output path is given the HTML is saved alongside the WAV with the same
stem and a .html extension.  Open it in any browser, click anywhere to load
your audio file, then press Play.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import librosa

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root)          # project root → get_bpm, many_transitions

from get_bpm import get_bpm
try:
    from many_transitions import get_key as _get_key
except Exception:
    _get_key = None


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------

N_POINTS = 2000
N_FFT    = 2048
HOP      = 512
LOW_MAX  = 300
MID_MAX  = 4000


def _rms_band(D: np.ndarray, mask: np.ndarray) -> np.ndarray:
    sub = D[mask, :]
    if sub.shape[0] == 0:
        return np.zeros(D.shape[1], dtype=np.float32)
    return np.sqrt(np.mean(sub ** 2, axis=0)).astype(np.float32)


def _downsample(arr: np.ndarray, n: int) -> list[float]:
    step = max(1, len(arr) / n)
    out  = []
    for i in range(n):
        lo = int(i * step)
        hi = min(int((i + 1) * step), len(arr))
        out.append(float(np.mean(arr[lo:hi])) if hi > lo else 0.0)
    return out


def _normalise(arr: np.ndarray) -> np.ndarray:
    return arr / (arr.max() + 1e-8)


def analyse(wav_path: str) -> dict:
    print(f"Loading {wav_path!r} …")
    y, sr    = librosa.load(wav_path, mono=True, sr=None)
    duration = float(len(y) / sr)
    print(f"  Duration : {duration:.1f} s   SR : {sr} Hz")

    print("  Computing STFT …")
    D     = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    low_rms  = _normalise(_rms_band(D, freqs < LOW_MAX))
    mid_rms  = _normalise(_rms_band(D, (freqs >= LOW_MAX) & (freqs < MID_MAX)))
    high_rms = _normalise(_rms_band(D, freqs >= MID_MAX))

    print("  Downsampling …")
    low_ds  = _downsample(low_rms,  N_POINTS)
    mid_ds  = _downsample(mid_rms,  N_POINTS)
    high_ds = _downsample(high_rms, N_POINTS)

    bpm = None
    try:
        print("  Estimating BPM …")
        bpm = float(get_bpm(wav_path))
        print(f"  BPM : {bpm:.1f}")
    except Exception as exc:
        print(f"  BPM estimation failed ({exc}).")

    key_str = "?"
    if _get_key is not None:
        try:
            print("  Detecting key …")
            k = _get_key(wav_path)
            key_str = f"{k[0]}{k[1]}"
            print(f"  Key : {key_str}")
        except Exception as exc:
            print(f"  Key detection failed ({exc}).")

    return {
        "duration": round(duration, 3),
        "bpm":      round(bpm, 2) if bpm else None,
        "key":      key_str,
        "n_points": N_POINTS,
        "low":      low_ds,
        "mid":      mid_ds,
        "high":     high_ds,
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DJ AI — %%TITLE%%</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0a0a;color:#ccc;font-family:'Courier New',monospace;min-height:100vh;user-select:none;overflow-x:hidden}

/* ── header ─────────────────────────────────────────────────────────── */
#hdr{display:flex;align-items:flex-start;justify-content:space-between;padding:18px 28px 12px}
#logo{line-height:1}
#logo-dj{font-size:36px;font-weight:900;color:#ff4400;letter-spacing:-1px}
#logo-sub{font-size:9px;letter-spacing:.25em;color:#444;margin-top:2px}
#status{display:flex;align-items:center;gap:10px;padding-top:10px;font-size:10px;letter-spacing:.15em;color:#555}
#dot{width:8px;height:8px;border-radius:50%;background:#22dd66;box-shadow:0 0 6px #22dd66;animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* ── decks ───────────────────────────────────────────────────────────── */
#decks{display:flex;align-items:center;justify-content:center;gap:0;padding:10px 0 4px;position:relative}
.deck-wrap{display:flex;flex-direction:column;align-items:center;gap:8px;flex:0 0 auto}
.deck-label{font-size:9px;letter-spacing:.18em;color:#444}
canvas.turntable{border-radius:50%}
#vs{font-size:22px;font-weight:700;color:#2a2a2a;letter-spacing:.1em;padding:0 32px;margin-top:28px}

/* ── frequency bars ──────────────────────────────────────────────────── */
#bands{padding:18px 28px 12px;display:flex;flex-direction:column;gap:12px}
.band-row{display:flex;align-items:center;gap:16px}
.band-lbl{font-size:10px;letter-spacing:.18em;font-weight:700;width:68px;text-align:right}
canvas.band-cv{flex:1;height:44px;background:#0d0d0d;border:1px solid #1a1a1a;border-radius:2px}

/* ── progress / seek ─────────────────────────────────────────────────── */
#prog-wrap{padding:4px 28px 14px;display:flex;flex-direction:column;gap:6px}
#prog-head{display:flex;justify-content:space-between;align-items:center}
#prog-lbl{font-size:9px;letter-spacing:.2em;color:#333}
#prog-time{font-size:10px;color:#444;letter-spacing:.08em}
#prog-track{width:100%;height:5px;background:#111;border:1px solid #1a1a1a;border-radius:2px;cursor:pointer;position:relative}
#prog-fill{height:100%;background:#ff4400;border-radius:2px;pointer-events:none;width:0%}

/* ── cards ───────────────────────────────────────────────────────────── */
#cards{display:flex;gap:16px;padding:10px 28px 24px}
.card{flex:1;border-radius:6px;padding:16px 20px;min-width:0}
#card-a{background:#0f0f0f;border:2px solid #ff4400}
#card-b{background:#0c0c0c;border:1px solid #1e1e1e}
.card-tag{font-size:9px;letter-spacing:.2em;margin-bottom:8px;display:flex;align-items:center;gap:6px}
#card-a .card-tag{color:#ff4400}
#card-b .card-tag{color:#333}
.card-name{font-size:22px;font-weight:900;color:#eee;letter-spacing:.04em;text-transform:uppercase;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:12px}
#card-b .card-name{color:#2a2a2a}
.card-pills{display:flex;gap:8px;flex-wrap:wrap}
.pill{font-size:10px;letter-spacing:.1em;padding:3px 11px;border-radius:20px;display:inline-flex;align-items:center}
#card-a .pill{border:1px solid #ff4400;color:#ff4400;background:transparent}
#card-b .pill{border:1px solid #222;color:#333;background:transparent}

/* ── controls ─────────────────────────────────────────────────────────── */
#ctrl{padding:0 28px 20px;display:flex;gap:10px;align-items:center}
#btn-play,#btn-load{background:#0f0f0f;border:1px solid #333;color:#ccc;padding:7px 22px;font-family:inherit;font-size:11px;letter-spacing:.12em;cursor:pointer;border-radius:3px}
#btn-play:hover,#btn-load:hover{background:#1a1a1a;border-color:#555}
#btn-play:disabled{opacity:.3;cursor:default}
#file-inp{display:none}

/* ── load overlay ─────────────────────────────────────────────────────── */
#overlay{position:fixed;inset:0;background:rgba(0,0,0,.78);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;z-index:99;transition:opacity .35s}
#overlay p{font-size:11px;letter-spacing:.22em;color:#555}
#overlay label{cursor:pointer;border:1px solid #333;padding:10px 28px;font-size:11px;letter-spacing:.15em;border-radius:3px;color:#aaa}
#overlay label:hover{border-color:#666;color:#ddd}
</style>
</head>
<body>

<!-- ── header ─────────────────────────────────────────────────────────── -->
<div id="hdr">
  <div id="logo">
    <div id="logo-dj">DJ AI</div>
    <div id="logo-sub">INTELLIGENT MIX ENGINE</div>
  </div>
  <div id="status">
    <span id="status-txt">AWAITING AUDIO</span>
    <div id="dot"></div>
    <span>SERVER ONLINE</span>
  </div>
</div>

<!-- ── decks ──────────────────────────────────────────────────────────── -->
<div id="decks">
  <div class="deck-wrap">
    <div class="deck-label">DECK A — NOW PLAYING</div>
    <canvas class="turntable" id="deck-a" width="220" height="220"></canvas>
  </div>
  <div id="vs">VS</div>
  <div class="deck-wrap">
    <div class="deck-label">DECK B — NEXT UP</div>
    <canvas class="turntable" id="deck-b" width="220" height="220"></canvas>
  </div>
</div>

<!-- ── frequency bars ─────────────────────────────────────────────────── -->
<div id="bands">
  <div class="band-row">
    <div class="band-lbl" style="color:#ff4433">BASS</div>
    <canvas class="band-cv" id="cv-bass"></canvas>
  </div>
  <div class="band-row">
    <div class="band-lbl" style="color:#33ddcc">VOCALS</div>
    <canvas class="band-cv" id="cv-vox"></canvas>
  </div>
  <div class="band-row">
    <div class="band-lbl" style="color:#9944ff">DRUMS</div>
    <canvas class="band-cv" id="cv-drm"></canvas>
  </div>
</div>

<!-- ── progress bar ───────────────────────────────────────────────────── -->
<div id="prog-wrap">
  <div id="prog-head">
    <span id="prog-lbl">COMPATIBILITY</span>
    <span id="prog-time">0:00 / %%DUR_FMT%%</span>
  </div>
  <div id="prog-track"><div id="prog-fill"></div></div>
</div>

<!-- ── song cards ─────────────────────────────────────────────────────── -->
<div id="cards">
  <div class="card" id="card-a">
    <div class="card-tag">▶ NOW PLAYING</div>
    <div class="card-name">%%TITLE%%</div>
    <div class="card-pills">
      <span class="pill">%%BPM_STR%%</span>
      <span class="pill">%%KEY%%</span>
    </div>
  </div>
  <div class="card" id="card-b">
    <div class="card-tag">⏭ NEXT</div>
    <div class="card-name">——</div>
    <div class="card-pills">
      <span class="pill">— BPM</span>
      <span class="pill">?</span>
    </div>
  </div>
</div>

<!-- ── controls ───────────────────────────────────────────────────────── -->
<div id="ctrl">
  <button id="btn-play" disabled>▶  PLAY</button>
  <label id="btn-load">LOAD AUDIO <input type="file" id="file-inp" accept=".wav,.mp3,.aiff,.flac,.ogg"></label>
</div>

<!-- ── overlay ────────────────────────────────────────────────────────── -->
<div id="overlay">
  <p>SELECT YOUR MIX TO BEGIN</p>
  <label>LOAD AUDIO <input type="file" accept=".wav,.mp3,.aiff,.flac,.ogg"
    onchange="handleFile(this.files[0])"></label>
</div>

<audio id="audio" preload="none"></audio>

<script>
// ── Embedded data ─────────────────────────────────────────────────────────
const DATA = %%DATA_JSON%%;

// ── Elements ──────────────────────────────────────────────────────────────
const audio    = document.getElementById('audio');
const btnPlay  = document.getElementById('btn-play');
const fileInp  = document.getElementById('file-inp');
const overlay  = document.getElementById('overlay');
const progFill = document.getElementById('prog-fill');
const progTime = document.getElementById('prog-time');
const progTrack= document.getElementById('prog-track');
const statusTx = document.getElementById('status-txt');

// ── Turntable drawing ─────────────────────────────────────────────────────
function drawTurntable(canvas, angle, active) {
  const dpr = window.devicePixelRatio || 1;
  const W   = canvas.width / dpr;
  const H   = canvas.height / dpr;
  const cx  = W / 2, cy = H / 2;
  const ctx = canvas.getContext('2d');

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  if (!active) ctx.globalAlpha = 0.35;

  // Background disc
  ctx.beginPath();
  ctx.arc(cx * dpr, cy * dpr, (cx - 1) * dpr, 0, Math.PI * 2);
  ctx.fillStyle = '#0d0d0d';
  ctx.fill();

  // Concentric rings
  const N_RINGS = 14;
  const maxR    = (cx - 2) * dpr;
  const minR    = 28 * dpr;
  ctx.save();
  ctx.translate(cx * dpr, cy * dpr);
  ctx.rotate(angle);
  for (let i = 0; i < N_RINGS; i++) {
    const t   = i / (N_RINGS - 1);
    const r   = minR + (maxR - minR) * (1 - t);
    const lum = Math.round(10 + t * 40);
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.strokeStyle = `rgb(${lum},${lum},${lum})`;
    ctx.lineWidth   = Math.max(1, (maxR - minR) / N_RINGS * 0.55);
    ctx.stroke();
  }
  ctx.restore();

  // Centre dot glow
  ctx.save();
  ctx.translate(cx * dpr, cy * dpr);
  const grd = ctx.createRadialGradient(0, 0, 0, 0, 0, 22 * dpr);
  grd.addColorStop(0,   '#ff6644');
  grd.addColorStop(0.55,'#cc2200');
  grd.addColorStop(1,   'rgba(180,20,0,0)');
  ctx.shadowBlur  = 24 * dpr;
  ctx.shadowColor = '#ff3300';
  ctx.beginPath();
  ctx.arc(0, 0, 14 * dpr, 0, Math.PI * 2);
  ctx.fillStyle = grd;
  ctx.fill();
  // Inner highlight
  ctx.shadowBlur  = 0;
  ctx.beginPath();
  ctx.arc(-3 * dpr, -3 * dpr, 5 * dpr, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255,180,140,0.45)';
  ctx.fill();
  ctx.restore();

  ctx.restore();
}

// Init turntable canvases (DPR-aware)
function initCanvas(el) {
  const dpr = window.devicePixelRatio || 1;
  const s   = 220;
  el.width  = s * dpr;
  el.height = s * dpr;
  el.style.width  = s + 'px';
  el.style.height = s + 'px';
}
const cvA = document.getElementById('deck-a');
const cvB = document.getElementById('deck-b');
initCanvas(cvA);
initCanvas(cvB);
drawTurntable(cvA, 0, false);
drawTurntable(cvB, 0, false);

// ── Band canvases ─────────────────────────────────────────────────────────
const BANDS = [
  { cv: document.getElementById('cv-bass'), data: DATA.low,  colour: '#ff4433' },
  { cv: document.getElementById('cv-vox'),  data: DATA.mid,  colour: '#33ddcc' },
  { cv: document.getElementById('cv-drm'),  data: DATA.high, colour: '#9944ff' },
];

function initBandCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const W   = cv.parentElement.offsetWidth - cv.previousElementSibling.offsetWidth - 16;
  cv.width  = Math.max(200, W) * dpr;
  cv.height = 44 * dpr;
  cv.style.height = '44px';
}
function drawBand(band, val) {
  const dpr = window.devicePixelRatio || 1;
  const cv  = band.cv;
  const ctx = cv.getContext('2d');
  const W   = cv.width;
  const H   = cv.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0d0d0d';
  ctx.fillRect(0, 0, W, H);

  // clamp val
  val = Math.max(0, Math.min(1, val));
  const PAD = 6 * dpr;
  const y   = H - PAD - val * (H - 2 * PAD);

  // Glow line
  ctx.save();
  ctx.shadowBlur  = 12 * dpr;
  ctx.shadowColor = band.colour;
  ctx.strokeStyle = band.colour;
  ctx.lineWidth   = 2 * dpr;
  ctx.beginPath();
  ctx.moveTo(0, y);
  ctx.lineTo(W, y);
  ctx.stroke();
  ctx.restore();

  // Dot at right end
  ctx.save();
  ctx.shadowBlur  = 16 * dpr;
  ctx.shadowColor = band.colour;
  ctx.fillStyle   = band.colour;
  ctx.beginPath();
  ctx.arc(W - 8 * dpr, y, 5 * dpr, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// Init band canvases
function initBands() {
  for (const b of BANDS) {
    initBandCanvas(b.cv);
    drawBand(b, 0.5);
  }
}
initBands();
window.addEventListener('resize', initBands);

// ── Animation ─────────────────────────────────────────────────────────────
let angleA = 0;
let lastTs = null;
let raf    = null;
const BPM  = DATA.bpm || 120;
// radians per second = (BPM beats/min) / 60 * (2π / 2 beats per rotation)
const RAD_PER_SEC = (BPM / 60) * Math.PI;

function fmtTime(s) {
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2,'0')}`;
}

function tick(ts) {
  const dt = lastTs === null ? 0 : (ts - lastTs) / 1000;
  lastTs   = ts;

  // Turntable rotation (Deck A only while playing)
  if (!audio.paused && !audio.ended) angleA += RAD_PER_SEC * dt;
  drawTurntable(cvA, angleA, audio.src !== '');
  drawTurntable(cvB, 0,      false);

  // Band bars
  const frac = DATA.duration > 0 ? audio.currentTime / DATA.duration : 0;
  const idx  = Math.min(DATA.n_points - 1, Math.floor(frac * DATA.n_points));
  for (const b of BANDS) drawBand(b, b.data[idx]);

  // Progress bar
  progFill.style.width = (frac * 100).toFixed(2) + '%';
  progTime.textContent = fmtTime(audio.currentTime) + ' / %%DUR_FMT%%';

  if (!audio.paused && !audio.ended) {
    raf = requestAnimationFrame(tick);
  } else {
    raf = null;
    lastTs = null;
  }
}

function startAnim() {
  if (!raf) raf = requestAnimationFrame(tick);
}

// ── Controls ──────────────────────────────────────────────────────────────
btnPlay.addEventListener('click', () => {
  if (audio.paused) {
    audio.play();
    btnPlay.textContent = '⏸  PAUSE';
    startAnim();
  } else {
    audio.pause();
    btnPlay.textContent = '▶  PLAY';
  }
});

audio.addEventListener('ended', () => {
  btnPlay.textContent = '▶  PLAY';
});

// Progress bar seek
progTrack.addEventListener('click', e => {
  if (!audio.src) return;
  const rect = progTrack.getBoundingClientRect();
  audio.currentTime  = ((e.clientX - rect.left) / rect.width) * DATA.duration;
  progFill.style.width = (audio.currentTime / DATA.duration * 100).toFixed(2) + '%';
  progTime.textContent = fmtTime(audio.currentTime) + ' / %%DUR_FMT%%';
  startAnim();
});

fileInp.addEventListener('change', e => handleFile(e.target.files[0]));

function handleFile(file) {
  if (!file) return;
  audio.src = URL.createObjectURL(file);
  audio.load();
  overlay.style.opacity = '0';
  setTimeout(() => overlay.style.display = 'none', 350);
  btnPlay.disabled = false;
  statusTx.textContent = 'NOW PLAYING';
  drawTurntable(cvA, angleA, true);
  startAnim();
}

// Initial static draw
drawTurntable(cvA, 0, false);
drawTurntable(cvB, 0, false);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _fmt_dur(s: float) -> str:
    return f"{int(s)//60}:{int(s)%60:02d}"


def generate_html(data: dict, title: str) -> str:
    bpm_str  = f"{data['bpm']:.2f} BPM" if data["bpm"] else "? BPM"
    dur_fmt  = _fmt_dur(data["duration"])
    data_js  = json.dumps(data, separators=(",", ":"))

    html = _HTML
    html = html.replace("%%TITLE%%",    title)
    html = html.replace("%%DUR_FMT%%",  dur_fmt)
    html = html.replace("%%BPM_STR%%",  bpm_str)
    html = html.replace("%%KEY%%",      data.get("key", "?"))
    html = html.replace("%%DATA_JSON%%", data_js)
    return html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python visualizer/generate_viz.py <mix.wav> [output.html]")
        sys.exit(1)

    wav_path = sys.argv[1]
    if not os.path.exists(wav_path):
        print(f"Error: file not found: {wav_path!r}")
        sys.exit(1)

    stem     = os.path.splitext(os.path.basename(wav_path))[0]
    out_path = sys.argv[2] if len(sys.argv) == 3 else os.path.join(
        os.path.dirname(os.path.abspath(wav_path)), stem + ".html"
    )

    data = analyse(wav_path)
    html = generate_html(data, title=stem)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nVisualizer saved → {out_path}")
    print("Open in Chrome / Safari, click 'LOAD AUDIO', select the WAV, press PLAY.")
