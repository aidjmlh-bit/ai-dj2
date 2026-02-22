"""DJ AI — Static Visualizer Generator

Generates a self-contained HTML file that:
  1. Lets you drop two WAV files
  2. Connects to server.py for BPM analysis + mixing
  3. Shows beat-reactive bass/vocals/drums pulses
  4. Plays the finished mix with spinning vinyls

Usage:
    python generate_viz.py              → generates dj_ai_viz.html in current folder
    python generate_viz.py output.html  → custom output path

Then open the HTML in Chrome and make sure server.py is running.
"""

import os
import sys

OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizer_htmls")
OUTPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUTPUT_DIR, "dj_ai_viz.html")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DJ AI</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Mono:wght@400;700&display=swap');

:root {
  --bg:     #070707;
  --surf:   #0f0f0f;
  --surf2:  #161616;
  --border: #222;
  --red:    #ff3232;
  --orange: #ff8800;
  --cyan:   #00e5ff;
  --purple: #c044ff;
  --green:  #00e676;
  --text:   #eeeeee;
  --muted:  #444;
}

* { margin:0; padding:0; box-sizing:border-box; }

body {
  background:var(--bg); color:var(--text);
  font-family:'Space Mono',monospace;
  height:100vh; display:flex; flex-direction:column; overflow:hidden;
}

body::after {
  content:''; position:fixed; inset:0;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
  pointer-events:none; z-index:9999; opacity:.5;
}

/* ── LOADING OVERLAY ── */
.loading-overlay {
  position:fixed; inset:0;
  background:rgba(7,7,7,0.92);
  z-index:800;
  display:flex; flex-direction:column;
  align-items:center; justify-content:center;
  gap:28px;
  opacity:0; pointer-events:none;
  transition:opacity .3s;
  backdrop-filter:blur(6px);
}
.loading-overlay.active { opacity:1; pointer-events:all; }

.loading-bars { display:flex; gap:5px; align-items:flex-end; height:32px; }
.loading-bar {
  width:5px; border-radius:3px; background:var(--red);
  animation:bar-bounce 1s ease-in-out infinite;
}
.loading-bar:nth-child(1){animation-delay:0s;   background:var(--red);}
.loading-bar:nth-child(2){animation-delay:.15s; background:#ff5500;}
.loading-bar:nth-child(3){animation-delay:.3s;  background:var(--orange);}
.loading-bar:nth-child(4){animation-delay:.45s; background:#ffaa00;}
.loading-bar:nth-child(5){animation-delay:.6s;  background:var(--orange);}
.loading-bar:nth-child(6){animation-delay:.75s; background:#ff5500;}
.loading-bar:nth-child(7){animation-delay:.9s;  background:var(--red);}
@keyframes bar-bounce { 0%,100%{height:6px;opacity:.4} 50%{height:32px;opacity:1} }

.loading-title {
  font-family:'Bebas Neue'; font-size:1.6rem; letter-spacing:4px;
  background:linear-gradient(90deg,var(--red),var(--orange));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.loading-stage {
  font-size:.65rem; letter-spacing:2px; color:var(--muted);
  text-align:center; max-width:320px; line-height:1.8;
}
.loading-stage::after {
  content:''; animation:dots 1.5s steps(4,end) infinite;
}
@keyframes dots { 0%{content:''} 25%{content:'.'} 50%{content:'..'} 75%{content:'...'} 100%{content:''} }
.loading-note {
  font-size:.55rem; letter-spacing:2px; color:var(--muted);
  border:1px solid var(--border); padding:6px 16px; border-radius:4px;
}

/* ── HEADER ── */
header {
  display:flex; align-items:center; justify-content:space-between;
  padding:12px 28px; background:var(--surf);
  border-bottom:1px solid var(--border); flex-shrink:0;
}
.logo {
  font-family:'Bebas Neue'; font-size:1.8rem; letter-spacing:6px;
  background:linear-gradient(90deg,var(--red),var(--orange));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.logo-sub { font-size:.52rem; letter-spacing:3px; color:var(--muted); -webkit-text-fill-color:var(--muted); display:block; }
.hdr-right { display:flex; align-items:center; gap:14px; }
.tm-badge {
  font-size:.55rem; letter-spacing:2px; padding:4px 12px;
  border-radius:20px; border:1px solid var(--border); color:var(--muted); transition:all .4s;
}
.tm-badge.tight { border-color:var(--green); color:var(--green); box-shadow:0 0 10px rgba(0,230,118,.15); }
.tm-badge.loose { border-color:var(--orange); color:var(--orange); box-shadow:0 0 10px rgba(255,136,0,.15); }
.dot { width:7px; height:7px; border-radius:50%; background:var(--muted); display:inline-block; margin-right:6px; transition:all .3s; }
.dot.on { background:var(--green); box-shadow:0 0 8px var(--green); }
.srv { font-size:.55rem; letter-spacing:2px; color:var(--muted); display:flex; align-items:center; }

/* ── MAIN ── */
main { flex:1; display:flex; flex-direction:column; align-items:center; padding:20px 28px 0; gap:16px; overflow:hidden; }

/* ── DECKS ── */
.decks { display:flex; align-items:center; gap:40px; flex-shrink:0; }
.deck  { display:flex; flex-direction:column; align-items:center; gap:10px; }
.deck-label { font-size:.6rem; letter-spacing:3px; color:var(--muted); font-family:'Bebas Neue'; }
.vinyl-wrap { position:relative; width:190px; height:190px; }
.vinyl {
  width:190px; height:190px; border-radius:50%;
  background:radial-gradient(circle,
    #555 0%,#111 14%,
    #444 14.5%,#0d0d0d 22%,
    #3a3a3a 22.5%,#0a0a0a 30%,
    #333 30.5%,#080808 38%,
    #2e2e2e 38.5%,#060606 46%,
    #2a2a2a 46.5%,#050505 54%,
    #252525 54.5%,#040404 62%,
    #222 62.5%,#030303 70%,
    #1e1e1e 70.5%,#020202 78%,
    #1a1a1a 78.5%,#010101 100%);
  box-shadow:0 0 0 3px #444,0 0 0 4px #111,0 16px 48px rgba(0,0,0,.9);
  animation:spin 2s linear infinite; animation-play-state:paused; position:relative;
}
.vinyl::after {
  content:''; position:absolute; top:50%; left:50%;
  transform:translate(-50%,-50%);
  width:48px; height:48px; border-radius:50%;
  background:radial-gradient(circle,#ff6666 0%,var(--red) 40%,#800 70%,#1a0000 71%);
  box-shadow:0 0 20px rgba(255,50,50,.8),0 0 40px rgba(255,50,50,.3); z-index:3;
}
.vinyl-wrap::before {
  content:''; position:absolute; inset:0; border-radius:50%;
  background:linear-gradient(130deg,rgba(255,255,255,.18) 0%,rgba(255,255,255,.04) 40%,transparent 60%);
  z-index:4; pointer-events:none;
}
.vinyl.spin {
  animation-play-state:running;
  box-shadow:0 0 0 3px #666,0 0 0 4px #222,
             0 0 30px rgba(255,50,50,.35),0 0 60px rgba(255,50,50,.15),
             0 16px 48px rgba(0,0,0,.9);
}
@keyframes spin { to { transform:rotate(360deg); } }
.vs { font-family:'Bebas Neue'; font-size:2.2rem; color:var(--border); padding-top:14px; }

/* ── DROP ZONES ── */
.drop-decks { display:flex; align-items:center; gap:40px; flex-shrink:0; }
.drop-zone {
  width:190px; height:190px; border-radius:50%;
  border:2px dashed var(--border);
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  gap:8px; cursor:pointer; transition:all .2s; background:var(--surf);
  position:relative;
}
.drop-zone:hover, .drop-zone.over { border-color:var(--red); background:rgba(255,50,50,.04); }
.drop-zone.loaded { border-color:var(--orange); border-style:solid; }
.drop-icon { font-size:1.6rem; }
.drop-txt { font-size:.52rem; letter-spacing:2px; color:var(--muted); text-align:center; text-transform:uppercase; line-height:1.7; }
.drop-txt strong { color:var(--red); display:block; }

/* ── VISUALIZERS ── */
.vizbox { width:100%; max-width:680px; display:flex; flex-direction:column; gap:6px; flex-shrink:0; }
.viz-row { display:flex; align-items:center; gap:10px; }
.viz-lbl { font-size:.55rem; letter-spacing:2px; min-width:58px; text-align:right; text-transform:uppercase; }
canvas.wv { flex:1; height:44px; display:block; border-radius:3px; background:var(--surf); border:1px solid var(--border); }

/* ── COMPAT BAR ── */
.compat-row { width:100%; max-width:680px; display:flex; align-items:center; gap:12px; flex-shrink:0; }
.compat-lbl { font-size:.55rem; letter-spacing:2px; color:var(--muted); min-width:100px; }
.cbar-out { flex:1; height:5px; background:var(--surf2); border:1px solid var(--border); border-radius:3px; overflow:hidden; }
.cbar-in  { height:100%; width:0%; background:linear-gradient(90deg,var(--red),var(--orange)); border-radius:3px; transition:width .6s; }
.compat-val { font-size:.65rem; color:var(--orange); min-width:34px; text-align:right; }

/* ── CARDS ── */
.cards { display:flex; gap:16px; width:100%; max-width:680px; flex-shrink:0; }
.card {
  flex:1; background:var(--surf); border:1px solid var(--border);
  border-radius:7px; padding:12px 16px; display:flex; flex-direction:column; gap:4px;
  transition:border-color .3s, opacity .3s;
}
.card.active { border-color:var(--red); }
.card-tag { font-size:.52rem; letter-spacing:3px; color:var(--red); }
.card-title { font-family:'Bebas Neue'; font-size:1.15rem; letter-spacing:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.pills { display:flex; gap:6px; flex-wrap:wrap; margin-top:3px; }
.pill { font-size:.55rem; letter-spacing:1px; padding:2px 8px; border-radius:20px; background:var(--surf2); border:1px solid var(--border); color:var(--muted); }
.pill.hi { border-color:var(--orange); color:var(--orange); }

/* ── FOOTER ── */
footer {
  background:var(--surf); border-top:1px solid var(--border);
  padding:12px 24px; display:flex; align-items:center; gap:12px; flex-shrink:0;
}
.btn {
  background:none; border:1px solid var(--border); color:var(--text);
  font-family:'Space Mono'; font-size:.6rem; letter-spacing:2px;
  padding:8px 14px; border-radius:4px; cursor:pointer; transition:all .15s; white-space:nowrap;
}
.btn:hover { border-color:var(--red); color:var(--red); }
.btn.play { background:var(--red); border-color:var(--red); color:#fff; font-size:.75rem; padding:9px 22px; }
.btn.play:hover { background:#c02020; }
.btn:disabled { opacity:.35; cursor:not-allowed; border-color:var(--border); color:var(--muted); }
.spacer { flex:1; }

.toast {
  position:fixed; bottom:70px; left:50%; transform:translateX(-50%) translateY(16px);
  background:var(--surf2); border:1px solid var(--border); padding:8px 18px;
  border-radius:5px; font-size:.6rem; letter-spacing:1px;
  opacity:0; transition:all .25s; pointer-events:none; z-index:600; white-space:nowrap;
}
.toast.show { opacity:1; transform:translateX(-50%) translateY(0); }

#file-a-inp, #file-b-inp { display:none; }
</style>
</head>
<body>

<!-- LOADING OVERLAY -->
<div class="loading-overlay" id="loading-overlay">
  <div class="loading-bars">
    <div class="loading-bar"></div><div class="loading-bar"></div>
    <div class="loading-bar"></div><div class="loading-bar"></div>
    <div class="loading-bar"></div><div class="loading-bar"></div>
    <div class="loading-bar"></div>
  </div>
  <div class="loading-title">MIX IN PROGRESS</div>
  <div class="loading-stage" id="loading-stage">Separating stems with DEMUCS</div>
  <div class="loading-note">THIS MAY TAKE A FEW MINUTES</div>
</div>

<header>
  <div class="logo">DJ AI<span class="logo-sub">INTELLIGENT MIX ENGINE</span></div>
  <div class="hdr-right">
    <div class="tm-badge" id="tm-badge">AWAITING SONGS</div>
    <div class="srv"><span class="dot" id="srv-dot"></span><span id="srv-txt">CONNECTING...</span></div>
  </div>
</header>

<main>
  <!-- Drop zones shown before songs loaded, replaced by vinyls after -->
  <div class="drop-decks" id="drop-decks">
    <div class="deck">
      <div class="deck-label">DECK A — DROP HERE</div>
      <div class="drop-zone" id="dz-a"
        onclick="document.getElementById('file-a-inp').click()"
        ondragover="dzOver(event,'dz-a')" ondragleave="dzLeave('dz-a')" ondrop="dzDrop(event,'a')">
        <div class="drop-icon">⬇</div>
        <div class="drop-txt"><strong>SONG 1</strong>drag .wav here</div>
      </div>
    </div>
    <div class="vs">VS</div>
    <div class="deck">
      <div class="deck-label">DECK B — DROP HERE</div>
      <div class="drop-zone" id="dz-b"
        onclick="document.getElementById('file-b-inp').click()"
        ondragover="dzOver(event,'dz-b')" ondragleave="dzLeave('dz-b')" ondrop="dzDrop(event,'b')">
        <div class="drop-icon">⬇</div>
        <div class="drop-txt"><strong>SONG 2</strong>drag .wav here</div>
      </div>
    </div>
  </div>

  <!-- Vinyls shown after both songs loaded -->
  <div class="decks" id="vinyl-decks" style="display:none">
    <div class="deck">
      <div class="deck-label">DECK A — NOW PLAYING</div>
      <div class="vinyl-wrap"><div class="vinyl" id="va"></div></div>
    </div>
    <div class="vs">VS</div>
    <div class="deck">
      <div class="deck-label">DECK B — NEXT UP</div>
      <div class="vinyl-wrap"><div class="vinyl" id="vb"></div></div>
    </div>
  </div>

  <div class="vizbox">
    <div class="viz-row">
      <div class="viz-lbl" style="color:var(--red)">BASS</div>
      <canvas class="wv" id="cv-bass"></canvas>
    </div>
    <div class="viz-row">
      <div class="viz-lbl" style="color:var(--cyan)">VOCALS</div>
      <canvas class="wv" id="cv-vocals"></canvas>
    </div>
    <div class="viz-row">
      <div class="viz-lbl" style="color:var(--purple)">DRUMS</div>
      <canvas class="wv" id="cv-drums"></canvas>
    </div>
  </div>

  <div class="compat-row">
    <div class="compat-lbl">COMPATIBILITY</div>
    <div class="cbar-out"><div class="cbar-in" id="cbar"></div></div>
    <div class="compat-val" id="cval">—</div>
  </div>

  <div class="cards">
    <div class="card active" id="card-a">
      <div class="card-tag">▶ NOW PLAYING</div>
      <div class="card-title" id="title-a">DROP SONG 1</div>
      <div class="pills"><span class="pill hi" id="bpm-a">— BPM</span><span class="pill" id="key-a">— KEY</span></div>
    </div>
    <div class="card" id="card-b" style="opacity:.35">
      <div class="card-tag">⏭ NEXT</div>
      <div class="card-title" id="title-b">DROP SONG 2</div>
      <div class="pills"><span class="pill" id="bpm-b">— BPM</span><span class="pill" id="key-b">— KEY</span></div>
    </div>
  </div>
</main>

<footer>
  <button class="btn play" id="play-btn" onclick="togglePlay()" disabled>▶ PLAY</button>
  <button class="btn" id="mix-btn" onclick="triggerMix()" disabled>⚡ MIX NOW</button>
  <div class="spacer"></div>
</footer>

<input type="file" id="file-a-inp" accept=".wav" onchange="fileSelect(event,'a')">
<input type="file" id="file-b-inp" accept=".wav" onchange="fileSelect(event,'b')">
<div class="toast" id="toast"></div>

<script>
const API = 'http://localhost:8000';

let songA = null, songB = null;
let isPlaying = false;
let audioCtx  = null, audioEl = null;
let analyserB = null, analyserV = null, analyserD = null;
let rafId = null, pollTimer = null;

const H_LEN  = 320;
const hBass   = new Float32Array(H_LEN);
const hVocals = new Float32Array(H_LEN);
const hDrums  = new Float32Array(H_LEN);

// Beat decay state
let beatDecayB = 0, beatDecayV = 0, beatDecayD = 0;
let lastBeatTime = 0;

// ── Server ────────────────────────────────────────────────────────
async function checkServer() {
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(2500) });
    if (r.ok) {
      document.getElementById('srv-dot').classList.add('on');
      document.getElementById('srv-txt').textContent = 'SERVER ONLINE';
      return true;
    }
  } catch(_) {}
  document.getElementById('srv-dot').classList.remove('on');
  document.getElementById('srv-txt').textContent = 'SERVER OFFLINE';
  return false;
}
checkServer();
setInterval(checkServer, 6000);

// ── Loading overlay ───────────────────────────────────────────────
function showLoading(msg) {
  document.getElementById('loading-stage').textContent = msg || 'Separating stems with DEMUCS';
  document.getElementById('loading-overlay').classList.add('active');
  document.getElementById('mix-btn').disabled = true;
}
function hideLoading() {
  document.getElementById('loading-overlay').classList.remove('active');
  document.getElementById('mix-btn').disabled = false;
}
function setStage(msg) { document.getElementById('loading-stage').textContent = msg; }

// ── Drop zones ────────────────────────────────────────────────────
function dzOver(e, id)  { e.preventDefault(); document.getElementById(id).classList.add('over'); }
function dzLeave(id)    { document.getElementById(id).classList.remove('over'); }
function dzDrop(e, deck) {
  e.preventDefault();
  dzLeave('dz-' + deck);
  const f = e.dataTransfer.files[0];
  if (f && f.name.toLowerCase().endsWith('.wav')) loadSong(f, deck);
}
function fileSelect(e, deck) {
  const f = e.target.files[0];
  if (f) loadSong(f, deck);
  e.target.value = '';
}

// ── Load + analyze a song ─────────────────────────────────────────
async function loadSong(file, deck) {
  const name = file.name.replace(/\.wav$/i, '').toUpperCase();
  toast(`Uploading ${name}...`);

  const dz = document.getElementById('dz-' + deck);
  dz.classList.add('loaded');
  dz.querySelector('.drop-txt').innerHTML = `<strong>${name}</strong>analyzing...`;

  const fd = new FormData();
  fd.append('file', file);

  try {
    const r    = await fetch(`${API}/analyze`, { method:'POST', body:fd });
    const data = await r.json();

    const song = {
      name,
      filename:  file.name,
      bpm:       data.bpm != null ? `${data.bpm} BPM` : '?',
      bpm_raw:   data.bpm,
      key:       data.key_name || '?',
      choruses:  data.choruses || [],
      verses:    data.verses   || [],
    };

    if (deck === 'a') songA = song;
    else              songB = song;

    dz.querySelector('.drop-txt').innerHTML = `<strong>${name}</strong>${song.bpm} · ${song.key}`;
    updateCards();
    toast(`✓ ${name}  ${song.bpm} · ${song.key}`);

    // Update page title dynamically based on loaded songs
    const titleA = songA ? songA.name : '?';
    const titleB = songB ? songB.name : '?';
    document.title = songA && songB
      ? `DJ AI — ${titleA} × ${titleB}`
      : `DJ AI — ${name}`;

    // Show vinyls once both loaded
    if (songA && songB) {
      document.getElementById('drop-decks').style.display  = 'none';
      document.getElementById('vinyl-decks').style.display = 'flex';
      document.getElementById('mix-btn').disabled = false;
      checkCompat();
    }

    if (audioEl) document.getElementById('play-btn').disabled = false;

  } catch(e) {
    toast('Server offline — could not analyze');
    dz.querySelector('.drop-txt').innerHTML = `<strong>${name}</strong>offline`;
  }
}

// ── Compatibility check ───────────────────────────────────────────
async function checkCompat() {
  if (!songA || !songB || !songA.bpm_raw || !songB.bpm_raw) return;
  try {
    const r = await fetch(`${API}/compatibility`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        camelot_a: songA.key, camelot_b: songB.key,
        bpm_a: songA.bpm_raw, bpm_b: songB.bpm_raw
      })
    });
    const d = await r.json();
    const pct = Math.round((d.combined ?? d.key_score ?? 0) * 100);
    document.getElementById('cbar').style.width = pct + '%';
    document.getElementById('cval').textContent  = pct + '%';

    const badge = document.getElementById('tm-badge');
    if (d.predicted_transition === 'TIGHT') {
      badge.textContent = '⚡ TIGHT TRANSITION';
      badge.className   = 'tm-badge tight';
    } else {
      badge.textContent = '〰 LOOSE TRANSITION';
      badge.className   = 'tm-badge loose';
    }
  } catch(_) {}
}

// ── Update song cards ─────────────────────────────────────────────
function updateCards() {
  if (songA) {
    document.getElementById('title-a').textContent = songA.name;
    document.getElementById('bpm-a').textContent   = songA.bpm;
    document.getElementById('key-a').textContent   = songA.key;
    document.getElementById('card-b').style.opacity = '1';
  }
  if (songB) {
    document.getElementById('title-b').textContent = songB.name;
    document.getElementById('bpm-b').textContent   = songB.bpm;
    document.getElementById('key-b').textContent   = songB.key;
    document.getElementById('card-b').style.opacity = '1';
  }
}

// ── Mix now ───────────────────────────────────────────────────────
async function triggerMix() {
  if (!songA || !songB) { toast('Drop two songs first'); return; }
  showLoading('Analyzing BPM and key compatibility...');
  toast(`Mixing: ${songA.name} → ${songB.name}`);

  try {
    const r = await fetch(`${API}/mix/start`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ file_a: songA.filename, file_b: songB.filename })
    });
    const d = await r.json();
    if (d.error) { hideLoading(); toast(`Error: ${d.error}`); return; }
    pollMix(d.job_id);
  } catch(e) {
    hideLoading(); toast(`Mix error: ${e.message}`);
  }
}

function pollMix(jobId) {
  const stages = [
    'Analyzing BPM and key compatibility...',
    'Running DEMUCS stem separation on Song 1...',
    'Running DEMUCS stem separation on Song 2...',
    'Matching BPM between songs...',
    'Building transition mix...',
    'Normalizing and saving output...',
  ];
  let si = 0;
  const stageInt = setInterval(() => {
    si = Math.min(si + 1, stages.length - 1);
    setStage(stages[si]);
  }, 8000);

  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const r = await fetch(`${API}/mix/status/${jobId}`);
      const d = await r.json();
      if (d.status === 'done') {
        clearInterval(pollTimer); clearInterval(stageInt);
        hideLoading();
        toast('✓ Mix ready! Press PLAY');
        loadAudio(d.output_file, 'outputs');
      } else if (d.status === 'error') {
        clearInterval(pollTimer); clearInterval(stageInt);
        hideLoading();
        toast(`Mix failed: ${d.error}`);
      }
    } catch(_) {}
  }, 2000);
}

// ── Audio ─────────────────────────────────────────────────────────
function loadAudio(filename, folder) {
  if (audioEl) { audioEl.pause(); audioEl.src = ''; }
  audioEl = new Audio(`${API}/stream/${folder}/${encodeURIComponent(filename)}`);
  audioEl.crossOrigin = 'anonymous';
  setupAudio(audioEl);
  document.getElementById('play-btn').disabled = false;
  toast('Mix loaded — press PLAY');
}

function setupAudio(el) {
  if (audioCtx) { try { audioCtx.close(); } catch(_) {} }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();

  analyserB = audioCtx.createAnalyser(); analyserB.fftSize = 2048; analyserB.smoothingTimeConstant = 0.82;
  analyserV = audioCtx.createAnalyser(); analyserV.fftSize = 2048; analyserV.smoothingTimeConstant = 0.75;
  analyserD = audioCtx.createAnalyser(); analyserD.fftSize = 2048; analyserD.smoothingTimeConstant = 0.65;

  const src = audioCtx.createMediaElementSource(el);
  const lpB = audioCtx.createBiquadFilter(); lpB.type='lowpass';  lpB.frequency.value=250;
  const hpV = audioCtx.createBiquadFilter(); hpV.type='highpass'; hpV.frequency.value=300;
  const lpV = audioCtx.createBiquadFilter(); lpV.type='lowpass';  lpV.frequency.value=3000;
  const hpD = audioCtx.createBiquadFilter(); hpD.type='highpass'; hpD.frequency.value=4000;

  src.connect(lpB); lpB.connect(analyserB); analyserB.connect(audioCtx.destination);
  src.connect(hpV); hpV.connect(lpV); lpV.connect(analyserV);
  src.connect(hpD); hpD.connect(analyserD);
}

function getEnergy(analyser, s, e) {
  if (!analyser) return 0;
  const buf = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(buf);
  return buf.slice(s,e).reduce((a,b)=>a+b,0)/((e-s)*255);
}

// ── Play / Pause ──────────────────────────────────────────────────
function togglePlay() {
  if (!audioEl) return;
  isPlaying = !isPlaying;
  const btn = document.getElementById('play-btn');
  if (isPlaying) {
    btn.textContent = '⏸ PAUSE';
    document.getElementById('va').classList.add('spin');
    document.getElementById('vb').classList.add('spin');
    const bpm = songA?.bpm_raw || 120;
    document.getElementById('va').style.animationDuration = ((60/bpm)*4) + 's';
    document.getElementById('vb').style.animationDuration = ((60/bpm)*4) + 's';
    if (audioCtx?.state==='suspended') audioCtx.resume();
    audioEl.play().catch(()=>{});
    resizeCanvases();
    if (!rafId) rafId = requestAnimationFrame(animLoop);
  } else {
    btn.textContent = '▶ PLAY';
    document.getElementById('va').classList.remove('spin');
    document.getElementById('vb').classList.remove('spin');
    audioEl.pause();
    if (audioCtx) audioCtx.suspend();
    stopViz();
  }
}

// ── Visualizer ────────────────────────────────────────────────────
function resizeCanvases() {
  ['cv-bass','cv-vocals','cv-drums'].forEach(id => {
    const c = document.getElementById(id);
    if (c) { c.width = c.offsetWidth; c.height = c.offsetHeight; }
  });
}

function push(arr, val) {
  arr.copyWithin(0,1);
  arr[arr.length-1] = Math.max(0, Math.min(1, val));
}

function drawWave(id, hist, color) {
  const c = document.getElementById(id);
  if (!c) return;
  const ctx = c.getContext('2d'), W = c.width, H = c.height;
  if (!W||!H) return;
  ctx.clearRect(0,0,W,H);

  // Build smooth bezier curve points
  const pts = hist.map((v,i) => ({
    x: (i/(hist.length-1))*W,
    y: H - v*H*0.86 - 2
  }));

  // Gradient fill under curve
  const grad = ctx.createLinearGradient(0,0,0,H);
  grad.addColorStop(0, color+'44'); grad.addColorStop(1, color+'00');

  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i=1; i<pts.length-1; i++) {
    const cpx = (pts[i].x + pts[i+1].x) / 2;
    const cpy = (pts[i].y + pts[i+1].y) / 2;
    ctx.quadraticCurveTo(pts[i].x, pts[i].y, cpx, cpy);
  }
  ctx.lineTo(pts[pts.length-1].x, pts[pts.length-1].y);
  ctx.lineTo(W,H); ctx.lineTo(0,H); ctx.closePath();
  ctx.fillStyle=grad; ctx.fill();

  // Smooth glowing line
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i=1; i<pts.length-1; i++) {
    const cpx = (pts[i].x + pts[i+1].x) / 2;
    const cpy = (pts[i].y + pts[i+1].y) / 2;
    ctx.quadraticCurveTo(pts[i].x, pts[i].y, cpx, cpy);
  }
  ctx.lineTo(pts[pts.length-1].x, pts[pts.length-1].y);
  ctx.strokeStyle=color; ctx.lineWidth=1.5;
  ctx.shadowColor=color; ctx.shadowBlur=8; ctx.stroke(); ctx.shadowBlur=0;

  // Live dot at right edge
  const last = pts[pts.length-1];
  ctx.beginPath(); ctx.arc(last.x-2, last.y, 3.5, 0, Math.PI*2);
  ctx.fillStyle=color; ctx.shadowColor=color; ctx.shadowBlur=14;
  ctx.fill(); ctx.shadowBlur=0;
}

function isBeat(bpm) {
  if (!bpm || !audioCtx) return false;
  const now = audioCtx.currentTime;
  const interval = 60/bpm;
  if (now - lastBeatTime >= interval) { lastBeatTime = now; return true; }
  return false;
}

function animLoop() {
  const bpm    = songA?.bpm_raw || 120;
  const onBeat = isBeat(bpm);
  const rawB   = getEnergy(analyserB, 0, 11);
  const rawV   = getEnergy(analyserV, 14, 140);
  const rawD   = getEnergy(analyserD, 186, 512);

  if (onBeat) {
    beatDecayB = Math.max(rawB*1.4, 0.85);
    beatDecayV = Math.max(rawV*1.4, 0.7);
    beatDecayD = Math.max(rawD*1.4, 0.9);
  }
  beatDecayB = Math.max(rawB, beatDecayB*0.82);
  beatDecayV = Math.max(rawV, beatDecayV*0.78);
  beatDecayD = Math.max(rawD, beatDecayD*0.75);

  push(hBass,   Math.min(1, rawB*0.4 + beatDecayB*0.6));
  push(hVocals, Math.min(1, rawV*0.5 + beatDecayV*0.5));
  push(hDrums,  Math.min(1, rawD*0.3 + beatDecayD*0.7));

  drawWave('cv-bass',   hBass,   '#ff3232');
  drawWave('cv-vocals', hVocals, '#00e5ff');
  drawWave('cv-drums',  hDrums,  '#c044ff');

  rafId = requestAnimationFrame(animLoop);
}

function stopViz() {
  if (rafId) { cancelAnimationFrame(rafId); rafId=null; }
  hBass.fill(0); hVocals.fill(0); hDrums.fill(0);
  ['cv-bass','cv-vocals','cv-drums'].forEach(id=>{
    const c=document.getElementById(id);
    if(c) c.getContext('2d').clearRect(0,0,c.width,c.height);
  });
}

// ── Toast ─────────────────────────────────────────────────────────
let toastT = null;
function toast(msg) {
  const el = document.getElementById('toast');
  el.textContent=msg; el.classList.add('show');
  clearTimeout(toastT);
  toastT = setTimeout(()=>el.classList.remove('show'), 3200);
}

window.addEventListener('resize', resizeCanvases);
resizeCanvases();
</script>
</body>
</html>
"""

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write(HTML)

print(f"\n✓ Visualizer generated → {OUTPUT_PATH}")
print("  1. Run: python server.py")
print("  2. Open the HTML in Chrome")
print("  3. Drop two WAV files → the page title and filename will update to match")
print("  4. Press MIX NOW → loading screen while demucs runs")
print("  5. Press PLAY → vinyls spin, curves react to the beat\n")
