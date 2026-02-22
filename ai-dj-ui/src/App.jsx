import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Music2, ListMusic, Pause, Play, Upload, X } from "lucide-react";

// If you have shadcn/ui available in your project, you can swap these simple components
// for shadcn ones (Button, Card, Sheet/Drawer, Progress, etc.).

function formatTime(sec) {
  if (!Number.isFinite(sec)) return "0:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

// A light-weight, no-lib bass/energy visualizer:
// - If WebAudio analyser is available, we drive a moving line from low-frequency energy.
// - Otherwise, we fall back to a tempo-driven metronome-ish motion.
function useBassOrTempoDriver({ audioRef, bpm, isPlaying }) {
  const [value, setValue] = useState(0);
  const rafRef = useRef(null);
  const ctxRef = useRef(null);
  const analyserRef = useRef(null);
  const dataRef = useRef(null);

  useEffect(() => {
    const audioEl = audioRef.current;
    if (!audioEl) return;


    async function setup() {
      try {
        // Some browsers block AudioContext until user interaction.
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (!AudioContext) return;

        const ctx = new AudioContext();
        ctxRef.current = ctx;
        const src = ctx.createMediaElementSource(audioEl);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 2048;

        src.connect(analyser);
        analyser.connect(ctx.destination);

        analyserRef.current = analyser;
        dataRef.current = new Uint8Array(analyser.frequencyBinCount);
      } catch {
        // ignore: fallback to tempo driver
      }
    }

    setup();

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      // We intentionally do not close AudioContext here because
      // some setups re-mount components; closing can break later playback.
    };
  }, [audioRef]);

  useEffect(() => {
    const analyser = analyserRef.current;
    const data = dataRef.current;

    const start = performance.now();

    const tick = (t) => {
      if (!isPlaying) {
        setValue(0);
        rafRef.current = requestAnimationFrame(tick);
        return;
      }

      // If we have analyser, approximate bass energy from the lowest bins.
      if (analyser && data) {
        analyser.getByteFrequencyData(data);
        const bins = Math.max(8, Math.floor(data.length * 0.06)); // ~lowest 6%
        let sum = 0;
        for (let i = 0; i < bins; i++) sum += data[i];
        const bass = sum / (bins * 255); // 0..1
        // Smooth it a bit
        setValue((prev) => prev * 0.8 + bass * 0.2);
      } else {
        // Fallback: tempo-based pulse.
        const bps = Math.max(1, bpm / 60);
        const phase = ((t - start) / 1000) * bps;
        const pulse = 0.5 + 0.5 * Math.sin(phase * Math.PI * 2);
        setValue((prev) => prev * 0.8 + pulse * 0.2);
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [bpm, isPlaying]);

  return value; // 0..1
}

function Vinyl({ bpm = 120, isPlaying, label = "Now" }) {
  // Vinyl rotation speed (roughly): 33.3 rpm default.
  // We modulate slightly with bpm so users *feel* tempo changes.
  const rpm = useMemo(() => {
    const base = 33.3;
    const mod = clamp((bpm - 120) / 120, -0.35, 0.35); // -35%..+35%
    return base * (1 + mod);
  }, [bpm]);

  const secondsPerRev = 60 / rpm;

  return (
    <div className="relative flex flex-col items-center gap-2">
      <div className="text-xs uppercase tracking-wider text-neutral-400">{label}</div>
      <div className="relative">
        <motion.div
          className="h-44 w-44 rounded-full bg-neutral-900 shadow-[0_20px_60px_rgba(0,0,0,0.45)] ring-1 ring-neutral-800"
          animate={isPlaying ? { rotate: 360 } : { rotate: 0 }}
          transition={
            isPlaying
              ? { repeat: Infinity, ease: "linear", duration: secondsPerRev }
              : { duration: 0.3 }
          }
        >
          {/* grooves */}
          <div className="absolute inset-3 rounded-full ring-1 ring-neutral-800" />
          <div className="absolute inset-6 rounded-full ring-1 ring-neutral-800" />
          <div className="absolute inset-9 rounded-full ring-1 ring-neutral-800" />
          <div className="absolute inset-12 rounded-full ring-1 ring-neutral-800" />

          {/* label */}
          <div className="absolute inset-0 grid place-items-center">
            <div className="h-14 w-14 rounded-full bg-neutral-200/10 ring-1 ring-neutral-700" />
            <div className="absolute h-2 w-2 rounded-full bg-neutral-200/50" />
          </div>
        </motion.div>

        {/* tonearm */}
        <motion.div
          className="absolute -right-10 top-6 h-28 w-28"
          initial={false}
          animate={isPlaying ? { rotate: -10 } : { rotate: -30 }}
          transition={{ type: "spring", stiffness: 180, damping: 18 }}
        >
          <div className="absolute right-0 top-0 h-2 w-20 rounded-full bg-neutral-700" />
          <div className="absolute right-16 top-1 h-20 w-2 rounded-full bg-neutral-700" />
          <div className="absolute right-14 top-18 h-5 w-6 rounded-md bg-neutral-600 ring-1 ring-neutral-500" />
        </motion.div>
      </div>
    </div>
  );
}

function MovingLine({ driver = 0 }) {
  // driver is 0..1. We'll shift a highlight across a line.
  const x = `${clamp(driver, 0, 1) * 100}%`;
  return (
    <div className="w-full">
      <div className="flex items-center gap-2 text-xs text-neutral-400">
        <span className="inline-flex items-center gap-1"><Music2 className="h-4 w-4" />Energy</span>
        <span className="text-neutral-600">(bass or tempo)</span>
      </div>
      <div className="mt-2 h-10 w-full rounded-xl bg-neutral-900 ring-1 ring-neutral-800 overflow-hidden">
        <div className="relative h-full">
          {/* baseline */}
          <div className="absolute left-0 right-0 top-1/2 h-px -translate-y-1/2 bg-neutral-800" />
          {/* moving cursor */}
          <motion.div
            className="absolute top-0 h-full w-1.5 rounded-full bg-neutral-200/70"
            style={{ left: x }}
            transition={{ type: "tween", duration: 0.08, ease: "linear" }}
          />
          {/* subtle glow that widens with energy */}
          <motion.div
            className="absolute top-0 h-full rounded-full bg-neutral-200/10"
            style={{ left: x }}
            animate={{ width: `${8 + driver * 80}px` }}
            transition={{ type: "tween", duration: 0.08, ease: "linear" }}
          />
        </div>
      </div>
    </div>
  );
}

function Dropzone({ onFiles }) {
  const [isOver, setIsOver] = useState(false);

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setIsOver(true);
      }}
      onDragLeave={() => setIsOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsOver(false);
        const files = Array.from(e.dataTransfer.files || []).filter((f) =>
          f.name.toLowerCase().endsWith(".wav")
        );
        onFiles?.(files);
      }}
      className={
        "rounded-2xl border border-dashed p-4 transition " +
        (isOver
          ? "border-neutral-200/70 bg-neutral-200/5"
          : "border-neutral-700 bg-neutral-950")
      }
    >
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 rounded-xl bg-neutral-900 p-2 ring-1 ring-neutral-800">
            <Upload className="h-5 w-5 text-neutral-200" />
          </div>
          <div>
            <div className="font-medium text-neutral-100">Drop .wav files to add to playlist</div>
            <div className="text-sm text-neutral-400">Reminder: don’t push test songs to GitHub • WAV only</div>
          </div>
        </div>
        <label className="cursor-pointer rounded-xl bg-neutral-200 px-3 py-2 text-sm font-medium text-neutral-900 hover:bg-neutral-100">
          Browse
          <input
            type="file"
            accept="audio/wav,.wav"
            multiple
            className="hidden"
            onChange={(e) => {
              const files = Array.from(e.target.files || []).filter((f) =>
                f.name.toLowerCase().endsWith(".wav")
              );
              onFiles?.(files);
              e.target.value = "";
            }}
          />
        </label>
      </div>
    </div>
  );
}

function QueuePanel({ open, onClose, playlist, onPick }) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="absolute right-0 top-0 h-full w-full max-w-md bg-neutral-950 ring-1 ring-neutral-800 shadow-2xl">
        <div className="flex items-center justify-between border-b border-neutral-800 p-4">
          <div>
            <div className="text-sm uppercase tracking-wider text-neutral-400">Queue</div>
            <div className="text-lg font-semibold text-neutral-100">Playlist</div>
          </div>
          <button
            onClick={onClose}
            className="rounded-xl p-2 text-neutral-300 hover:bg-neutral-900"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="p-3">
          {playlist.length === 0 ? (
            <div className="rounded-2xl border border-neutral-800 bg-neutral-900/30 p-4 text-neutral-300">
              No tracks yet. Drop a .wav file to add.
            </div>
          ) : (
            <div className="space-y-2">
              {playlist.map((t, idx) => (
                <button
                  key={t.id}
                  onClick={() => onPick?.(idx)}
                  className="w-full rounded-2xl border border-neutral-800 bg-neutral-900/30 p-3 text-left hover:bg-neutral-900"
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate font-medium text-neutral-100">{t.title || t.filename}</div>
                      <div className="mt-0.5 flex flex-wrap gap-x-3 gap-y-1 text-xs text-neutral-400">
                        <span>BPM {t.bpm ?? "—"}</span>
                        <span>Key {t.key ?? "—"}</span>
                        <span>Camelot {t.camelot ?? "—"}</span>
                      </div>
                    </div>
                    <div className="text-xs text-neutral-500">#{idx + 1}</div>
                  </div>
                </button>
              ))}
            </div>
          )}

          <div className="mt-4 rounded-2xl border border-neutral-800 bg-neutral-900/20 p-4">
            <div className="text-sm font-medium text-neutral-100">Workflow (what the backend will do)</div>
            <ol className="mt-2 list-decimal pl-5 text-sm text-neutral-400 space-y-1">
              <li>Collect metadata (tempo + key)</li>
              <li>Reorder playlist for easier transitions</li>
              <li>Segment into sections (intro/verse/chorus/...)</li>
              <li>Render EQ transition (swap low-end then high-end; time-stretch/pitch-shift as needed)</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function AIDJ_UI_Prototype() {
  // Mock playlist entries — in your real app these will come from your Python backend JSON metadata.
  const [playlist, setPlaylist] = useState([
    {
      id: "t1",
      filename: "song_a.wav",
      title: "Song A",
      artist: "—",
      bpm: 128,
      key: "A minor",
      camelot: "8A",
      sections: {
        intro: [0, 18.2],
        verse: [18.2, 48.1],
        chorus: [48.1, 78.0],
      },
    },
    {
      id: "t2",
      filename: "song_b.wav",
      title: "Song B",
      artist: "—",
      bpm: 124,
      key: "C major",
      camelot: "8B",
      sections: {
        intro: [0, 12.0],
        verse: [12.0, 44.0],
        chorus: [44.0, 74.0],
      },
    },
  ]);

  const [currentIdx, setCurrentIdx] = useState(0);
  const [nextIdx, setNextIdx] = useState(1);
  const [queueOpen, setQueueOpen] = useState(false);

  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef(null);

  const current = playlist[currentIdx] || null;
  const next = playlist[nextIdx] || null;

  const bpmNow = current?.bpm ?? 120;
  const driver = useBassOrTempoDriver({ audioRef, bpm: bpmNow, isPlaying });

  // Fake progress (replace with audio currentTime / duration once you play real audio)
  const [progress, setProgress] = useState(0);
  useEffect(() => {
    let id;
    if (isPlaying) {
      id = window.setInterval(() => {
        setProgress((p) => (p + 0.25) % 100);
      }, 250);
    }
    return () => {
      if (id) window.clearInterval(id);
    };
  }, [isPlaying]);

  function addFiles(files) {
    const newTracks = files.map((f) => ({
      id: `${f.name}-${crypto.randomUUID()}`,
      filename: f.name,
      title: f.name.replace(/\.wav$/i, ""),
      artist: "Local file",
      bpm: null,
      key: null,
      camelot: null,
      sections: null,
      file: f,
    }));
    setPlaylist((prev) => [...prev, ...newTracks]);
    // In real app:
    // 1) POST files to backend
    // 2) backend computes metadata JSON (bpm/key/sections)
    // 3) UI refreshes playlist with computed fields
  }

  function togglePlay() {
    setIsPlaying((v) => !v);
    // If you wire an <audio> element later:
    // const el = audioRef.current;
    // if (!el) return;
    // v ? el.pause() : el.play();
  }

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      {/* Top bar */}
      <div className="sticky top-0 z-40 border-b border-neutral-800 bg-neutral-950/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2">
            <div className="rounded-xl bg-neutral-200 p-2 text-neutral-900">
              <Music2 className="h-5 w-5" />
            </div>
            <div>
              <div className="text-sm uppercase tracking-wider text-neutral-400">AI DJ</div>
              <div className="text-base font-semibold">Section-Aware Mixing</div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setQueueOpen(true)}
              className="inline-flex items-center gap-2 rounded-xl bg-neutral-900 px-3 py-2 text-sm text-neutral-100 ring-1 ring-neutral-800 hover:bg-neutral-800"
            >
              <ListMusic className="h-4 w-4" /> Playlist
            </button>
            <button
              onClick={togglePlay}
              className="inline-flex items-center gap-2 rounded-xl bg-neutral-200 px-3 py-2 text-sm font-medium text-neutral-900 hover:bg-neutral-100"
            >
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              {isPlaying ? "Pause" : "Play"}
            </button>
          </div>
        </div>
      </div>

      {/* Main */}
      <div className="mx-auto grid max-w-6xl gap-6 px-4 py-6 lg:grid-cols-[1.2fr_0.8fr]">
        <div className="space-y-6">
          {/* Vinyls */}
          <div className="rounded-2xl border border-neutral-800 bg-neutral-900/20 p-5">
            <div className="flex flex-col gap-5">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-sm uppercase tracking-wider text-neutral-400">Now Mixing</div>
                  <div className="text-lg font-semibold text-neutral-100">
                    {current?.title ?? "No track"}
                    <span className="text-neutral-400 font-normal"> {current?.artist ? `• ${current.artist}` : ""}</span>
                  </div>
                </div>
                <div className="text-sm text-neutral-300">
                  <div className="text-neutral-400">Transition length</div>
                  <div className="font-semibold">8–16 bars</div>
                </div>
              </div>

              <div className="grid gap-6 md:grid-cols-2">
                <div className="flex items-center justify-center">
                  <Vinyl bpm={current?.bpm ?? 120} isPlaying={isPlaying} label="Song 1" />
                </div>
                <div className="flex items-center justify-center">
                  <Vinyl bpm={next?.bpm ?? 120} isPlaying={isPlaying} label="Song 2" />
                </div>
              </div>

              <MovingLine driver={driver} />

              {/* progress */}
              <div>
                <div className="flex items-center justify-between text-xs text-neutral-400">
                  <span>{formatTime((progress / 100) * 180)}</span>
                  <span>{formatTime(180)}</span>
                </div>
                <div className="mt-2 h-2 w-full rounded-full bg-neutral-900 ring-1 ring-neutral-800 overflow-hidden">
                  <div className="h-full bg-neutral-200" style={{ width: `${progress}%` }} />
                </div>
              </div>

              <div className="grid gap-3 md:grid-cols-2">
                <div className="rounded-2xl border border-neutral-800 bg-neutral-950 p-4">
                  <div className="text-sm font-semibold">Song 1 metadata</div>
                  <div className="mt-2 grid grid-cols-2 gap-2 text-sm text-neutral-300">
                    <div className="text-neutral-400">BPM</div><div className="text-right font-medium">{current?.bpm ?? "—"}</div>
                    <div className="text-neutral-400">Key</div><div className="text-right font-medium">{current?.key ?? "—"}</div>
                    <div className="text-neutral-400">Camelot</div><div className="text-right font-medium">{current?.camelot ?? "—"}</div>
                    <div className="text-neutral-400">Sections</div>
                    <div className="text-right font-medium">{current?.sections ? "intro/verse/chorus" : "—"}</div>
                  </div>
                </div>
                <div className="rounded-2xl border border-neutral-800 bg-neutral-950 p-4">
                  <div className="text-sm font-semibold">Song 2 metadata</div>
                  <div className="mt-2 grid grid-cols-2 gap-2 text-sm text-neutral-300">
                    <div className="text-neutral-400">BPM</div><div className="text-right font-medium">{next?.bpm ?? "—"}</div>
                    <div className="text-neutral-400">Key</div><div className="text-right font-medium">{next?.key ?? "—"}</div>
                    <div className="text-neutral-400">Camelot</div><div className="text-right font-medium">{next?.camelot ?? "—"}</div>
                    <div className="text-neutral-400">Sections</div>
                    <div className="text-right font-medium">{next?.sections ? "intro/verse/chorus" : "—"}</div>
                  </div>
                </div>
              </div>

              <div className="rounded-2xl border border-neutral-800 bg-neutral-950 p-4">
                <div className="text-sm font-semibold">Transition strategy</div>
                <div className="mt-2 text-sm text-neutral-300">
                  Your Python engine selects a strategy using chorus/verse timestamps (e.g., Chorus→Chorus, Verse→Verse) and renders an EQ-style mix (swap low-end, swap high-end, optional time-stretch/pitch-shift).
                </div>
                <div className="mt-3 grid gap-2 sm:grid-cols-2">
                  {[
                    { k: "Chorus → Chorus", v: "High Energy Drop Swap" },
                    { k: "Verse → Verse", v: "Smooth Blend" },
                    { k: "Chorus → Verse", v: "Energy Drop / Reset" },
                    { k: "Verse → Chorus", v: "Energy Build" },
                  ].map((x) => (
                    <div key={x.k} className="rounded-xl border border-neutral-800 bg-neutral-900/30 p-3">
                      <div className="text-sm font-medium text-neutral-100">{x.k}</div>
                      <div className="text-xs text-neutral-400">{x.v}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <Dropzone onFiles={addFiles} />

          {/* Hidden audio element placeholder (optional) */}
          <audio ref={audioRef} className="hidden" />
        </div>

        {/* Right column: queue preview + controls */}
        <div className="space-y-6">
          <div className="rounded-2xl border border-neutral-800 bg-neutral-900/20 p-5">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm uppercase tracking-wider text-neutral-400">Up Next</div>
                <div className="text-lg font-semibold">Queue Preview</div>
              </div>
              <button
                onClick={() => setQueueOpen(true)}
                className="inline-flex items-center gap-2 rounded-xl bg-neutral-900 px-3 py-2 text-sm ring-1 ring-neutral-800 hover:bg-neutral-800"
              >
                <ListMusic className="h-4 w-4" /> Open
              </button>
            </div>

            <div className="mt-4 space-y-2">
              {playlist.slice(0, 4).map((t, idx) => (
                <div
                  key={t.id}
                  className={
                    "rounded-2xl border p-3 " +
                    (idx === currentIdx
                      ? "border-neutral-200/40 bg-neutral-200/5"
                      : "border-neutral-800 bg-neutral-950")
                  }
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate font-medium">{t.title || t.filename}</div>
                      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-neutral-400">
                        <span>BPM {t.bpm ?? "—"}</span>
                        <span>Key {t.key ?? "—"}</span>
                        <span>Camelot {t.camelot ?? "—"}</span>
                      </div>
                    </div>
                    <button
                      onClick={() => {
                        setCurrentIdx(idx);
                        setNextIdx(clamp(idx + 1, 0, playlist.length - 1));
                      }}
                      className="rounded-xl bg-neutral-900 px-2 py-1 text-xs ring-1 ring-neutral-800 hover:bg-neutral-800"
                    >
                      Load
                    </button>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-4 rounded-2xl border border-neutral-800 bg-neutral-950 p-4">
              <div className="text-sm font-semibold">Notes</div>
              <ul className="mt-2 list-disc pl-5 text-sm text-neutral-400 space-y-1">
                <li>WAV-only drop zone to avoid format conversion issues.</li>
                <li>Don’t push test songs to GitHub (keep them local).</li>
                <li>Backend will compute metadata once and cache JSON.</li>
              </ul>
            </div>
          </div>

          <div className="rounded-2xl border border-neutral-800 bg-neutral-900/20 p-5">
            <div className="text-sm uppercase tracking-wider text-neutral-400">Backend API (suggested)</div>
            <div className="mt-2 rounded-2xl border border-neutral-800 bg-neutral-950 p-4 text-sm text-neutral-300">
              <div className="font-medium text-neutral-100">Suggested endpoints</div>
              <div className="mt-2 space-y-2 text-neutral-300">
                <div><span className="text-neutral-400">POST</span> /tracks (upload .wav)</div>
                <div><span className="text-neutral-400">GET</span> /tracks (list + metadata)</div>
                <div><span className="text-neutral-400">POST</span> /mix (songA, songB, strategy, bars) → returns mix.wav</div>
                <div><span className="text-neutral-400">GET</span> /queue (ordered playlist)</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <QueuePanel
        open={queueOpen}
        onClose={() => setQueueOpen(false)}
        playlist={playlist}
        onPick={(idx) => {
          setCurrentIdx(idx);
          setNextIdx(clamp(idx + 1, 0, playlist.length - 1));
          setQueueOpen(false);
        }}
      />
    </div>
  );
}
