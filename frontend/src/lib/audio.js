let ctx = null;
let muted = (typeof localStorage !== "undefined" && localStorage.getItem("rlfiller.muted") === "1") || false;
const subscribers = new Set();

function getCtx() {
  if (typeof window === "undefined") return null;
  if (!ctx) {
    const Ctor = window.AudioContext || window.webkitAudioContext;
    if (!Ctor) return null;
    ctx = new Ctor();
  }
  if (ctx.state === "suspended") void ctx.resume();
  return ctx;
}

export function isMuted() {
  return muted;
}

export function setMuted(v) {
  muted = v;
  try {
    localStorage.setItem("rlfiller.muted", v ? "1" : "0");
  } catch {
    // ignore
  }
  subscribers.forEach((fn) => fn(v));
}

export function subscribeMuted(fn) {
  subscribers.add(fn);
  return () => subscribers.delete(fn);
}

function vibrate(ms) {
  if (typeof navigator !== "undefined" && typeof navigator.vibrate === "function") {
    navigator.vibrate(ms);
  }
}

/** Light click when the player picks a colour. */
export function tick() {
  if (muted) return;
  const c = getCtx();
  if (!c) return;
  const t = c.currentTime;
  const osc = c.createOscillator();
  const filt = c.createBiquadFilter();
  const gain = c.createGain();

  osc.type = "triangle";
  const f0 = 760 + Math.random() * 120;
  osc.frequency.setValueAtTime(f0, t);
  osc.frequency.exponentialRampToValueAtTime(180, t + 0.04);

  filt.type = "bandpass";
  filt.frequency.value = 1100;
  filt.Q.value = 1.2;

  gain.gain.setValueAtTime(0.0001, t);
  gain.gain.exponentialRampToValueAtTime(0.18, t + 0.003);
  gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.06);

  osc.connect(filt).connect(gain).connect(c.destination);
  osc.start(t);
  osc.stop(t + 0.08);
  vibrate(4);
}

/** Heavier sweep when the AI plays. */
export function thock() {
  if (muted) return;
  const c = getCtx();
  if (!c) return;
  const t = c.currentTime;
  const osc = c.createOscillator();
  const gain = c.createGain();

  osc.type = "sine";
  osc.frequency.setValueAtTime(220, t);
  osc.frequency.exponentialRampToValueAtTime(80, t + 0.12);

  gain.gain.setValueAtTime(0.0001, t);
  gain.gain.exponentialRampToValueAtTime(0.24, t + 0.005);
  gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.22);

  osc.connect(gain).connect(c.destination);
  osc.start(t);
  osc.stop(t + 0.25);
  vibrate(12);
}

function chord(freqs, durations) {
  if (muted) return;
  const c = getCtx();
  if (!c) return;
  const start = c.currentTime;
  freqs.forEach((f, i) => {
    const t = start + (durations[i]?.delay ?? 0);
    const osc = c.createOscillator();
    const gain = c.createGain();
    osc.type = "triangle";
    osc.frequency.value = f;
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(0.18, t + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + (durations[i]?.dur ?? 0.4));
    osc.connect(gain).connect(c.destination);
    osc.start(t);
    osc.stop(t + (durations[i]?.dur ?? 0.4) + 0.05);
  });
}

/** Rising triad on win. */
export function winChime() {
  if (muted) return;
  chord(
    [523.25, 659.25, 783.99], // C5, E5, G5
    [
      { delay: 0.0, dur: 0.35 },
      { delay: 0.08, dur: 0.35 },
      { delay: 0.16, dur: 0.5 },
    ],
  );
  vibrate([20, 40, 30]);
}

/** Falling minor on loss. */
export function loseChime() {
  if (muted) return;
  chord(
    [392.0, 311.13, 233.08], // G4, Eb4, Bb3
    [
      { delay: 0.0, dur: 0.35 },
      { delay: 0.1, dur: 0.4 },
      { delay: 0.22, dur: 0.55 },
    ],
  );
  vibrate(40);
}

/** Neutral two-note for a tie. */
export function tieChime() {
  if (muted) return;
  chord(
    [440.0, 440.0],
    [
      { delay: 0.0, dur: 0.25 },
      { delay: 0.18, dur: 0.35 },
    ],
  );
  vibrate(20);
}

export function primeAudio() {
  getCtx();
}
