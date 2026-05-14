import { useEffect, useMemo, useRef, useState } from "react";
import Board from "../components/Board";
import AILog from "../components/AILog";
import PageHeader from "../components/PageHeader";
import Footer from "../components/Footer";
import ThemeToggle from "../components/ThemeToggle";
import NavLink from "../components/NavLink";
import Select from "../components/Select";

const SPEED_OPTIONS = [
  { value: "1500", label: "0.5×" },
  { value: "1000", label: "0.75×" },
  { value: "700", label: "1×" },
  { value: "400", label: "1.75×" },
  { value: "200", label: "3×" },
];
import { COLORS } from "../lib/colors";
import { getMatch } from "../lib/storage";

function buildFrames(match) {
  const frames = [{ board: match.initial_board, scores: [0, 0], move: null, moveIndex: -1 }];
  match.moves.forEach((m, i) => {
    frames.push({
      board: m.board_after,
      scores: m.scores_after || [0, 0],
      move: m,
      moveIndex: i,
    });
  });
  return frames;
}

export default function ReplayPage({ matchId }) {
  const match = useMemo(() => (matchId ? getMatch(matchId) : null), [matchId]);
  const frames = useMemo(() => (match ? buildFrames(match) : []), [match]);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(700);
  const timer = useRef(null);

  // AI-move entries visible up to and including the current frame.
  const aiEntriesUpToNow = useMemo(() => {
    if (!match) return [];
    const entries = [];
    let aiTurn = 0;
    for (let i = 0; i < idx; i++) {
      const m = match.moves[i];
      if (!m) continue;
      if (m.by === "ai" && Array.isArray(m.q_values)) {
        aiTurn += 1;
        entries.push({ move: m.color, qValues: m.q_values, turn: aiTurn });
      } else if (m.by === "ai") {
        aiTurn += 1;
      }
    }
    return entries;
  }, [match, idx]);

  useEffect(() => {
    if (!playing) return;
    if (idx >= frames.length - 1) { setPlaying(false); return; }
    timer.current = setTimeout(() => setIdx((i) => Math.min(i + 1, frames.length - 1)), speed);
    return () => clearTimeout(timer.current);
  }, [playing, idx, speed, frames.length]);

  if (!matchId || !match) {
    return (
      <div className="min-h-screen bg-white text-ink dark:bg-dark-bg dark:text-dark-ink">
        <PageHeader
          left={<NavLink to="/history">← history</NavLink>}
          right={<><NavLink to="/play">play</NavLink><ThemeToggle /></>}
        >
          Replay
        </PageHeader>
        <main className="mx-auto max-w-3xl px-6 pt-20 text-center font-mono text-sm text-muted dark:text-dark-muted">
          {!matchId ? "No match selected." : "Match not found — it may have been deleted."}
        </main>
      </div>
    );
  }

  const frame = frames[idx];
  const move = frame.move;
  const [h, a] = match.final_scores || [0, 0];
  const winnerTone =
    match.winner === "human" ? "var(--you-stroke)" :
    match.winner === "ai"    ? "var(--ai-stroke)" : undefined;
  const winnerText =
    match.winner === "human" ? "you won" :
    match.winner === "ai"    ? "you lost" : "tie";

  return (
    <div className="flex min-h-screen flex-col bg-white text-ink dark:bg-dark-bg dark:text-dark-ink">
      <PageHeader
        left={<NavLink to="/history">← history</NavLink>}
        right={<><NavLink to="/play">play</NavLink><ThemeToggle /></>}
      >
        Replay
      </PageHeader>

      <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col items-center gap-5 px-4 pb-12 sm:px-6">
        <div className="flex flex-wrap items-center justify-center gap-4 font-mono text-[11px]">
          <span className="text-muted dark:text-dark-muted">{match.turns} turns</span>
          <span className="text-muted dark:text-dark-muted">final {h}–{a}</span>
          <span style={{ color: winnerTone }}>{winnerText}</span>
        </div>

        <div className="flex items-center gap-6 font-mono text-sm">
          <span style={{ color: "var(--you-stroke)" }}>you {frame.scores[0]}</span>
          <span className="text-muted dark:text-dark-muted">·</span>
          <span style={{ color: "var(--ai-stroke)" }}>ai {frame.scores[1]}</span>
        </div>

        <Board board={frame.board} />

        <div className="flex flex-wrap items-center justify-center gap-2 font-mono text-[10px] uppercase tracking-wider">
          <button onClick={() => { setPlaying(false); setIdx(0); }} className="rounded-lg border border-line px-3 py-1 text-muted hover:border-ink hover:text-ink dark:border-dark-line dark:text-dark-muted dark:hover:border-dark-muted dark:hover:text-dark-ink">⏮</button>
          <button onClick={() => { setPlaying(false); setIdx((i) => Math.max(0, i - 1)); }} className="rounded-lg border border-line px-3 py-1 text-muted hover:border-ink hover:text-ink dark:border-dark-line dark:text-dark-muted dark:hover:border-dark-muted dark:hover:text-dark-ink">◀</button>
          <button
            onClick={() => setPlaying((p) => !p)}
            className="rounded-lg border border-ink bg-ink px-4 py-1 text-white dark:border-dark-ink dark:bg-dark-ink dark:text-dark-bg"
          >
            {playing ? "pause" : "play"}
          </button>
          <button onClick={() => { setPlaying(false); setIdx((i) => Math.min(frames.length - 1, i + 1)); }} className="rounded-lg border border-line px-3 py-1 text-muted hover:border-ink hover:text-ink dark:border-dark-line dark:text-dark-muted dark:hover:border-dark-muted dark:hover:text-dark-ink">▶</button>
          <button onClick={() => { setPlaying(false); setIdx(frames.length - 1); }} className="rounded-lg border border-line px-3 py-1 text-muted hover:border-ink hover:text-ink dark:border-dark-line dark:text-dark-muted dark:hover:border-dark-muted dark:hover:text-dark-ink">⏭</button>
          <Select
            ariaLabel="Playback speed"
            value={String(speed)}
            onChange={(v) => setSpeed(Number(v))}
            options={SPEED_OPTIONS}
          />
        </div>

        <input
          type="range"
          min={0}
          max={frames.length - 1}
          value={idx}
          onChange={(e) => { setPlaying(false); setIdx(Number(e.target.value)); }}
          className="w-full max-w-md accent-ink dark:accent-dark-ink"
        />

        <div className="flex min-h-7 items-center gap-2 font-mono text-[11px]">
          {idx === 0 ? (
            <span className="text-muted dark:text-dark-muted">initial position</span>
          ) : (
            <>
              <span className="text-muted dark:text-dark-muted">ply {idx} / {frames.length - 1}</span>
              <span className="opacity-50">·</span>
              <span style={{ color: move.by === "human" ? "var(--you-stroke)" : "var(--ai-stroke)" }}>
                {move.by === "human" ? "you" : "ai"} played
              </span>
              <span
                className="inline-block h-4 w-4 rounded border border-line dark:border-dark-line"
                style={{ backgroundColor: COLORS[move.color] }}
              />
              {move.by === "ai" && move.q_values && (
                <span className="tabular-nums text-muted dark:text-dark-muted">
                  · Q {move.q_values[move.color].toFixed(2)}
                </span>
              )}
            </>
          )}
        </div>

        {aiEntriesUpToNow.length > 0 && (
          <div className="mt-4 w-full">
            <AILog entries={aiEntriesUpToNow} initiallyExpanded />
          </div>
        )}

        <Footer className="mt-auto" />
      </main>
    </div>
  );
}
