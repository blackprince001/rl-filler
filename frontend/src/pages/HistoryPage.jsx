import { useMemo, useState } from "react";
import PageHeader from "../components/PageHeader";
import Footer from "../components/Footer";
import ThemeToggle from "../components/ThemeToggle";
import NavLink from "../components/NavLink";
import MiniBoard from "../components/MiniBoard";
import { loadMatches, deleteMatch, clearMatches } from "../lib/storage";
import { navigate } from "../lib/router";

const OUTCOMES = [
  { value: "any", label: "any" },
  { value: "won", label: "won" },
  { value: "lost", label: "lost" },
  { value: "tie", label: "tie" },
];

function timeAgo(ms) {
  if (!ms) return "—";
  const s = Math.floor((Date.now() - ms) / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

function outcomeMeta(winner) {
  if (winner === "human") return { text: "you won", tone: "text-[var(--you-stroke)]" };
  if (winner === "ai")    return { text: "you lost", tone: "text-[var(--ai-stroke)]" };
  return { text: "tie", tone: "text-muted dark:text-dark-muted" };
}

function FilterRow({ label, options, value, onChange }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-20 shrink-0 font-mono text-[10px] uppercase tracking-wider text-muted dark:text-dark-muted">
        {label}
      </span>
      <div className="flex flex-wrap gap-1.5">
        {options.map((o) => {
          const active = o.value === value;
          return (
            <button
              key={o.value}
              onClick={() => onChange(o.value)}
              className={`rounded-full border px-2.5 py-0.5 font-mono text-[11px] transition-colors ${
                active
                  ? "border-ink bg-ink text-white dark:border-dark-ink dark:bg-dark-ink dark:text-dark-bg"
                  : "border-line text-muted hover:border-muted hover:text-ink dark:border-dark-line dark:text-dark-muted dark:hover:border-dark-muted dark:hover:text-dark-ink"
              }`}
            >
              {o.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default function HistoryPage() {
  const [matches, setMatches] = useState(loadMatches);
  const [outcomeFilter, setOutcomeFilter] = useState("any");

  const filtered = useMemo(() => {
    return matches.filter((m) => {
      if (outcomeFilter === "any") return true;
      if (outcomeFilter === "won") return m.winner === "human";
      if (outcomeFilter === "lost") return m.winner === "ai";
      if (outcomeFilter === "tie") return m.winner === "tie";
      return true;
    });
  }, [matches, outcomeFilter]);

  function handleDelete(id, e) {
    e.stopPropagation();
    setMatches(deleteMatch(id));
  }

  function handleClearAll() {
    if (!confirm("Delete all stored matches? This cannot be undone.")) return;
    clearMatches();
    setMatches([]);
  }

  return (
    <div className="min-h-screen bg-white text-ink dark:bg-dark-bg dark:text-dark-ink">
      <PageHeader
        left={<NavLink to="/play">← play</NavLink>}
        right={<><NavLink to="/stats">stats →</NavLink><ThemeToggle /></>}
      >
        Game history
      </PageHeader>

      <main className="mx-auto max-w-6xl px-4 pb-12 sm:px-6">
        {matches.length === 0 ? (
          <div className="flex justify-center pt-20 font-mono text-sm text-muted dark:text-dark-muted">
            No completed games yet. Finish one on the play page and it'll appear here.
          </div>
        ) : (
          <>
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <FilterRow
                label="outcome"
                options={OUTCOMES}
                value={outcomeFilter}
                onChange={setOutcomeFilter}
              />
              <div className="flex items-center gap-3 font-mono text-[11px] text-muted dark:text-dark-muted">
                <span>
                  {outcomeFilter === "any"
                    ? `${matches.length} ${matches.length === 1 ? "match" : "matches"}`
                    : `${filtered.length} of ${matches.length}`}
                </span>
                <button
                  onClick={handleClearAll}
                  className="rounded-lg border border-line px-3 py-1 uppercase tracking-wider transition-colors hover:border-ink hover:text-ink dark:border-dark-line dark:hover:border-dark-muted dark:hover:text-dark-ink"
                >
                  clear all
                </button>
              </div>
            </div>

            {filtered.length === 0 ? (
              <div className="flex justify-center pt-12 font-mono text-sm text-muted dark:text-dark-muted">
                No games match the current filter.
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
                {filtered.map((m) => {
                  const out = outcomeMeta(m.winner);
                  const [h, a] = m.final_scores || [0, 0];
                  return (
                    <div
                      key={m.id}
                      role="button"
                      tabIndex={0}
                      onClick={() => navigate(`/replay?id=${m.id}`)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") navigate(`/replay?id=${m.id}`);
                      }}
                      className="group flex cursor-pointer flex-col gap-3 overflow-hidden rounded-2xl border border-line bg-white/40 p-4 text-left transition-colors hover:border-ink dark:border-dark-line dark:bg-white/0 dark:hover:border-dark-muted"
                    >
                      <div className="flex items-baseline justify-between font-mono text-[11px]">
                        <span className="text-ink dark:text-dark-ink">vs DQN</span>
                        <span className="text-muted dark:text-dark-muted">
                          {timeAgo(m.ended_at || m.started_at)}
                        </span>
                      </div>

                      <div className="flex justify-center rounded-xl bg-canvas/40 p-3 dark:bg-dark-bg/40">
                        <MiniBoard board={m.final_board || m.initial_board} />
                      </div>

                      <div className="flex items-baseline justify-between font-mono text-[11px]">
                        <span className={`uppercase tracking-wider ${out.tone}`}>{out.text}</span>
                        <span className="tabular-nums text-muted dark:text-dark-muted">
                          {h}–{a} · {m.turns}t
                        </span>
                      </div>

                      <div className="flex items-baseline justify-between font-mono text-[10px] text-muted dark:text-dark-muted">
                        <button
                          onClick={(e) => handleDelete(m.id, e)}
                          className="uppercase tracking-wider transition-colors hover:text-ink dark:hover:text-dark-ink"
                        >
                          delete
                        </button>
                        <span className="uppercase tracking-wider opacity-0 transition-opacity group-hover:opacity-100">
                          replay →
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </>
        )}

        <Footer className="mt-10" />
      </main>
    </div>
  );
}
