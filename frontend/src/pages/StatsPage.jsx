import { useMemo } from "react";
import PageHeader from "../components/PageHeader";
import Footer from "../components/Footer";
import ThemeToggle from "../components/ThemeToggle";
import NavLink from "../components/NavLink";
import WinBar from "../components/WinBar";
import { loadMatches, computeStats } from "../lib/storage";

function pct(n, d) {
  if (!d) return "—";
  return `${Math.round((n / d) * 100)}%`;
}

function StatCard({ label, value, sub }) {
  return (
    <div className="rounded-2xl border border-line bg-white/40 p-4 dark:border-dark-line dark:bg-white/0">
      <div className="font-mono text-[10px] uppercase tracking-wider text-muted dark:text-dark-muted">{label}</div>
      <div className="mt-1 font-mono text-3xl font-medium tabular-nums">{value}</div>
      {sub && <div className="mt-0.5 font-mono text-[10px] text-muted dark:text-dark-muted">{sub}</div>}
    </div>
  );
}

export default function StatsPage() {
  const matches = useMemo(loadMatches, []);
  const stats = useMemo(() => computeStats(matches), [matches]);
  const last10 = useMemo(() => computeStats(matches.slice(0, 10)), [matches]);

  return (
    <div className="min-h-screen bg-white text-ink dark:bg-dark-bg dark:text-dark-ink">
      <PageHeader
        left={<NavLink to="/play">← play</NavLink>}
        right={<><NavLink to="/history">history →</NavLink><ThemeToggle /></>}
      >
        Analytics
      </PageHeader>

      <main className="mx-auto max-w-5xl px-4 pb-12 sm:px-6">
        {matches.length === 0 ? (
          <div className="pt-20 text-center font-mono text-sm text-muted dark:text-dark-muted">
            No completed games yet. Play a match and come back.
          </div>
        ) : (
          <div className="flex flex-col gap-8">
            <section>
              <h2 className="mb-3 font-mono text-xs uppercase tracking-wide text-muted dark:text-dark-muted">Lifetime</h2>
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <StatCard label="games" value={String(stats.games)} />
                <StatCard label="win rate" value={pct(stats.human_wins, stats.games)} sub={`${stats.human_wins}W ${stats.ai_wins}L ${stats.ties}T`} />
                <StatCard label="avg turns" value={String(stats.avg_turns)} />
                <StatCard label="avg margin" value={stats.avg_margin > 0 ? `+${stats.avg_margin}` : String(stats.avg_margin)} sub="you − ai" />
              </div>
            </section>

            <section className="rounded-2xl border border-line bg-white/40 p-5 dark:border-dark-line dark:bg-white/0">
              <h2 className="mb-3 font-mono text-xs uppercase tracking-wide text-muted dark:text-dark-muted">Outcomes</h2>
              <WinBar wins={stats.human_wins} draws={stats.ties} losses={stats.ai_wins} />
              <div className="mt-2 flex justify-between font-mono text-[11px] text-muted dark:text-dark-muted">
                <span style={{ color: "var(--you-stroke)" }}>you {stats.human_wins} · {pct(stats.human_wins, stats.games)}</span>
                <span>ties {stats.ties} · {pct(stats.ties, stats.games)}</span>
                <span style={{ color: "var(--ai-stroke)" }}>ai {stats.ai_wins} · {pct(stats.ai_wins, stats.games)}</span>
              </div>
            </section>

            <section>
              <h2 className="mb-3 font-mono text-xs uppercase tracking-wide text-muted dark:text-dark-muted">Last 10</h2>
              <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
                <StatCard label="win rate" value={pct(last10.human_wins, last10.games)} sub={`${last10.human_wins}W ${last10.ai_wins}L ${last10.ties}T`} />
                <StatCard label="avg turns" value={String(last10.avg_turns)} />
                <StatCard label="avg margin" value={last10.avg_margin > 0 ? `+${last10.avg_margin}` : String(last10.avg_margin)} />
              </div>
            </section>

            <section className="rounded-2xl border border-line bg-white/40 p-5 dark:border-dark-line dark:bg-white/0">
              <h2 className="mb-3 font-mono text-xs uppercase tracking-wide text-muted dark:text-dark-muted">Average score</h2>
              <div className="flex items-baseline justify-between font-mono text-sm">
                <span style={{ color: "var(--you-stroke)" }}>you {stats.avg_human_score}</span>
                <span className="text-muted dark:text-dark-muted">·</span>
                <span style={{ color: "var(--ai-stroke)" }}>ai {stats.avg_ai_score}</span>
              </div>
            </section>
          </div>
        )}

        <Footer className="mt-10" />
      </main>
    </div>
  );
}
