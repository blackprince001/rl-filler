import { useEffect, useState } from "react";
import PageHeader from "../components/PageHeader";
import Footer from "../components/Footer";
import ThemeToggle from "../components/ThemeToggle";
import NavLink from "../components/NavLink";
import WinBar from "../components/WinBar";
import ScopeToggle from "../components/ScopeToggle";
import { fetchStats } from "../lib/api";
import { navigate } from "../lib/router";

function pct(n, d) {
  if (!d) return "—";
  return `${Math.round((n / d) * 100)}%`;
}

function timeAgo(ms) {
  if (!ms) return "—";
  const s = Math.floor((Date.now() - ms) / 1000);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h`;
  return `${Math.floor(s / 86400)}d`;
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

export default function StatsPage({ scope }) {
  const [stats, setStats] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setStats(null);
    fetchStats({ scope })
      .then((s) => { if (!cancelled) setStats(s); })
      .catch(() => { if (!cancelled) setStats({ totals: { games: 0 }, recent: [] }); });
    return () => { cancelled = true; };
  }, [scope]);

  function changeScope(s) {
    navigate(`/stats?scope=${s}`);
  }

  const totals = stats?.totals;
  const games = totals?.games ?? 0;

  return (
    <div className="min-h-screen bg-white text-ink dark:bg-dark-bg dark:text-dark-ink">
      <PageHeader
        left={<NavLink to="/play">← play</NavLink>}
        right={<><NavLink to={`/history?scope=${scope}`}>history →</NavLink><ThemeToggle /></>}
      >
        Analytics
      </PageHeader>

      <main className="mx-auto max-w-5xl px-4 pb-12 sm:px-6">
        <div className="mb-6">
          <ScopeToggle scope={scope} onChange={changeScope} />
        </div>

        {stats === null && (
          <div className="flex justify-center pt-20 font-mono text-xs text-muted dark:text-dark-muted">
            loading…
          </div>
        )}

        {stats !== null && games === 0 && (
          <div className="flex justify-center pt-20 font-mono text-sm text-muted dark:text-dark-muted">
            {scope === "mine"
              ? "No completed games yet. Play a match and come back."
              : "No completed games recorded yet."}
          </div>
        )}

        {stats !== null && games > 0 && (
          <div className="flex flex-col gap-8">
            <section>
              <h2 className="mb-3 font-mono text-xs uppercase tracking-wide text-muted dark:text-dark-muted">Totals</h2>
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <StatCard label="games" value={String(games)} />
                <StatCard label="win rate" value={pct(totals.you_wins, games)} sub={`${totals.you_wins}W ${totals.ai_wins}L ${totals.ties}T`} />
                <StatCard label="avg turns" value={String(totals.avg_plies)} />
                <StatCard
                  label={scope === "all" ? "unique players" : "tie rate"}
                  value={scope === "all" ? String(totals.unique_clients) : pct(totals.ties, games)}
                />
              </div>
            </section>

            <section className="rounded-2xl border border-line bg-white/40 p-5 dark:border-dark-line dark:bg-white/0">
              <h2 className="mb-3 font-mono text-xs uppercase tracking-wide text-muted dark:text-dark-muted">Outcomes</h2>
              <WinBar wins={totals.you_wins} draws={totals.ties} losses={totals.ai_wins} />
              <div className="mt-2 flex justify-between font-mono text-[11px] text-muted dark:text-dark-muted">
                <span style={{ color: "var(--you-stroke)" }}>you {totals.you_wins} · {pct(totals.you_wins, games)}</span>
                <span>ties {totals.ties} · {pct(totals.ties, games)}</span>
                <span style={{ color: "var(--ai-stroke)" }}>ai {totals.ai_wins} · {pct(totals.ai_wins, games)}</span>
              </div>
            </section>

            <section className="rounded-2xl border border-line bg-white/40 p-5 dark:border-dark-line dark:bg-white/0">
              <h2 className="mb-3 font-mono text-xs uppercase tracking-wide text-muted dark:text-dark-muted">Average score</h2>
              <div className="flex items-baseline justify-between font-mono text-sm">
                <span style={{ color: "var(--you-stroke)" }}>you {totals.avg_you}</span>
                <span className="text-muted dark:text-dark-muted">·</span>
                <span style={{ color: "var(--ai-stroke)" }}>ai {totals.avg_ai}</span>
              </div>
            </section>

            {stats.recent?.length > 0 && (
              <section className="rounded-2xl border border-line bg-white/40 p-6 dark:border-dark-line dark:bg-white/0">
                <h2 className="mb-4 font-mono text-xs uppercase tracking-wide text-muted dark:text-dark-muted">Recent</h2>
                <div className="flex flex-col gap-1">
                  {stats.recent.map((r) => {
                    const tone =
                      r.winner === "human" ? "text-[var(--you-stroke)]"
                      : r.winner === "ai"  ? "text-[var(--ai-stroke)]"
                      : "text-muted dark:text-dark-muted";
                    const text =
                      r.winner === "human" ? "you won"
                      : r.winner === "ai"  ? "you lost"
                      : "tie";
                    return (
                      <button
                        key={r.game_id}
                        onClick={() => navigate(`/replay?id=${r.game_id}&scope=${scope}`)}
                        className="grid grid-cols-[1fr_auto] items-baseline gap-x-6 rounded-lg px-3 py-2.5 text-left font-mono text-[11px] transition-colors hover:bg-ink/5 dark:hover:bg-dark-ink/10"
                      >
                        <span className={`truncate ${tone}`}>{text}</span>
                        <span className="tabular-nums text-muted dark:text-dark-muted">
                          {r.final_scores[0]}–{r.final_scores[1]}
                          <span className="mx-1.5 opacity-50">·</span>
                          {r.plies}p
                          <span className="mx-1.5 opacity-50">·</span>
                          {timeAgo(r.ended_at)} ago
                        </span>
                      </button>
                    );
                  })}
                </div>
              </section>
            )}
          </div>
        )}

        <Footer className="mt-10" />
      </main>
    </div>
  );
}
