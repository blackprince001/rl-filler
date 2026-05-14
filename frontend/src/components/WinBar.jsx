export default function WinBar({ wins, draws, losses }) {
  const total = wins + draws + losses;
  if (total === 0) {
    return <div className="h-2 w-full rounded-full bg-line/40 dark:bg-dark-line/40" />;
  }
  return (
    <div className="flex h-2 w-full overflow-hidden rounded-full bg-line/30 dark:bg-dark-line/30">
      <div className="bg-[var(--you-stroke)]" style={{ width: `${(wins / total) * 100}%` }} />
      <div className="bg-muted/50 dark:bg-dark-muted/40" style={{ width: `${(draws / total) * 100}%` }} />
      <div className="bg-[var(--ai-stroke)]" style={{ width: `${(losses / total) * 100}%` }} />
    </div>
  );
}
