export default function Footer({ className = "" }) {
  return (
    <footer
      className={`flex items-center justify-center gap-2 py-4 font-mono text-[10px] uppercase tracking-widest text-muted dark:text-dark-muted ${className}`}
    >
      <span>Flood-It · DQN</span>
      <span className="opacity-50">·</span>
      <a
        href="https://github.com/blackprince001"
        target="_blank"
        rel="noreferrer"
        className="transition-colors hover:text-ink dark:hover:text-dark-ink"
      >
        @blackprince001
      </a>
    </footer>
  );
}
