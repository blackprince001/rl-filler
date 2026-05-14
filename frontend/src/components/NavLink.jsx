import { navigate } from "../lib/router";

export default function NavLink({ to, children, active = false }) {
  return (
    <button
      onClick={() => navigate(to)}
      className={`rounded-lg border px-3 py-1 font-mono text-[10px] uppercase tracking-wider transition-colors ${
        active
          ? "border-ink text-ink dark:border-dark-ink dark:text-dark-ink"
          : "border-line text-muted hover:border-ink hover:text-ink dark:border-dark-line dark:text-dark-muted dark:hover:border-dark-muted dark:hover:text-dark-ink"
      }`}
    >
      {children}
    </button>
  );
}
