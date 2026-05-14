const OPTS = [
  { value: "mine", label: "mine" },
  { value: "all", label: "all" },
];

export default function ScopeToggle({ scope, onChange }) {
  return (
    <div className="flex items-center gap-1 font-mono text-[10px]">
      {OPTS.map((o) => {
        const active = o.value === scope;
        return (
          <button
            key={o.value}
            onClick={() => onChange(o.value)}
            className={`rounded-full border px-2.5 py-0.5 transition-colors ${
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
  );
}
