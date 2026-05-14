/**
 * On-theme styled native <select>. Uses `appearance-none` plus a chevron so
 * it inherits accessibility/keyboard behaviour while looking like the rest of
 * the bordered mono controls.
 */
export default function Select({
  value,
  onChange,
  options,
  ariaLabel,
  className = "",
}) {
  return (
    <div className={`relative inline-block ${className}`}>
      <select
        aria-label={ariaLabel}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="appearance-none rounded-lg border border-line bg-transparent py-1 pl-3 pr-7 font-mono text-[11px] uppercase tracking-wider text-muted transition-colors hover:border-ink hover:text-ink focus:border-ink focus:outline-none dark:border-dark-line dark:text-dark-muted dark:hover:border-dark-muted dark:hover:text-dark-ink dark:focus:border-dark-muted"
      >
        {options.map((o) => (
          <option
            key={String(o.value)}
            value={o.value}
            className="bg-white text-ink dark:bg-dark-bg dark:text-dark-ink"
          >
            {o.label}
          </option>
        ))}
      </select>
      <span
        aria-hidden
        className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 font-mono text-[9px] text-muted dark:text-dark-muted"
      >
        ▾
      </span>
    </div>
  );
}
