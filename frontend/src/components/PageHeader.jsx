export default function PageHeader({ left, right, children }) {
  return (
    <header className="px-4 py-3 sm:px-6 sm:py-4">
      <div className="flex flex-col gap-2 sm:hidden">
        <div className="text-center font-mono text-base font-semibold uppercase tracking-widest">
          {children}
        </div>
        {(left || right) && (
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">{left ?? <span />}</div>
            <div className="flex items-center gap-2">{right ?? <span />}</div>
          </div>
        )}
      </div>

      <div className="hidden sm:flex sm:items-center sm:justify-between">
        <div className="flex flex-1 items-center gap-2">{left}</div>
        <div className="font-mono text-base font-semibold uppercase tracking-widest">
          {children}
        </div>
        <div className="flex flex-1 items-center justify-end gap-2">{right}</div>
      </div>
    </header>
  );
}
