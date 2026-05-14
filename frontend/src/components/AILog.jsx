import { useState } from "react";
import { COLORS } from "../lib/colors";

function maxQ(qs) { return qs?.length ? Math.max(...qs) : 0; }
function minQ(qs) { return qs?.length ? Math.min(...qs) : 0; }

export default function AILog({ entries, initiallyExpanded = false }) {
  const [expanded, setExpanded] = useState(initiallyExpanded);
  if (!entries || entries.length === 0) return null;
  return (
    <div className="w-full rounded-2xl border border-line bg-white/40 p-4 dark:border-dark-line dark:bg-white/0">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center justify-between font-mono text-[11px] uppercase tracking-widest text-muted transition-colors hover:text-ink dark:text-dark-muted dark:hover:text-dark-ink"
      >
        <span>AI move history · {entries.length}</span>
        <span>{expanded ? "▼" : "▶"}</span>
      </button>

      {expanded && (
        <div className="mt-4 flex max-h-96 flex-col gap-3 overflow-y-auto pr-1">
          {entries.map((entry, idx) => {
            const lo = minQ(entry.qValues);
            const hi = maxQ(entry.qValues);
            return (
              <div
                key={idx}
                className="rounded-xl border border-line p-3 dark:border-dark-line"
              >
                <div className="mb-2 flex items-center justify-between font-mono text-[11px]">
                  <span className="text-ink dark:text-dark-ink">
                    Turn {entry.turn} · Color {entry.move + 1}
                  </span>
                  <span className="tabular-nums text-muted dark:text-dark-muted">
                    Q {entry.qValues[entry.move].toFixed(2)}
                  </span>
                </div>
                <div className="grid grid-cols-6 gap-1.5">
                  {entry.qValues.map((q, colorIdx) => {
                    const isChosen = colorIdx === entry.move;
                    const normalized = hi !== lo ? (q - lo) / (hi - lo) : 0.5;
                    return (
                      <div
                        key={colorIdx}
                        title={`Color ${colorIdx + 1}: ${q.toFixed(2)}`}
                        className={`flex flex-col items-center justify-end gap-1 rounded-md border p-1.5 ${
                          isChosen
                            ? "border-[var(--you-stroke)] bg-[var(--you-stroke)]/10"
                            : "border-line dark:border-dark-line"
                        }`}
                        style={{ minHeight: 72 }}
                      >
                        <div
                          className="w-full rounded-sm transition-[height]"
                          style={{
                            height: `${Math.max(18, normalized * 60)}px`,
                            backgroundColor: isChosen ? "var(--you-stroke)" : "#9a9a9a",
                          }}
                        />
                        <div
                          className="h-4 w-4 rounded border border-line dark:border-dark-line"
                          style={{ backgroundColor: COLORS[colorIdx] }}
                        />
                        <span className="font-mono text-[9px] tabular-nums text-muted dark:text-dark-muted">
                          {q.toFixed(1)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
