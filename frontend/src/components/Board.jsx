import { COLORS } from "../lib/colors";

/**
 * Board renders the flood-fill grid. The human plays from the bottom-right
 * corner, the AI from the top-left. Corner labels make the orientation
 * explicit, and each starting cell wears its owner's stroke.
 */
export default function Board({ board, youTerritory, aiTerritory }) {
  if (!board || board.length === 0) {
    return (
      <div className="flex min-h-[200px] items-center justify-center rounded-2xl border border-line p-6 font-mono text-xs text-muted dark:border-dark-line dark:text-dark-muted">
        loading board…
      </div>
    );
  }

  return (
    <div className="flex flex-col items-stretch gap-2">
      {/* Top label: AI corner */}
      <div className="flex justify-start pl-1 font-mono text-[10px] uppercase tracking-widest" style={{ color: "var(--ai-stroke)" }}>
        <span className="flex items-center gap-1">
          <span>AI starts here</span>
        </span>
      </div>

      <div className="inline-block self-center overflow-hidden rounded-xl border border-line shadow-sm dark:border-dark-line">
        {board.map((row, rIndex) => (
          <div key={rIndex} className="flex">
            {row?.map((colorCode, cIndex) => {
              const isYou = youTerritory?.[rIndex]?.[cIndex];
              const isAi = aiTerritory?.[rIndex]?.[cIndex];
              return (
                <div
                  key={cIndex}
                  className="relative h-[clamp(28px,4.5vw,40px)] w-[clamp(28px,4.5vw,40px)]"
                  style={{ backgroundColor: COLORS[colorCode] }}
                >
                  {isYou && (
                    <span
                      className="pointer-events-none absolute inset-0 border-2 animate-[territoryPulse_2s_ease-in-out_infinite]"
                      style={{ borderColor: "var(--you-stroke)" }}
                    />
                  )}
                  {isAi && (
                    <span
                      className="pointer-events-none absolute inset-0 border-2 animate-[territoryPulse_2s_ease-in-out_infinite]"
                      style={{ borderColor: "var(--ai-stroke)" }}
                    />
                  )}
                </div>
              );
            })}
          </div>
        ))}
      </div>

      {/* Bottom label: human corner */}
      <div className="flex justify-end pr-1 font-mono text-[11px] font-medium uppercase tracking-widest" style={{ color: "var(--you-stroke)" }}>
        <span className="flex items-center gap-1">
          <span>You start here</span>
        </span>
      </div>
    </div>
  );
}
