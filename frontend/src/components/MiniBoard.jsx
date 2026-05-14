import { COLORS } from "../lib/colors";

/**
 * Compact, non-interactive board used inside cards and previews.
 * Cells are smaller and there are no territory pulses or labels.
 */
export default function MiniBoard({ board, cellSize = 16 }) {
  if (!board || board.length === 0) return null;
  return (
    <div className="inline-block overflow-hidden rounded-lg border border-line dark:border-dark-line">
      {board.map((row, r) => (
        <div key={r} className="flex">
          {row?.map((c, i) => (
            <div
              key={i}
              style={{
                width: cellSize,
                height: cellSize,
                backgroundColor: COLORS[c],
              }}
            />
          ))}
        </div>
      ))}
    </div>
  );
}
