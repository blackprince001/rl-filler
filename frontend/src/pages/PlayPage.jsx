import { useEffect, useRef, useState } from "react";
import Board from "../components/Board";
import AILog from "../components/AILog";
import PageHeader from "../components/PageHeader";
import Footer from "../components/Footer";
import ThemeToggle from "../components/ThemeToggle";
import NavLink from "../components/NavLink";
import { COLORS } from "../lib/colors";
import { WS_URL } from "../lib/config";
import { navigate } from "../lib/router";

export default function PlayPage() {
  const [board, setBoard] = useState([]);
  const [scores, setScores] = useState([0, 0]);
  const [status, setStatus] = useState("connecting…");
  const [conn, setConn] = useState("connecting");
  const [youTerritory, setYouTerritory] = useState([]);
  const [aiTerritory, setAiTerritory] = useState([]);
  const [lastAiMove, setLastAiMove] = useState(null);
  const [aiMoveLog, setAiMoveLog] = useState([]);
  const [gameOver, setGameOver] = useState(false);
  const [gameId, setGameId] = useState(null);

  const ws = useRef(null);
  const reconnectTimer = useRef(null);
  const aiTurnRef = useRef(0);

  useEffect(() => {
    function connect() {
      const sock = new WebSocket(WS_URL);
      ws.current = sock;
      setConn("connecting");
      sock.onopen = () => { setConn("open"); setStatus("your turn"); };
      sock.onclose = () => {
        setConn("closed");
        setStatus("disconnected");
        reconnectTimer.current = setTimeout(connect, 3000);
      };
      sock.onerror = () => setStatus("connection error");
      sock.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "INIT") {
          setBoard(msg.board);
          setScores(msg.scores);
          setYouTerritory(msg.you_territory || []);
          setAiTerritory(msg.ai_territory || []);
          setLastAiMove(msg.last_ai_move);
          setAiMoveLog([]);
          setGameOver(false);
          setStatus("your turn");
          setGameId(msg.game_id || null);
          aiTurnRef.current = 0;
        } else if (msg.type === "UPDATE") {
          setBoard(msg.board);
          setScores(msg.scores);
          setYouTerritory(msg.you_territory || []);
          setAiTerritory(msg.ai_territory || []);
          setLastAiMove(msg.last_ai_move);
          if (msg.ai_decision) {
            aiTurnRef.current += 1;
            const turn = aiTurnRef.current;
            setAiMoveLog((prev) => [
              ...prev,
              { move: msg.ai_decision.chosen_action, qValues: msg.ai_decision.q_values, turn },
            ]);
          }
          setStatus("your turn");
          setGameOver(false);
        } else if (msg.type === "GAME_OVER") {
          setBoard(msg.board);
          setScores(msg.scores);
          setYouTerritory(msg.you_territory || []);
          setAiTerritory(msg.ai_territory || []);
          setGameOver(true);
          if (msg.game_id) setGameId(msg.game_id);
          const [h, a] = msg.scores;
          const w = h > a ? "you win" : h < a ? "ai wins" : "tie";
          setStatus(`game over · ${w} · ${h}–${a}`);
        }
      };
    }
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      ws.current?.close();
    };
  }, []);

  function isColorDisabled(colorIndex) {
    // Human plays from the bottom-right corner. Disallow:
    //  - the AI's last move
    //  - the human's own current corner colour (no-op move)
    if (lastAiMove !== null && colorIndex === lastAiMove) return true;
    if (board.length > 0) {
      const last = board[board.length - 1];
      if (last && last[last.length - 1] === colorIndex) return true;
    }
    return false;
  }

  function handleMove(colorIndex) {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return;
    if (gameOver || isColorDisabled(colorIndex)) return;
    setStatus("ai thinking…");
    ws.current.send(JSON.stringify({ type: "MOVE", color: colorIndex }));
  }

  function handleReset() {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return;
    ws.current.send(JSON.stringify({ type: "RESET" }));
    setStatus("starting…");
  }

  const winnerLine = gameOver
    ? scores[0] > scores[1] ? "you win" : scores[0] < scores[1] ? "ai wins" : "tie"
    : null;

  return (
    <div className="flex min-h-screen flex-col bg-white text-ink dark:bg-dark-bg dark:text-dark-ink">
      <PageHeader
        left={<span className="font-mono text-[10px] uppercase tracking-widest text-muted dark:text-dark-muted">play</span>}
        right={
          <>
            <NavLink to="/history">history →</NavLink>
            <NavLink to="/stats">stats</NavLink>
            <ThemeToggle />
          </>
        }
      >
        <span className="flex items-center gap-3 text-2xl">
          <img src="/logo.png" alt="Flood-It" className="h-10 w-10 rounded-xl object-cover shadow-sm" />
          Flood-It - DQN
        </span>
      </PageHeader>

      <div className="flex flex-1 flex-col lg:grid lg:grid-cols-[18rem_1fr_18rem]">
        {/* Left sidebar */}
        <aside className="flex shrink-0 flex-col gap-3 p-4 font-mono text-[11px] leading-tight lg:p-6">
          <div className="rounded-xl border border-line p-3 dark:border-dark-line">
            <div className="text-muted dark:text-dark-muted"># Flood-It</div>
            <div>{conn === "open" ? "connected" : conn === "connecting" ? "connecting…" : "disconnected"}</div>
          </div>

          <div className="rounded-xl border border-line p-3 space-y-1 dark:border-dark-line">
            <div className="text-muted dark:text-dark-muted">Score</div>
            <div className="flex items-baseline justify-between">
              <span>You</span>
              <span className="tabular-nums" style={{ color: "var(--you-stroke)" }}>{scores[0]}</span>
            </div>
            <div className="flex items-baseline justify-between">
              <span>AI</span>
              <span className="tabular-nums" style={{ color: "var(--ai-stroke)" }}>{scores[1]}</span>
            </div>
          </div>

          <div className="rounded-xl border border-line p-3 dark:border-dark-line">
            <div className="text-muted dark:text-dark-muted">Status</div>
            <div>{status}</div>
          </div>

          <div className="rounded-xl border border-line p-3 text-[10px] leading-relaxed text-muted dark:border-dark-line dark:text-dark-muted">
            <div className="mb-1 font-medium text-ink dark:text-dark-ink">Rules</div>
            You flood from the <span style={{ color: "var(--you-stroke)" }}>bottom-right</span>; the AI floods from the <span style={{ color: "var(--ai-stroke)" }}>top-left</span>. Pick a color to claim every adjacent same-coloured tile. You can't pick the AI's last color or your own current one. Game ends when the board is filled — most tiles wins.
          </div>
        </aside>

        {/* Center */}
        <main className="flex flex-1 flex-col items-center justify-center gap-5 p-4 lg:p-8">
          <Board board={board} youTerritory={youTerritory} aiTerritory={aiTerritory} />

          <div className="flex flex-wrap items-center justify-center gap-2">
            {COLORS.map((color, idx) => {
              const disabled = isColorDisabled(idx) || gameOver;
              return (
                <button
                  key={idx}
                  onClick={() => handleMove(idx)}
                  disabled={disabled}
                  aria-label={`Play color ${idx + 1}`}
                  className={`relative h-12 w-12 rounded-lg border-2 transition-transform ${
                    disabled
                      ? "cursor-not-allowed opacity-30 grayscale"
                      : "border-line hover:-translate-y-0.5 hover:border-ink dark:border-dark-line dark:hover:border-dark-muted"
                  }`}
                  style={{ backgroundColor: color }}
                >
                  {disabled && !gameOver && (
                    <span className="pointer-events-none absolute inset-0 flex items-center justify-center text-white drop-shadow">✕</span>
                  )}
                </button>
              );
            })}
          </div>

          {gameOver && (
            <div className="flex flex-wrap items-center gap-4 font-mono text-sm">
              <span
                className="font-medium"
                style={{
                  color:
                    winnerLine === "you win"
                      ? "var(--you-stroke)"
                      : winnerLine === "ai wins"
                        ? "var(--ai-stroke)"
                        : undefined,
                }}
              >
                {winnerLine}
              </span>
              <span className="text-muted dark:text-dark-muted">you {scores[0]} · ai {scores[1]}</span>
              {gameId && <NavLink to={`/replay?id=${gameId}&scope=mine`}>replay this →</NavLink>}
              <NavLink to="/history">history →</NavLink>
            </div>
          )}

          <button
            onClick={handleReset}
            className="rounded-xl border border-line px-4 py-2 font-mono text-xs uppercase tracking-wide text-muted transition-colors hover:border-ink hover:text-ink active:scale-95 dark:border-dark-line dark:text-dark-muted dark:hover:border-dark-muted dark:hover:text-dark-ink"
          >
            {gameOver ? "new game" : "reset"}
          </button>
        </main>

        {/* Right sidebar */}
        <aside className="shrink-0 p-4 lg:p-6">
          <AILog entries={aiMoveLog} />
        </aside>
      </div>

      <Footer />
    </div>
  );
}
