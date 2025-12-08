import React, { useEffect, useState, useRef } from 'react';
import './App.css';

const COLORS = ["#333", "#ff67ff", "#ff3838", "#47f847", "#5454ff", "#ffff26"];

function App() {
  const [board, setBoard] = useState([]);
  const [scores, setScores] = useState([0, 0]);
  const [status, setStatus] = useState("Connecting...");
  const [p1Territory, setP1Territory] = useState([]);
  const [p2Territory, setP2Territory] = useState([]);
  const [lastP1Move, setLastP1Move] = useState(null);
  const [lastP2Move, setLastP2Move] = useState(null);
  const [aiMoveLog, setAiMoveLog] = useState([]);
  const [aiLogExpanded, setAiLogExpanded] = useState(false);
  const [gameOver, setGameOver] = useState(false);
  const ws = useRef(null);

  useEffect(() => {
    ws.current = new WebSocket("ws://localhost:8000/ws/game");

    ws.current.onopen = () => setStatus("Connected. Your Turn.");
    ws.current.onclose = () => {
      setStatus("Disconnected");
      // Attempt to reconnect after a delay
      setTimeout(() => {
        if (ws.current?.readyState === WebSocket.CLOSED) {
          ws.current = new WebSocket("ws://localhost:8000/ws/game");
        }
      }, 3000);
    };
    ws.current.onerror = (error) => {
      console.error("WebSocket error:", error);
      setStatus("Connection error. Please check if the server is running.");
    };

    ws.current.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "UPDATE") {
        setBoard(msg.board);
        setScores(msg.scores);
        setP1Territory(msg.p1_territory || []);
        setP2Territory(msg.p2_territory || []);
        setLastP1Move(msg.last_p1_move);
        setLastP2Move(msg.last_p2_move);

        // Add AI decision to log
        if (msg.ai_decision) {
          setAiMoveLog(prev => [...prev, {
            move: msg.ai_decision.chosen_action,
            qValues: msg.ai_decision.q_values,
            turn: prev.length + 1
          }]);
        }

        setStatus("Your Turn");
        setGameOver(false);
      }
      if (msg.type === "GAME_OVER") {
        setBoard(msg.board);
        setScores(msg.scores);
        setP1Territory(msg.p1_territory || []);
        setP2Territory(msg.p2_territory || []);
        setGameOver(true);

        // Determine winner
        const winner = msg.scores[0] > msg.scores[1]
          ? "You Win!"
          : msg.scores[0] < msg.scores[1]
            ? "AI Wins!"
            : "It's a Tie!";
        setStatus(`Game Over! ${winner} (${msg.scores[0]} - ${msg.scores[1]})`);
      }
      if (msg.type === "INIT") {
        // Reset all state when receiving INIT (for new game)
        setBoard(msg.board);
        setScores(msg.scores);
        setP1Territory(msg.p1_territory || []);
        setP2Territory(msg.p2_territory || []);
        setLastP1Move(msg.last_p1_move);
        setLastP2Move(msg.last_p2_move);
        setAiMoveLog([]); // Clear AI move log
        setAiLogExpanded(false); // Collapse log
        setGameOver(false);
        setStatus("Your Turn");
      }
    };

    return () => ws.current.close();
  }, []);

  const handleMove = (colorIndex) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return;
    if (gameOver) return;

    // Check if this color is disabled
    if (isColorDisabled(colorIndex)) {
      return;
    }

    setStatus("AI Thinking...");
    ws.current.send(JSON.stringify({ type: "MOVE", color: colorIndex }));
  };

  const handleReset = () => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return;
    ws.current.send(JSON.stringify({ type: "RESET" }));
    setStatus("Starting new game...");
  };

  const isColorDisabled = (colorIndex) => {
    // Cannot pick AI's last move
    if (lastP2Move !== null && colorIndex === lastP2Move) {
      return true;
    }
    // Cannot pick your own current color (top-left corner)
    if (board.length > 0 && board[0] && board[0][0] === colorIndex) {
      return true;
    }
    return false;
  };

  const getMaxQValue = (qValues) => {
    if (!qValues || qValues.length === 0) return 0;
    return Math.max(...qValues);
  };

  const getMinQValue = (qValues) => {
    if (!qValues || qValues.length === 0) return 0;
    return Math.min(...qValues);
  };

  return (
    <div className="app-wrapper">
      <div className="game-container">
        <h1>Flood-It AI Battle</h1>
        <div className="scoreboard">
          <span>You: {scores[0]}</span>
          <span>AI: {scores[1]}</span>
        </div>
        <div className={`status ${gameOver ? 'game-over' : ''}`}>{status}</div>

        {board.length > 0 && (
          <button
            className="new-game-btn"
            onClick={handleReset}
          >
            Play Next Game
          </button>
        )}

        <div className={board && board.length > 0 ? "board" : "board board-loading"}>
          {board && board.length > 0 ? (
            board.map((row, rIndex) => (
              <div key={rIndex} className="row">
                {row && row.length > 0 ? (
                  row.map((colorCode, cIndex) => {
                    const isP1 = p1Territory[rIndex] && p1Territory[rIndex][cIndex];
                    const isP2 = p2Territory[rIndex] && p2Territory[rIndex][cIndex];
                    return (
                      <div
                        key={cIndex}
                        className={`cell ${isP1 ? 'player-territory' : ''} ${isP2 ? 'ai-territory' : ''}`}
                        style={{ backgroundColor: COLORS[colorCode] }}
                      />
                    );
                  })
                ) : null}
              </div>
            ))
          ) : (
            <div className="loading-message">Loading board...</div>
          )}
        </div>

        <div className="controls">
          {COLORS.map((color, idx) => (
            <button
              key={idx}
              className={`color-btn ${isColorDisabled(idx) ? 'disabled' : ''}`}
              style={{ backgroundColor: color }}
              onClick={() => handleMove(idx)}
              disabled={isColorDisabled(idx) || gameOver}
            />
          ))}
        </div>

        {aiMoveLog.length > 0 && (
          <div className="ai-log">
            <div
              className="ai-log-header"
              onClick={() => setAiLogExpanded(!aiLogExpanded)}
            >
              <h3>AI Move History ({aiMoveLog.length})</h3>
              <span className="toggle-icon">{aiLogExpanded ? '▼' : '▶'}</span>
            </div>
            {aiLogExpanded && (
              <div className="log-entries">
                {aiMoveLog.map((entry, idx) => {
                  const minQ = getMinQValue(entry.qValues);
                  const maxQ = getMaxQValue(entry.qValues);
                  return (
                    <div key={idx} className="log-entry">
                      <div className="log-header">
                        <span>Turn {entry.turn}: Color {entry.move + 1}</span>
                        <span className="q-value">Q: {entry.qValues[entry.move].toFixed(2)}</span>
                      </div>
                      <div className="q-values-row">
                        {entry.qValues.map((qValue, colorIdx) => {
                          const isChosen = colorIdx === entry.move;
                          const normalized = maxQ !== minQ ? (qValue - minQ) / (maxQ - minQ) : 0.5;
                          return (
                            <div
                              key={colorIdx}
                              className={`q-value-cell ${isChosen ? 'chosen' : ''}`}
                              title={`Color ${colorIdx + 1}: ${qValue.toFixed(2)}`}
                            >
                              <div
                                className="q-bar"
                                style={{
                                  height: `${Math.max(20, normalized * 100)}%`,
                                  backgroundColor: isChosen ? '#4CAF50' : '#555'
                                }}
                              />
                              <div className="q-label">
                                <div className="mini-color" style={{ backgroundColor: COLORS[colorIdx] }} />
                                <span>{qValue.toFixed(1)}</span>
                              </div>
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
        )}
      </div>
    </div>
  );
}

export default App;
