"""SQLite telemetry for Flood-It games.

Writer is an async queue drained by a background worker, so the WebSocket
hot path never blocks on disk IO. Schema mirrors the oware pattern: a row
per game in `games`, one row per ply in `moves`.
"""

import asyncio
import json
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id            TEXT PRIMARY KEY,
    created_at         INTEGER NOT NULL,
    ended_at           INTEGER,
    client_id_hash     TEXT,
    initial_board_json TEXT NOT NULL,
    winner             TEXT,
    final_score_you    INTEGER,
    final_score_ai     INTEGER,
    total_plies        INTEGER,
    schema_version     INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_games_created_at ON games(created_at);
CREATE INDEX IF NOT EXISTS idx_games_client     ON games(client_id_hash);

CREATE TABLE IF NOT EXISTS moves (
    game_id          TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    ply              INTEGER NOT NULL,
    side             TEXT NOT NULL,
    color            INTEGER NOT NULL,
    board_after_json TEXT NOT NULL,
    score_you_after  INTEGER NOT NULL,
    score_ai_after   INTEGER NOT NULL,
    q_values_json    TEXT,
    PRIMARY KEY (game_id, ply)
);

CREATE INDEX IF NOT EXISTS idx_moves_game_id ON moves(game_id);
"""


def _now_ms() -> int:
  return int(time.time() * 1000)


class Telemetry:
  def __init__(self, db_path: Path) -> None:
    self._db_path = db_path
    self._db_path.parent.mkdir(parents=True, exist_ok=True)
    self._conn = sqlite3.connect(
      str(db_path), check_same_thread=False, isolation_level=None
    )
    self._conn.execute("PRAGMA journal_mode=WAL")
    self._conn.execute("PRAGMA synchronous=NORMAL")
    self._conn.executescript(SCHEMA)
    self._queue: asyncio.Queue[tuple[str, tuple] | None] = asyncio.Queue()
    self._worker: asyncio.Task | None = None

  async def start(self) -> None:
    if self._worker is None:
      self._worker = asyncio.create_task(self._run())

  async def stop(self) -> None:
    if self._worker is not None:
      await self._queue.put(None)
      await self._worker
      self._worker = None
    self._conn.close()

  async def _run(self) -> None:
    while True:
      item = await self._queue.get()
      if item is None:
        return
      sql, params = item
      try:
        await asyncio.to_thread(self._conn.execute, sql, params)
      except sqlite3.Error:
        # Drop the row rather than tear down the writer.
        pass

  def record_game_start(
    self,
    *,
    game_id: str,
    client_id_hash: str | None,
    initial_board: list[list[int]],
  ) -> None:
    self._queue.put_nowait(
      (
        """
        INSERT OR IGNORE INTO games
            (game_id, created_at, client_id_hash, initial_board_json)
        VALUES (?, ?, ?, ?)
        """,
        (
          game_id,
          _now_ms(),
          client_id_hash,
          json.dumps(initial_board),
        ),
      )
    )

  def record_move(
    self,
    *,
    game_id: str,
    ply: int,
    side: str,
    color: int,
    board_after: list[list[int]],
    score_you: int,
    score_ai: int,
    q_values: list[float] | None,
  ) -> None:
    self._queue.put_nowait(
      (
        """
        INSERT OR IGNORE INTO moves
            (game_id, ply, side, color, board_after_json,
             score_you_after, score_ai_after, q_values_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
          game_id,
          ply,
          side,
          int(color),
          json.dumps(board_after),
          int(score_you),
          int(score_ai),
          json.dumps(q_values) if q_values is not None else None,
        ),
      )
    )

  def record_game_end(
    self,
    *,
    game_id: str,
    winner: str,
    final_score_you: int,
    final_score_ai: int,
    total_plies: int,
  ) -> None:
    self._queue.put_nowait(
      (
        """
        UPDATE games
           SET ended_at = ?,
               winner = ?,
               final_score_you = ?,
               final_score_ai = ?,
               total_plies = ?
         WHERE game_id = ?
        """,
        (
          _now_ms(),
          winner,
          int(final_score_you),
          int(final_score_ai),
          int(total_plies),
          game_id,
        ),
      )
    )

  # ── Sync read helpers (called from request handlers) ───────────────────
  def read(self) -> sqlite3.Connection:
    """Return a fresh read-only connection; cheap, but caller must close."""
    return sqlite3.connect(str(self._db_path))


@asynccontextmanager
async def telemetry_context(db_path: Path):
  t = Telemetry(db_path)
  await t.start()
  try:
    yield t
  finally:
    await t.stop()


def fetch_games(
  db_path: Path,
  *,
  scope_hash: str | None,
  page: int,
  page_size: int,
) -> dict[str, Any]:
  """List finished games, paginated, newest first."""
  page = max(1, page)
  page_size = max(1, min(100, page_size))
  offset = (page - 1) * page_size

  where_parts = ["ended_at IS NOT NULL"]
  params: list[Any] = []
  if scope_hash is not None:
    where_parts.append("client_id_hash = ?")
    params.append(scope_hash)
  where = " AND ".join(where_parts)

  conn = sqlite3.connect(str(db_path))
  try:
    total = conn.execute(
      f"SELECT COUNT(*) FROM games WHERE {where}", tuple(params)
    ).fetchone()[0]
    rows = conn.execute(
      f"""
      SELECT g.game_id, g.created_at, g.ended_at, g.winner,
             g.final_score_you, g.final_score_ai, g.total_plies,
             (SELECT m.board_after_json FROM moves m
                WHERE m.game_id = g.game_id
                ORDER BY m.ply DESC LIMIT 1)        AS final_board_json,
             g.initial_board_json
      FROM games g
      WHERE {where}
      ORDER BY g.ended_at DESC
      LIMIT ? OFFSET ?
      """,
      tuple(params) + (page_size, offset),
    ).fetchall()
  finally:
    conn.close()

  items = []
  for r in rows:
    final_board = json.loads(r[7]) if r[7] else json.loads(r[8])
    items.append(
      {
        "game_id": r[0],
        "created_at": r[1],
        "ended_at": r[2],
        "winner": r[3],
        "final_scores": [r[4], r[5]],
        "plies": r[6],
        "final_board": final_board,
      }
    )
  return {"total": total, "page": page, "page_size": page_size, "items": items}


def fetch_game(
  db_path: Path,
  *,
  game_id: str,
  scope_hash: str | None,
) -> dict[str, Any] | None:
  """Full game detail with every move. scope_hash=None means \"any\"."""
  conn = sqlite3.connect(str(db_path))
  try:
    if scope_hash is None:
      meta = conn.execute(
        """
        SELECT game_id, created_at, ended_at, winner,
               final_score_you, final_score_ai, total_plies,
               initial_board_json, client_id_hash
        FROM games WHERE game_id = ?
        """,
        (game_id,),
      ).fetchone()
    else:
      meta = conn.execute(
        """
        SELECT game_id, created_at, ended_at, winner,
               final_score_you, final_score_ai, total_plies,
               initial_board_json, client_id_hash
        FROM games WHERE game_id = ? AND client_id_hash = ?
        """,
        (game_id, scope_hash),
      ).fetchone()
    if meta is None:
      return None
    moves = conn.execute(
      """
      SELECT ply, side, color, board_after_json,
             score_you_after, score_ai_after, q_values_json
      FROM moves WHERE game_id = ? ORDER BY ply ASC
      """,
      (game_id,),
    ).fetchall()
  finally:
    conn.close()

  return {
    "game_id": meta[0],
    "created_at": meta[1],
    "ended_at": meta[2],
    "winner": meta[3],
    "final_scores": [meta[4], meta[5]],
    "plies": meta[6],
    "initial_board": json.loads(meta[7]),
    "moves": [
      {
        "ply": m[0],
        "side": m[1],
        "color": m[2],
        "board_after": json.loads(m[3]),
        "score_you_after": m[4],
        "score_ai_after": m[5],
        "q_values": json.loads(m[6]) if m[6] else None,
      }
      for m in moves
    ],
  }


def fetch_stats(db_path: Path, *, scope_hash: str | None) -> dict[str, Any]:
  """Aggregate stats. scope_hash=None means \"all clients\"."""
  where_parts = ["ended_at IS NOT NULL"]
  params: list[Any] = []
  if scope_hash is not None:
    where_parts.append("client_id_hash = ?")
    params.append(scope_hash)
  where = " AND ".join(where_parts)

  conn = sqlite3.connect(str(db_path))
  try:
    totals = conn.execute(
      f"""
      SELECT COUNT(*) AS games,
             SUM(CASE WHEN winner = 'human' THEN 1 ELSE 0 END) AS you,
             SUM(CASE WHEN winner = 'ai'    THEN 1 ELSE 0 END) AS ai,
             SUM(CASE WHEN winner = 'tie'   THEN 1 ELSE 0 END) AS ties,
             AVG(total_plies)                                   AS avg_plies,
             AVG(final_score_you)                               AS avg_you,
             AVG(final_score_ai)                                AS avg_ai,
             COUNT(DISTINCT client_id_hash)                     AS unique_clients
      FROM games WHERE {where}
      """,
      tuple(params),
    ).fetchone()

    recent = conn.execute(
      f"""
      SELECT game_id, winner, total_plies,
             final_score_you, final_score_ai, ended_at
      FROM games WHERE {where}
      ORDER BY ended_at DESC LIMIT 10
      """,
      tuple(params),
    ).fetchall()
  finally:
    conn.close()

  games = totals[0] or 0
  return {
    "totals": {
      "games": games,
      "you_wins": totals[1] or 0,
      "ai_wins": totals[2] or 0,
      "ties": totals[3] or 0,
      "avg_plies": round(totals[4], 2) if totals[4] is not None else 0,
      "avg_you": round(totals[5], 2) if totals[5] is not None else 0,
      "avg_ai": round(totals[6], 2) if totals[6] is not None else 0,
      "unique_clients": totals[7] or 0,
    },
    "recent": [
      {
        "game_id": r[0],
        "winner": r[1],
        "plies": r[2],
        "final_scores": [r[3], r[4]],
        "ended_at": r[5],
      }
      for r in recent
    ],
  }
