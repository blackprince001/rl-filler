import hashlib
import os
import secrets
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Cookie, FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
  sys.path.insert(0, project_root)

from backend.core_game import FloodItGame
from backend.onnx_runner import QValueRunner
from backend.telemetry import Telemetry, fetch_game, fetch_games, fetch_stats

CLIENT_COOKIE = "rlfiller_client"
CLIENT_SALT = os.environ.get("RL_FILLER_CLIENT_SALT", "dev-salt-not-for-prod")
DB_PATH = Path(os.environ.get("RL_FILLER_DB", "backend/data/telemetry.db"))

# Cross-site cookies (Vercel frontend ↔ Railway API) need SameSite=None + Secure.
COOKIE_SAMESITE = os.environ.get("RL_FILLER_COOKIE_SAMESITE", "none").lower()
COOKIE_SECURE = os.environ.get("RL_FILLER_COOKIE_SECURE", "1") == "1"
COOKIE_MAX_AGE = 60 * 60 * 24 * 365  # 1 year

_RAW_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _RAW_ORIGINS.split(",") if o.strip()] or ["*"]


def _hash_client(cookie: str | None) -> str | None:
  if cookie is None:
    return None
  return hashlib.sha256(f"{cookie}{CLIENT_SALT}".encode()).hexdigest()[:16]


def _new_client_token() -> str:
  return secrets.token_urlsafe(16)


def _new_game_id() -> str:
  return f"g_{secrets.token_urlsafe(9)}"


@asynccontextmanager
async def lifespan(app: FastAPI):
  app.state.telemetry = Telemetry(DB_PATH)
  await app.state.telemetry.start()
  try:
    yield
  finally:
    await app.state.telemetry.stop()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
  CORSMiddleware,
  allow_origins=ALLOWED_ORIGINS,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


# ── Model load ────────────────────────────────────────────────────────────
model_path = Path(__file__).resolve().parent / "models" / "floodit_dqn.onnx"
if model_path.exists():
  runner: QValueRunner | None = QValueRunner(model_path)
  print(f"ONNX model loaded from {model_path}")
else:
  print(f"Warning: ONNX model not found at {model_path}. Random fallback only.")
  runner = None


# ── HTTP endpoints ────────────────────────────────────────────────────────
@app.get("/healthz")
async def healthz():
  return {"ok": True, "model_loaded": runner is not None}


def _set_client_cookie(response: Response) -> str:
  token = _new_client_token()
  response.set_cookie(
    CLIENT_COOKIE,
    token,
    max_age=COOKIE_MAX_AGE,
    samesite=COOKIE_SAMESITE,  # type: ignore[arg-type]
    secure=COOKIE_SECURE,
    httponly=False,
    path="/",
  )
  return token


@app.post("/session")
async def session(
  response: Response,
  rlfiller_client: str | None = Cookie(default=None),
):
  """Ensure the browser carries a client cookie before opening the WS."""
  if rlfiller_client is None:
    _set_client_cookie(response)
    return {"created": True}
  return {"created": False}


@app.get("/games")
async def games_list(
  scope: str = "mine",
  page: int = 1,
  page_size: int = 24,
  rlfiller_client: str | None = Cookie(default=None),
):
  if scope == "mine":
    h = _hash_client(rlfiller_client)
    if h is None:
      return {"total": 0, "page": page, "page_size": page_size, "items": []}
    return fetch_games(DB_PATH, scope_hash=h, page=page, page_size=page_size)
  return fetch_games(DB_PATH, scope_hash=None, page=page, page_size=page_size)


@app.get("/games/{game_id}")
async def game_detail(
  game_id: str,
  response: Response,
  scope: str = "mine",
  rlfiller_client: str | None = Cookie(default=None),
):
  if scope == "all":
    g = fetch_game(DB_PATH, game_id=game_id, scope_hash=None)
  else:
    h = _hash_client(rlfiller_client)
    if h is None:
      response.status_code = 404
      return {"error": "not_found"}
    g = fetch_game(DB_PATH, game_id=game_id, scope_hash=h)
  if g is None:
    response.status_code = 404
    return {"error": "not_found"}
  return g


@app.get("/stats")
async def stats(
  scope: str = "mine",
  rlfiller_client: str | None = Cookie(default=None),
):
  if scope == "mine":
    h = _hash_client(rlfiller_client)
    if h is None:
      return {"totals": {"games": 0}, "recent": []}
    return fetch_stats(DB_PATH, scope_hash=h)
  return fetch_stats(DB_PATH, scope_hash=None)


# ── Game session helpers ──────────────────────────────────────────────────
def build_state(game: FloodItGame, msg_type: str, game_id: str, ai_decision_info=None):
  ai_score, you_score = game.get_score()
  ai_mask, you_mask = game.get_territory_masks()
  payload = {
    "type": msg_type,
    "game_id": game_id,
    "board": game.board.tolist(),
    "scores": [int(you_score), int(ai_score)],
    "you_territory": you_mask.tolist(),
    "ai_territory": ai_mask.tolist(),
    "last_you_move": game.last_p2_move,
    "last_ai_move": game.last_p1_move,
  }
  if ai_decision_info is not None:
    payload["ai_decision"] = ai_decision_info
  return payload


def pick_ai_move(game: FloodItGame):
  ai_start_color = int(game.board[0, 0])
  human_last = game.last_p2_move

  def is_valid(action):
    if action == ai_start_color:
      return False
    if human_last is not None and action == human_last:
      return False
    return True

  if runner is not None:
    q_values = runner.q_values(game.board)
    valid_actions = [a for a in range(6) if is_valid(a)]
    valid_q = [float(q_values[a]) for a in valid_actions]
    all_q = [float(q_values[i]) for i in range(6)]

    if valid_actions:
      best = int(np.argmax(valid_q))
      ai_color = valid_actions[best]
    else:
      ai_color = (ai_start_color + 1) % 6

    return ai_color, {
      "q_values": all_q,
      "chosen_action": int(ai_color),
      "valid_actions": valid_actions,
      "valid_q_values": valid_q,
    }

  valid = [c for c in range(6) if is_valid(c)]
  ai_color = int(np.random.choice(valid)) if valid else (ai_start_color + 1) % 6
  return ai_color, None


@app.websocket("/ws/game")
async def game_endpoint(websocket: WebSocket):
  await websocket.accept()
  telemetry: Telemetry = app.state.telemetry
  client_hash = _hash_client(websocket.cookies.get(CLIENT_COOKIE))

  game = FloodItGame()
  game_id = _new_game_id()
  ply = 0
  ended = False

  def start_new_game(initial_board: list[list[int]]):
    nonlocal game_id, ply, ended
    game_id = _new_game_id()
    ply = 0
    ended = False
    telemetry.record_game_start(
      game_id=game_id, client_id_hash=client_hash, initial_board=initial_board
    )

  def log_move(side: str, color: int, q_values: list[float] | None):
    nonlocal ply
    ply += 1
    ai_score, you_score = game.get_score()
    telemetry.record_move(
      game_id=game_id,
      ply=ply,
      side=side,
      color=color,
      board_after=game.board.tolist(),
      score_you=int(you_score),
      score_ai=int(ai_score),
      q_values=q_values,
    )

  def finalize(winner_scores):
    nonlocal ended
    if ended:
      return
    ended = True
    you, ai = int(winner_scores[0]), int(winner_scores[1])
    winner = "human" if you > ai else "ai" if ai > you else "tie"
    telemetry.record_game_end(
      game_id=game_id,
      winner=winner,
      final_score_you=you,
      final_score_ai=ai,
      total_plies=ply,
    )

  try:
    start_new_game(game.board.tolist())
    await websocket.send_json(build_state(game, "INIT", game_id))

    while True:
      data = await websocket.receive_json()

      if data.get("type") == "RESET":
        # If a game was in progress and not finalized, leave its row open;
        # the client explicitly abandoned it. Start fresh.
        game.reset()
        start_new_game(game.board.tolist())
        await websocket.send_json(build_state(game, "INIT", game_id))
        continue

      if data.get("type") != "MOVE":
        continue

      user_color = int(data["color"])
      if not game.play_move(user_color, is_player_1=False):
        continue
      log_move("you", user_color, None)

      if game.is_game_over():
        finalize(game.get_score()[::-1])  # get_score = (ai, you); we want (you, ai)
        await websocket.send_json(build_state(game, "GAME_OVER", game_id))
        continue

      ai_color, decision = pick_ai_move(game)
      move_valid = game.play_move(ai_color, is_player_1=True)
      if not move_valid:
        ai_start = int(game.board[0, 0])
        for c in range(6):
          if c != ai_start and (game.last_p2_move is None or c != game.last_p2_move):
            game.play_move(c, is_player_1=True)
            ai_color = c
            break
        if decision is not None:
          decision["chosen_action"] = int(ai_color)
      log_move("ai", ai_color, decision["q_values"] if decision else None)

      if game.is_game_over():
        finalize(game.get_score()[::-1])
        await websocket.send_json(build_state(game, "GAME_OVER", game_id, ai_decision_info=decision))
        continue

      await websocket.send_json(build_state(game, "UPDATE", game_id, ai_decision_info=decision))

  except WebSocketDisconnect:
    return
