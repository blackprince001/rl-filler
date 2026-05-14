import os
import sys

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from stable_baselines3 import DQN

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
  sys.path.insert(0, project_root)

from backend.core_game import FloodItGame

app = FastAPI()

allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
  CORSMiddleware,
  allow_origins=allowed_origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


@app.get("/healthz")
async def healthz():
  return {"ok": True, "model_loaded": model is not None}


device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
  print("Warning: CUDA not available, using CPU for inference. Inference will be slower.")
else:
  print(f"Using device: {device} ({torch.cuda.get_device_name(0)}) for inference")


model_path = os.path.join(os.path.dirname(__file__), "models", "floodit_dqn")
if os.path.exists(model_path + ".zip"):
  model = DQN.load(model_path, device=device)
  if hasattr(model.policy, "q_net"):
    model.policy.q_net = model.policy.q_net.to(device)
  print(f"Model loaded from {model_path} on {device}")
else:
  print(f"Warning: Model not found at {model_path}. Please train the model first.")
  model = None


def preprocess_board(board):
  obs = np.zeros((8, 7, 6), dtype=np.uint8)
  for r in range(8):
    for c in range(7):
      color = board[r][c]
      obs[r, c, color] = 1
  return obs


class ConnectionManager:
  def __init__(self):
    self.active_connections: list[WebSocket] = []

  async def connect(self, websocket: WebSocket):
    await websocket.accept()
    self.active_connections.append(websocket)

  def disconnect(self, websocket: WebSocket):
    if websocket in self.active_connections:
      self.active_connections.remove(websocket)


manager = ConnectionManager()


def build_state(game: FloodItGame, msg_type: str, ai_decision_info=None):
  """Game state with human-centric naming.

  In the game engine, P1 is top-left and P2 is bottom-right. We assign:
    - P1 (top-left)     -> AI
    - P2 (bottom-right) -> human player
  So the human always plays from the bottom corner.
  """
  ai_score, you_score = game.get_score()
  ai_mask, you_mask = game.get_territory_masks()
  payload = {
    "type": msg_type,
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
  """Pick AI's move. AI plays as P1 (top-left), matching the trained model."""
  ai_start_color = int(game.board[0, 0])
  human_last = game.last_p2_move

  def is_valid(action):
    if action == ai_start_color:
      return False
    if human_last is not None and action == human_last:
      return False
    return True

  if model is not None:
    obs = preprocess_board(game.board)
    obs_tensor = model.policy.obs_to_tensor(obs)[0].to(device)
    model.policy.q_net.eval()
    with torch.no_grad():
      q_values = model.policy.q_net(obs_tensor).cpu().numpy()[0]

    valid_actions = [a for a in range(6) if is_valid(a)]
    valid_q = [float(q_values[a]) for a in valid_actions]
    all_q = [float(q_values[i]) for i in range(6)]

    if valid_actions:
      best = int(np.argmax(valid_q))
      ai_color = valid_actions[best]
    else:
      ai_color = (ai_start_color + 1) % 6

    decision = {
      "q_values": all_q,
      "chosen_action": int(ai_color),
      "valid_actions": valid_actions,
      "valid_q_values": valid_q,
    }
    return ai_color, decision

  # Fallback random
  valid = [c for c in range(6) if is_valid(c)]
  ai_color = int(np.random.choice(valid)) if valid else (ai_start_color + 1) % 6
  return ai_color, None


@app.websocket("/ws/game")
async def game_endpoint(websocket: WebSocket):
  await manager.connect(websocket)
  game = FloodItGame()

  try:
    await websocket.send_json(build_state(game, "INIT"))

    while True:
      data = await websocket.receive_json()

      if data["type"] == "RESET":
        game.reset()
        await websocket.send_json(build_state(game, "INIT"))
        continue

      if data["type"] == "MOVE":
        user_color = data["color"]
        # Human plays as P2 (bottom-right).
        game.play_move(user_color, is_player_1=False)

        if game.is_game_over():
          await websocket.send_json(build_state(game, "GAME_OVER"))
          continue

        # AI plays as P1 (top-left), no perspective flip needed.
        ai_color, decision = pick_ai_move(game)
        move_valid = game.play_move(ai_color, is_player_1=True)
        if not move_valid:
          # Fallback: try any valid color
          ai_start_color = int(game.board[0, 0])
          for c in range(6):
            if c != ai_start_color and (game.last_p2_move is None or c != game.last_p2_move):
              game.play_move(c, is_player_1=True)
              ai_color = c
              break
          if decision is not None:
            decision["chosen_action"] = int(ai_color)

        if game.is_game_over():
          await websocket.send_json(build_state(game, "GAME_OVER", ai_decision_info=decision))
          continue

        await websocket.send_json(build_state(game, "UPDATE", ai_decision_info=decision))

  except WebSocketDisconnect:
    manager.disconnect(websocket)
