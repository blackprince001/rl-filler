import os
import sys

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from stable_baselines3 import DQN

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
  sys.path.insert(0, project_root)

from backend.core_game import FloodItGame

app = FastAPI()


# Load Model once on startup
model_path = os.path.join(os.path.dirname(__file__), "models", "floodit_dqn")
if os.path.exists(model_path + ".zip"):
  model = DQN.load(model_path)
  print(f"Model loaded from {model_path}")
else:
  print(f"Warning: Model not found at {model_path}. Please train the model first.")
  model = None


def preprocess_board(board):
  # Convert pure 2D array to One-Hot 3D array for the AI
  obs = np.zeros((8, 7, 6), dtype=np.uint8)
  for r in range(8):
    for c in range(7):
      color = board[r][c]
      obs[r, c, color] = 1
  return obs


def flip_board_for_p2(board):
  """
  Flip the board horizontally and vertically so the model trained as P1
  can play correctly as P2. This transforms the bottom-right corner to top-left.
  """
  # Flip both axes: flip vertically then horizontally
  return np.flip(np.flip(board, axis=0), axis=1)


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


@app.websocket("/ws/game")
async def game_endpoint(websocket: WebSocket):
  await manager.connect(websocket)
  game = FloodItGame()  # New instance per connection

  try:
    # Send initial board
    p1_score, p2_score = game.get_score()
    p1_mask, p2_mask = game.get_territory_masks()
    await websocket.send_json(
      {
        "type": "INIT",
        "board": game.board.tolist(),
        "scores": [int(p1_score), int(p2_score)],
        "p1_territory": p1_mask.tolist(),
        "p2_territory": p2_mask.tolist(),
        "last_p1_move": game.last_p1_move,
        "last_p2_move": game.last_p2_move,
      }
    )

    while True:
      data = await websocket.receive_json()

      if data["type"] == "MOVE":
        user_color = data["color"]

        # 1. Apply User Move (Player 1)
        game.play_move(user_color, is_player_1=True)

        # Check Win
        if game.is_game_over():
          p1_score, p2_score = game.get_score()
          p1_mask, p2_mask = game.get_territory_masks()
          await websocket.send_json(
            {
              "type": "GAME_OVER",
              "board": game.board.tolist(),
              "scores": [int(p1_score), int(p2_score)],
              "p1_territory": p1_mask.tolist(),
              "p2_territory": p2_mask.tolist(),
            }
          )
          continue

        # 2. AI Turn (Player 2 - AI Logic)
        # The model was trained as P1 (top-left), but here it plays as P2 (bottom-right).
        # We flip the board perspective so the model sees it from its training perspective.
        ai_decision_info = None
        if model is not None:
          # Flip board for P2 perspective
          flipped_board = flip_board_for_p2(game.board)
          obs = preprocess_board(flipped_board)

          # Get Q-values for all actions to show decision-making
          obs_tensor = model.policy.obs_to_tensor(obs)[0]
          model.policy.q_net.eval()
          with torch.no_grad():
            q_values = model.policy.q_net(obs_tensor).cpu().numpy()[0]

          # Filter out invalid actions (opponent's last move, current color)
          ai_start_color = game.board[game.height - 1, game.width - 1]
          valid_actions = []
          valid_q_values = []

          for action in range(6):
            # Skip if same as current color
            if action == ai_start_color:
              continue
            # Skip if same as player's last move
            if game.last_p1_move is not None and action == game.last_p1_move:
              continue
            valid_actions.append(action)
            valid_q_values.append(float(q_values[action]))

          if valid_actions:
            # Choose action with highest Q-value
            best_idx = np.argmax(valid_q_values)
            ai_color = valid_actions[best_idx]

            # Create decision info for frontend
            all_q_values = [float(q_values[i]) for i in range(6)]
            ai_decision_info = {
              "q_values": all_q_values,
              "chosen_action": int(ai_color),
              "valid_actions": valid_actions,
              "valid_q_values": valid_q_values,
            }
          else:
            # Fallback if no valid actions (shouldn't happen)
            ai_color = (ai_start_color + 1) % 6
            all_q_values = [float(q_values[i]) for i in range(6)]
            ai_decision_info = {
              "q_values": all_q_values,
              "chosen_action": int(ai_color),
              "valid_actions": [],
              "valid_q_values": [],
            }
        else:
          # Fallback to random if model not loaded
          ai_start_color = game.board[game.height - 1, game.width - 1]
          valid_colors = [
            c
            for c in range(6)
            if c != ai_start_color
            and (game.last_p1_move is None or c != game.last_p1_move)
          ]
          if valid_colors:
            ai_color = np.random.choice(valid_colors)
          else:
            ai_color = (ai_start_color + 1) % 6

        move_valid = game.play_move(ai_color, is_player_1=False)
        if not move_valid:
          # Should not happen, but fallback
          ai_start_color = game.board[game.height - 1, game.width - 1]
          for c in range(6):
            if c != ai_start_color and (
              game.last_p1_move is None or c != game.last_p1_move
            ):
              ai_color = c
              game.play_move(ai_color, is_player_1=False)
              break

        # 3. Check if game is over after AI move
        if game.is_game_over():
          p1_score, p2_score = game.get_score()
          p1_mask, p2_mask = game.get_territory_masks()
          await websocket.send_json(
            {
              "type": "GAME_OVER",
              "board": game.board.tolist(),
              "scores": [int(p1_score), int(p2_score)],
              "p1_territory": p1_mask.tolist(),
              "p2_territory": p2_mask.tolist(),
            }
          )
          continue

        # 4. Send update back if game continues
        p1_score, p2_score = game.get_score()
        p1_mask, p2_mask = game.get_territory_masks()
        await websocket.send_json(
          {
            "type": "UPDATE",
            "board": game.board.tolist(),
            "scores": [int(p1_score), int(p2_score)],
            "p1_territory": p1_mask.tolist(),
            "p2_territory": p2_mask.tolist(),
            "last_p1_move": game.last_p1_move,
            "last_p2_move": game.last_p2_move,
            "last_ai_move": int(ai_color),
            "ai_decision": ai_decision_info,
          }
        )

  except WebSocketDisconnect:
    manager.disconnect(websocket)
