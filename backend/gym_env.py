import copy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from backend.core_game import FloodItGame


class FloodItEnv(gym.Env):
  def __init__(self, opponent_type="greedy"):
    """
    Args:
      opponent_type: "greedy" or "random" - type of opponent to use
    """
    super(FloodItEnv, self).__init__()
    self.game = FloodItGame()
    self.opponent_type = opponent_type  # "greedy" or "random"

    # Action Space: 6 Colors
    self.action_space = spaces.Discrete(6)

    # Observation Space: 8x7 grid with 6 channels (One-Hot Encoded)
    # Shape: (Height, Width, Channels) -> (8, 7, 6)
    self.observation_space = spaces.Box(low=0, high=1, shape=(8, 7, 6), dtype=np.uint8)

    # For training, AI is Player 1 (Top Left)
    self.current_player_ai = True

  def get_obs(self):
    obs = np.zeros((8, 7, 6), dtype=np.uint8)
    for r in range(8):
      for c in range(7):
        color = self.game.board[r, c]
        obs[r, c, color] = 1
    return obs

  def get_valid_actions(self, is_player_1=True):
    """Returns a list of valid actions for the specified player"""
    if is_player_1:
      current_color = self.game.board[0, 0]
      opponent_last_move = self.game.last_p2_move

    else:
      current_color = self.game.board[self.game.height - 1, self.game.width - 1]
      opponent_last_move = self.game.last_p1_move

    valid = []
    for action in range(6):
      if action == current_color:
        continue  # Cannot pick own color
      if opponent_last_move is not None and action == opponent_last_move:
        continue  # Cannot pick opponent's last move
      valid.append(action)
    return valid if valid else [c for c in range(6) if c != current_color]  # Fallback

  def get_greedy_opponent_move(self):
    """Returns the best immediate move for the opponent (greedy strategy)"""
    valid_moves = self.get_valid_actions(is_player_1=False)
    if not valid_moves:
      return None

    best_move = None
    best_score_gain = -1

    # Save current state
    prev_p2_score = self.game.get_score()[1]

    # Try each valid move and see which gives most tiles
    for move in valid_moves:
      # Create a temporary copy to test the move
      temp_game = copy.deepcopy(self.game)
      if temp_game.play_move(move, is_player_1=False):
        new_p2_score = temp_game.get_score()[1]
        score_gain = new_p2_score - prev_p2_score
        if score_gain > best_score_gain:
          best_score_gain = score_gain
          best_move = move

    return best_move if best_move is not None else np.random.choice(valid_moves)

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.game.reset()
    return self.get_obs(), {}

  def step(self, action):
    valid_actions = self.get_valid_actions(is_player_1=True)
    if action not in valid_actions:
      if len(valid_actions) > 0:
        best_action = None
        best_gain = -1
        prev_score = self.game.get_score()[0]

        for test_action in valid_actions:
          temp_game = copy.deepcopy(self.game)
          if temp_game.play_move(test_action, is_player_1=True):
            new_score = temp_game.get_score()[0]
            gain = new_score - prev_score
            if gain > best_gain:
              best_gain = gain
              best_action = test_action

        action = (
          best_action if best_action is not None else np.random.choice(valid_actions)
        )
      else:
        action = (self.game.board[0, 0] + 1) % 6

    prev_p1_score, prev_p2_score = self.game.get_score()
    prev_score_diff = prev_p1_score - prev_p2_score

    valid = self.game.play_move(action, is_player_1=True)
    new_p1_score, new_p2_score = self.game.get_score()
    new_score_diff = new_p1_score - new_p2_score

    reward = 0
    terminated = False
    truncated = False

    if not valid:
      reward = -1
    else:
      tile_gain = new_p1_score - prev_p1_score
      reward += tile_gain

      score_diff_change = new_score_diff - prev_score_diff
      reward += 0.3 * score_diff_change

      if new_score_diff > 0:
        reward += 0.2
      elif new_score_diff < 0:
        reward -= 0.1

      opp_tile_gain = new_p2_score - prev_p2_score
      if opp_tile_gain == 0:
        reward += 0.1

    valid_opp_moves = self.get_valid_actions(is_player_1=False)

    if valid_opp_moves:
      if self.opponent_type == "greedy":
        opp_move = self.get_greedy_opponent_move()
        if opp_move is None:
          opp_move = np.random.choice(valid_opp_moves)
      else:
        opp_move = np.random.choice(valid_opp_moves)

      self.game.play_move(opp_move, is_player_1=False)

      final_p1_score, final_p2_score = self.game.get_score()

      # Adjust reward based on opponent's move impact
      opp_tile_gain = final_p2_score - new_p2_score
      if opp_tile_gain > tile_gain:
        reward -= 0.2  # Penalty if opponent gained more than we did

    if self.game.is_game_over():
      terminated = True
      p1, p2 = self.game.get_score()
      final_diff = p1 - p2

      # Final reward based on outcome
      if p1 > p2:
        reward += 50 + final_diff * 0.5
      elif p1 < p2:
        reward -= 30 - final_diff * 0.3
      else:
        reward += 10

    return self.get_obs(), reward, terminated, truncated, {}
