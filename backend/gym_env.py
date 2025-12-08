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
    prev_p1_score, prev_p2_score = self.game.get_score()

    # Try each valid move and see which gives most tiles
    # Use test_move_score instead of deepcopy for better performance
    for move in valid_moves:
      result = self.game.test_move_score(move, is_player_1=False)
      if result is not None:
        new_p1_score, new_p2_score = result
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
          result = self.game.test_move_score(test_action, is_player_1=True)
          if result is not None:
            new_score, _ = result
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

    # Check if this move would end the game BEFORE playing it (for reward shaping)
    would_end_game = False
    would_win = False
    move_result = self.game.test_move_score(action, is_player_1=True)
    if move_result is not None:
      future_p1, future_p2 = move_result
      total_tiles = self.game.width * self.game.height
      would_end_game = (future_p1 + future_p2) >= total_tiles
      would_win = future_p1 > future_p2

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

      # Bonus reward for moves that would end the game favorably
      if would_end_game:
        if would_win:
          reward += 20  # Bonus for winning move
        else:
          reward -= 15  # Penalty for losing move

    # Only let opponent move if game is not over after our move
    final_p1_score, final_p2_score = new_p1_score, new_p2_score

    if not self.game.is_game_over():
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

    # Check for forced end state (early truncation to save training time)
    if not terminated and self.game.is_forced_end_state(is_player_1=True):
      truncated = True
      # Give reward based on current state since game is effectively over
      p1, p2 = self.game.get_score()
      final_diff = p1 - p2
      if p1 > p2:
        reward += 30 + final_diff * 0.3
      elif p1 < p2:
        reward -= 20 - final_diff * 0.2
      else:
        reward += 5

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

    # Add information about remaining tiles for better learning
    unowned_mask = self.game.get_unowned_tiles()
    remaining_tiles = np.sum(unowned_mask)
    total_tiles = self.game.width * self.game.height
    remaining_ratio = remaining_tiles / total_tiles if total_tiles > 0 else 0.0

    info = {
      "remaining_tiles": int(remaining_tiles),
      "remaining_ratio": float(remaining_ratio),
      "p1_score": int(final_p1_score),
      "p2_score": int(final_p2_score),
      "score_diff": int(final_p1_score - final_p2_score),
    }

    return self.get_obs(), reward, terminated, truncated, info

  def render(self):
    """Render game state to console"""
    p1_score, p2_score = self.game.get_score()
    p1_mask, p2_mask = self.game.get_territory_masks()

    # Color symbols for console
    color_symbols = ["⬛", "🟪", "🟥", "🟩", "🟦", "🟨"]

    for r in range(self.game.height):
      row_str = ""
      for c in range(self.game.width):
        color = self.game.board[r, c]
        symbol = color_symbols[color]

        # Mark territories
        if p1_mask[r, c]:
          symbol = f"[{symbol}]"
        elif p2_mask[r, c]:
          symbol = f"({symbol})"

        row_str += symbol + " "
      print(row_str)

    print("=" * 50)

    if self.game.is_game_over():
      winner = (
        "AI (P1)"
        if p1_score > p2_score
        else "Opponent (P2)"
        if p2_score > p1_score
        else "Tie"
      )
      print(f"Game Over! Winner: {winner}")
      print()
