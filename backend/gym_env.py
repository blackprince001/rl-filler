import gymnasium as gym
import numpy as np
from gymnasium import spaces

from backend.core_game import FloodItGame


class FloodItEnv(gym.Env):
  def __init__(self):
    super(FloodItEnv, self).__init__()
    self.game = FloodItGame()

    # Action Space: 6 Colors
    self.action_space = spaces.Discrete(6)

    # Observation Space: 8x7 grid with 6 channels (One-Hot Encoded)
    # Shape: (Height, Width, Channels) -> (8, 7, 6)
    self.observation_space = spaces.Box(low=0, high=1, shape=(8, 7, 6), dtype=np.uint8)

    # For training, AI is Player 1 (Top Left)
    self.current_player_ai = True

  def get_obs(self):
    # Convert 2D (8,7) board to 3D (8,7,6) one-hot
    obs = np.zeros((8, 7, 6), dtype=np.uint8)
    for r in range(8):
      for c in range(7):
        color = self.game.board[r, c]
        obs[r, c, color] = 1
    return obs

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.game.reset()
    return self.get_obs(), {}

  def step(self, action):
    # 1. AI plays
    prev_p1_score, _ = self.game.get_score()
    valid = self.game.play_move(action, is_player_1=True)
    new_p1_score, _ = self.game.get_score()

    reward = 0
    terminated = False
    truncated = False

    if not valid:
      # Punish invalid moves (choosing own color) heavy
      reward = -10
    else:
      # Reward is how many tiles it gained
      reward = new_p1_score - prev_p1_score

      # 2. Opponent plays (Random or Simple Greedy for training)
      # To train a robust AI, we simulate a random opponent
      opp_move = np.random.randint(0, 6)
      self.game.play_move(opp_move, is_player_1=False)

    if self.game.is_game_over():
      terminated = True
      # Final bonus for winning
      p1, p2 = self.game.get_score()
      if p1 > p2:
        reward += 50
      else:
        reward -= 50

    return self.get_obs(), reward, terminated, truncated, {}
