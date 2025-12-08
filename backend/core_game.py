import numpy as np


class FloodItGame:
  def __init__(self, width=7, height=8, n_colors=6):
    self.width = width
    self.height = height
    self.n_colors = n_colors
    self.board = np.zeros((height, width), dtype=int)
    self.last_p1_move = None
    self.last_p2_move = None
    self.reset()

  def reset(self):
    # Generate random board
    self.board = np.random.randint(0, self.n_colors, (self.height, self.width))
    # Ensure corners are different
    while self.board[0, 0] == self.board[-1, -1]:
      self.board[-1, -1] = (self.board[-1, -1] + 1) % self.n_colors
    self.last_p1_move = None
    self.last_p2_move = None
    return self.board

  def get_owner_mask(self, start_x, start_y):
    """Returns a boolean mask of tiles owned by the player at start_x, start_y"""
    owner_color = self.board[start_y, start_x]
    mask = np.zeros_like(self.board, dtype=bool)
    stack = [(start_x, start_y)]
    mask[start_y, start_x] = True

    while stack:
      x, y = stack.pop()
      neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
      for nx, ny in neighbors:
        if 0 <= nx < self.width and 0 <= ny < self.height:
          if not mask[ny, nx] and self.board[ny, nx] == owner_color:
            mask[ny, nx] = True
            stack.append((nx, ny))
    return mask

  def play_move(self, color, is_player_1=True):
    """
    Updates board.
    Player 1 starts Top-Left (0,0).
    Player 2 starts Bottom-Right.
    Cannot pick the same color the opponent just picked.
    """
    start_x, start_y = (0, 0) if is_player_1 else (self.width - 1, self.height - 1)

    current_color = self.board[start_y, start_x]
    if color == current_color:
      return False  # Invalid move (no change)

    # Check if trying to pick opponent's last move
    opponent_last_move = self.last_p2_move if is_player_1 else self.last_p1_move
    if opponent_last_move is not None and color == opponent_last_move:
      return False  # Cannot pick opponent's last color

    mask = self.get_owner_mask(start_x, start_y)
    self.board[mask] = color  # Flood fill

    # Update last move
    if is_player_1:
      self.last_p1_move = color
    else:
      self.last_p2_move = color

    return True

  def get_score(self):
    p1_mask = self.get_owner_mask(0, 0)
    p2_mask = self.get_owner_mask(self.width - 1, self.height - 1)
    return np.sum(p1_mask), np.sum(p2_mask)

  def is_game_over(self):
    p1, p2 = self.get_score()
    return (p1 + p2) >= (self.width * self.height)

  def get_territory_masks(self):
    """Returns boolean masks for both players' territories"""
    p1_mask = self.get_owner_mask(0, 0)
    p2_mask = self.get_owner_mask(self.width - 1, self.height - 1)
    return p1_mask, p2_mask
