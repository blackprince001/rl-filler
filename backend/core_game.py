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
    self.board = self._generate_scattered_board()

    while self.board[0, 0] == self.board[-1, -1]:
      self.board[-1, -1] = (self.board[-1, -1] + 1) % self.n_colors
    self.last_p1_move = None
    self.last_p2_move = None

    return self.board

  def _generate_scattered_board(self):
    """
    Generates a board with small, scattered clusters to limit tile gains per move.
    Maximum gain per move should be around 3-4 tiles for strategic gameplay.
    """
    board = np.zeros((self.height, self.width), dtype=int)

    # Start with random distribution
    board = np.random.randint(0, self.n_colors, (self.height, self.width))

    for _ in range(1):
      new_board = board.copy()
      for y in range(self.height):
        for x in range(self.width):
          neighbor_colors = []
          for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.height and 0 <= nx < self.width:
              neighbor_colors.append(board[ny, nx])

          if neighbor_colors:
            color_counts = np.bincount(neighbor_colors, minlength=self.n_colors)
            most_common = np.argmax(color_counts)

            # LOW chance (30%) to match - keeps clusters small
            if np.random.random() < 0.3:
              new_board[y, x] = most_common
      board = new_board

    # Break up any large regions that might have formed
    visited = np.zeros((self.height, self.width), dtype=bool)
    max_cluster_size = 3  # Maximum 3 tiles per cluster (2-3 tiles per move)

    for y in range(self.height):
      for x in range(self.width):
        if not visited[y, x]:
          # Flood fill to find cluster size
          region_color = board[y, x]
          region_size = 0
          stack = [(x, y)]
          region_cells = []

          while stack:
            cx, cy = stack.pop()
            if visited[cy, cx] or board[cy, cx] != region_color:
              continue
            visited[cy, cx] = True
            region_size += 1
            region_cells.append((cx, cy))

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
              nx, ny = cx + dx, cy + dy
              if 0 <= nx < self.width and 0 <= ny < self.height:
                if not visited[ny, nx] and board[ny, nx] == region_color:
                  stack.append((nx, ny))

          # If cluster is too large, break it up
          if region_size > max_cluster_size:
            # Change excess cells to different colors
            cells_to_change = region_size - max_cluster_size
            np.random.shuffle(region_cells)
            for cx, cy in region_cells[:cells_to_change]:
              # Change to a different color
              new_color = (
                region_color + np.random.randint(1, self.n_colors)
              ) % self.n_colors
              board[cy, cx] = new_color

    # Final pass: ensure colors are well distributed and scattered
    # Add some randomness to break up any remaining patterns
    for _ in range(2):
      for y in range(self.height):
        for x in range(self.width):
          # 10% chance to randomize a cell to add variety
          if np.random.random() < 0.1:
            board[y, x] = np.random.randint(0, self.n_colors)

    return board

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

  def test_move_score(self, color, is_player_1=True):
    """
    Test a move and return the score gain without modifying the game state.
    This is more efficient than deep copying the entire game.
    Returns: (new_p1_score, new_p2_score) or None if move is invalid
    """
    start_x, start_y = (0, 0) if is_player_1 else (self.width - 1, self.height - 1)
    current_color = self.board[start_y, start_x]

    if color == current_color:
      return None  # Invalid move

    # Check if trying to pick opponent's last move
    opponent_last_move = self.last_p2_move if is_player_1 else self.last_p1_move
    if opponent_last_move is not None and color == opponent_last_move:
      return None  # Invalid move

    # Create a temporary board copy (much cheaper than deep copying entire game)
    temp_board = self.board.copy()

    # Apply the move on the temp board
    mask = self.get_owner_mask(start_x, start_y)
    temp_board[mask] = color

    # Calculate new scores
    # We need to recalculate masks on the temp board
    temp_p1_mask = np.zeros_like(temp_board, dtype=bool)
    temp_p2_mask = np.zeros_like(temp_board, dtype=bool)

    # P1 territory
    stack = [(0, 0)]
    p1_color = temp_board[0, 0]
    temp_p1_mask[0, 0] = True
    while stack:
      x, y = stack.pop()
      for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < self.width and 0 <= ny < self.height:
          if not temp_p1_mask[ny, nx] and temp_board[ny, nx] == p1_color:
            temp_p1_mask[ny, nx] = True
            stack.append((nx, ny))

    # P2 territory
    stack = [(self.width - 1, self.height - 1)]
    p2_color = temp_board[self.height - 1, self.width - 1]
    temp_p2_mask[self.height - 1, self.width - 1] = True
    while stack:
      x, y = stack.pop()
      for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < self.width and 0 <= ny < self.height:
          if not temp_p2_mask[ny, nx] and temp_board[ny, nx] == p2_color:
            temp_p2_mask[ny, nx] = True
            stack.append((nx, ny))

    return np.sum(temp_p1_mask), np.sum(temp_p2_mask)

  def is_game_over(self):
    p1, p2 = self.get_score()
    return (p1 + p2) >= (self.width * self.height)

  def get_territory_masks(self):
    """Returns boolean masks for both players' territories"""
    p1_mask = self.get_owner_mask(0, 0)
    p2_mask = self.get_owner_mask(self.width - 1, self.height - 1)
    return p1_mask, p2_mask

  def get_unowned_tiles(self):
    """Returns a mask of tiles that are not owned by either player"""
    p1_mask, p2_mask = self.get_territory_masks()
    return ~(p1_mask | p2_mask)

  def would_move_end_game(self, color, is_player_1=True):
    """Check if playing this color would end the game"""
    # Use test_move_score for efficiency instead of deepcopy
    result = self.test_move_score(color, is_player_1)
    if result is None:
      return False

    new_p1_score, new_p2_score = result
    total_tiles = self.width * self.height
    return (new_p1_score + new_p2_score) >= total_tiles

  def is_forced_end_state(self, is_player_1=True):
    """
    Check if the game is in a forced end state where:
    1. Any valid move would end the game, OR
    2. There are only small isolated clusters left that can't change the outcome
    """
    p1_score, p2_score = self.get_score()
    total_tiles = self.width * self.height
    owned_tiles = p1_score + p2_score
    unowned_count = total_tiles - owned_tiles

    # If very few tiles left, check if any move would end it
    if unowned_count <= 3:  # Threshold for "almost done"
      valid_actions = []
      current_color = (
        self.board[0, 0] if is_player_1 else self.board[self.height - 1, self.width - 1]
      )
      opponent_last_move = self.last_p2_move if is_player_1 else self.last_p1_move

      for action in range(6):
        if action == current_color:
          continue
        if opponent_last_move is not None and action == opponent_last_move:
          continue
        valid_actions.append(action)

      # Check if all valid moves would end the game
      if valid_actions:
        all_moves_end_game = True
        for action in valid_actions:
          if not self.would_move_end_game(action, is_player_1):
            all_moves_end_game = False
            break

        if all_moves_end_game:
          return True

    # Check if there are only isolated small clusters left that won't change outcome
    # If the score difference is large and remaining tiles are small, force end
    score_diff = p1_score - p2_score if is_player_1 else p2_score - p1_score
    if unowned_count > 0 and score_diff < -unowned_count:
      # AI is losing by more than the remaining tiles, so can't win
      # Check if remaining clusters are small and isolated
      unowned_mask = self.get_unowned_tiles()
      remaining_tiles = np.sum(unowned_mask)
      if remaining_tiles <= 5:  # Only a few tiles left
        return True

    return False
