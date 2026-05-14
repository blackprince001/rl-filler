"""Thin wrapper around onnxruntime so callers stay numpy-only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class QValueRunner:
  """Loads `floodit_dqn.onnx` and returns Q-values for a single observation."""

  def __init__(self, model_path: Path) -> None:
    self.model_path = model_path
    self.session = ort.InferenceSession(
      str(model_path),
      providers=["CPUExecutionProvider"],
    )
    self._input_name = self.session.get_inputs()[0].name
    self._output_name = self.session.get_outputs()[0].name

  def q_values(self, board) -> np.ndarray:
    """Build a (1, 8, 7, 6) one-hot observation, return a (6,) Q-vector."""
    obs = np.zeros((1, 8, 7, 6), dtype=np.float32)
    for r in range(8):
      for c in range(7):
        obs[0, r, c, int(board[r][c])] = 1.0
    out = self.session.run([self._output_name], {self._input_name: obs})[0]
    return out[0]
