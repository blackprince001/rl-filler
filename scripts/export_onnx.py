"""Export the trained SB3 DQN policy network to ONNX.

Run with the *training* dependency group (it needs torch + stable-baselines3):

    uv run --group train python scripts/export_onnx.py

Reads `backend/models/floodit_dqn.zip` and writes
`backend/models/floodit_dqn.onnx`. A short numerical-equivalence check is
run at the end so a silent op-fusion drift gets caught.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN

ROOT = Path(__file__).resolve().parent.parent
SB3_MODEL = ROOT / "backend" / "models" / "floodit_dqn"
ONNX_OUT = ROOT / "backend" / "models" / "floodit_dqn.onnx"

OBS_SHAPE = (8, 7, 6)


def _random_obs(rng: np.random.Generator) -> np.ndarray:
  """One-hot observation matching the env spec."""
  board = rng.integers(0, 6, size=OBS_SHAPE[:2])
  obs = np.zeros(OBS_SHAPE, dtype=np.float32)
  for r in range(OBS_SHAPE[0]):
    for c in range(OBS_SHAPE[1]):
      obs[r, c, int(board[r, c])] = 1.0
  return obs


def main() -> int:
  if not SB3_MODEL.with_suffix(".zip").exists():
    print(f"Model not found at {SB3_MODEL}.zip", file=sys.stderr)
    return 1

  print(f"Loading {SB3_MODEL}.zip on CPU…")
  model = DQN.load(str(SB3_MODEL), device="cpu")
  q_net = model.policy.q_net.to("cpu").eval()

  dummy = torch.zeros(1, *OBS_SHAPE, dtype=torch.float32)
  ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)
  print(f"Exporting → {ONNX_OUT}")
  # dynamo=False uses the legacy TorchScript path, which writes a single
  # self-contained .onnx file. The dynamo path splits weights into a
  # sibling .onnx.data, which complicates packaging without buying us
  # anything for a model this small.
  torch.onnx.export(
    q_net,
    dummy,
    str(ONNX_OUT),
    input_names=["obs"],
    output_names=["q_values"],
    dynamic_axes={"obs": {0: "batch"}, "q_values": {0: "batch"}},
    opset_version=17,
    dynamo=False,
  )

  # Numerical equivalence: argmax must agree between torch and ORT on a
  # batch of random one-hot boards. Q-values can drift by ~1e-5 because of
  # op fusion; argmax should be identical.
  try:
    import onnxruntime as ort  # noqa: WPS433 — optional check
  except ImportError:
    print("(skipping equivalence check: onnxruntime not installed)")
    return 0

  rng = np.random.default_rng(42)
  batch = np.stack([_random_obs(rng) for _ in range(64)])  # (64, 8, 7, 6)

  with torch.no_grad():
    torch_q = q_net(torch.from_numpy(batch)).cpu().numpy()
  ort_q = ort.InferenceSession(str(ONNX_OUT), providers=["CPUExecutionProvider"]).run(
    ["q_values"], {"obs": batch}
  )[0]

  max_abs = float(np.max(np.abs(torch_q - ort_q)))
  same_argmax = int(np.sum(torch_q.argmax(1) == ort_q.argmax(1)))
  print(f"  max |Δ| = {max_abs:.6f}")
  print(f"  argmax match: {same_argmax}/{batch.shape[0]}")
  if same_argmax != batch.shape[0]:
    print("Argmax mismatch — refusing to commit a drifted export.", file=sys.stderr)
    return 2
  print("OK")
  return 0


if __name__ == "__main__":
  sys.exit(main())
