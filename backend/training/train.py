import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
  sys.path.insert(0, project_root)

from stable_baselines3 import DQN

from backend.gym_env import FloodItEnv


def main():
  # 1. Create Env
  env = FloodItEnv()

  # 2. Define Model (MlpPolicy for one-hot encoded grid observations)
  model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.0001)

  # 3. Train
  print("Starting training... this may take a while.")
  model.learn(total_timesteps=500_000)  # Increase to 1M+ for good results

  # 4. Save
  models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
  os.makedirs(models_dir, exist_ok=True)
  model.save(os.path.join(models_dir, "floodit_dqn"))
  print("Model saved.")


if __name__ == "__main__":
  main()
