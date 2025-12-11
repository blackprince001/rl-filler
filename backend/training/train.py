import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
  sys.path.insert(0, project_root)

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

import wandb
from backend.gym_env import FloodItEnv


class WandbCallback(BaseCallback):
  def __init__(self, verbose=0):
    super().__init__(verbose)
    self.episode_count = 0
    self.last_log_time = 0

  def _on_step(self) -> bool:
    if self.logger is not None and self.num_timesteps - self.last_log_time >= 100:
      for key, value in self.logger.name_to_value.items():
        if isinstance(value, (int, float)):
          # Clean up key names for wandb
          clean_key = key.replace("/", "_").replace(" ", "_").lower()
          wandb.log({f"train/{clean_key}": value}, step=self.num_timesteps)
      self.last_log_time = self.num_timesteps

    # Log episode info if available
    if len(self.locals.get("infos", [])) > 0:
      for info in self.locals["infos"]:
        if "episode" in info:
          episode_info = info["episode"]
          self.episode_count += 1
          wandb.log(
            {
              "episode/reward": episode_info.get("r", 0),
              "episode/length": episode_info.get("l", 0),
              "episode/time": episode_info.get("t", 0),
              "episode/episode": self.episode_count,
            },
            step=self.num_timesteps,
          )

    return True

  def _on_training_end(self) -> None:
    """Called when training ends. Log final summary."""
    if self.logger is not None:
      # Log final metrics
      for key, value in self.logger.name_to_value.items():
        if isinstance(value, (int, float)):
          clean_key = key.replace("/", "_").replace(" ", "_").lower()
          wandb.summary[f"final_{clean_key}"] = value


class WandbEvalCallback(EvalCallback):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.last_logged_eval_idx = -1

  def _on_step(self) -> bool:
    result = super()._on_step()

    # Log evaluation results to wandb using current training step
    # Only log new evaluations (not ones we've already logged)
    if self.evaluations_timesteps and len(self.evaluations_timesteps) > 0:
      last_eval_idx = len(self.evaluations_timesteps) - 1

      if last_eval_idx > self.last_logged_eval_idx and last_eval_idx < len(
        self.evaluations_results
      ):
        mean_reward = np.mean(self.evaluations_results[last_eval_idx])
        std_reward = np.std(self.evaluations_results[last_eval_idx])
        mean_ep_length = (
          np.mean(self.evaluations_length[last_eval_idx])
          if last_eval_idx < len(self.evaluations_length)
          else 0
        )
        # Use current training step, not evaluation timestep
        wandb.log(
          {
            "eval/mean_reward": mean_reward,
            "eval/std_reward": std_reward,
            "eval/mean_ep_length": mean_ep_length,
          },
          step=self.num_timesteps,  # Use current training step
        )
        self.last_logged_eval_idx = last_eval_idx

    return result


class CurriculumLearningCallback(BaseCallback):
  def __init__(self, switch_timestep=300_000, verbose=0):
    super().__init__(verbose)
    self.switch_timestep = switch_timestep
    self.switched = False

  def _on_step(self) -> bool:
    if not self.switched and self.num_timesteps >= self.switch_timestep:
      try:
        if hasattr(self.training_env, "envs") and self.training_env.envs is not None:
          envs_list = self.training_env.envs
          if isinstance(envs_list, (list, tuple)) or hasattr(envs_list, "__iter__"):
            for env_wrapper in envs_list:  # type: ignore[attr-defined]
              env = getattr(env_wrapper, "env", env_wrapper)
              if hasattr(env, "opponent_type"):
                env.opponent_type = "random"  # type: ignore[assignment]
                if self.verbose > 0:
                  print(
                    f"✓ Switched to random opponent at timestep {self.num_timesteps}"
                  )
        else:
          env = self.training_env
          if hasattr(env, "env"):
            env = env.env
          if hasattr(env, "opponent_type"):
            env.opponent_type = "random"  # type: ignore[assignment]
            if self.verbose > 0:
              print(f"✓ Switched to random opponent at timestep {self.num_timesteps}")
      except (AttributeError, TypeError) as e:
        if self.verbose > 0:
          print(f"Warning: Could not switch opponent type: {e}")

      self.switched = True
      wandb.log({"curriculum/opponent_type": 1}, step=self.num_timesteps)

    return True


def main():
  # Check and set device
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device == "cpu":
    print("Warning: CUDA not available, using CPU. Training will be slower.")
    print("To use GPU, ensure CUDA is properly installed and configured.")
  else:
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

  # Initialize wandb
  wandb.init(
    project="floodit-rl",
    name="dqn-floodit",
    config={
      "algorithm": "DQN",
      "policy": "MlpPolicy",
      "learning_rate": 0.0001,
      "total_timesteps": 1_000_000,
      "env": "FloodItEnv",
      "board_size": "8x7",
      "n_colors": 6,
      "device": device,
      "curriculum_learning": True,
      "opponent_start": "greedy",
      "opponent_switch_timestep": 300_000,
    },
    sync_tensorboard=False,
    monitor_gym=True,
  )

  try:
    # Create Env with Monitor for logging
    # Start with greedy opponent for curriculum learning
    env = FloodItEnv(opponent_type="greedy")
    env = Monitor(env, filename=None, allow_early_resets=True)

    # Create evaluation environment (use random for evaluation)
    eval_env = FloodItEnv(opponent_type="random")
    eval_env = Monitor(eval_env)

    # Define Model (MlpPolicy for one-hot encoded grid observations)
    model = DQN(
      "MlpPolicy",
      env,
      verbose=1,
      learning_rate=0.0001,
      tensorboard_log=None,
      device=device,
      exploration_fraction=0.3,
      exploration_initial_eps=1.0,
      exploration_final_eps=0.05,
      buffer_size=1_000_000,
      learning_starts=500,
      batch_size=64,
      gamma=0.42,
      target_update_interval=500,
      train_freq=(4, "step"),
      gradient_steps=1,
    )

    # Set up callbacks
    # Create eval log directory for EvalCallback
    eval_log_dir = os.path.join(
      os.path.dirname(os.path.dirname(__file__)), "logs", "evaluations"
    )
    os.makedirs(eval_log_dir, exist_ok=True)

    eval_callback = WandbEvalCallback(
      eval_env,
      best_model_save_path=None,
      log_path=eval_log_dir,
      eval_freq=10000,
      deterministic=True,
      render=False,
    )

    wandb_callback = WandbCallback()

    # Curriculum learning: switch from greedy to random at 30% of training
    curriculum_callback = CurriculumLearningCallback(
      switch_timestep=int(1_000_000 * 0.5),  # Switch at 30% of training
      verbose=1,
    )

    # Combine callbacks
    callback_list = CallbackList([eval_callback, wandb_callback, curriculum_callback])

    # Train
    url = None
    if wandb.run is not None and wandb.run.url is not None:
      url = wandb.run.url

    print("Starting training... this may take a while.")
    if url:
      print(f"View training progress at: {url}")

    model.learn(
      total_timesteps=1_000_000,
      callback=callback_list,
      progress_bar=True,
    )

    # Save
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "floodit_dqn")
    model.save(model_path)

    # Log model artifact to wandb
    artifact = wandb.Artifact("floodit-dqn-model", type="model")
    artifact.add_file(model_path + ".zip")
    wandb.log_artifact(artifact)

    print(f"Model saved to {model_path}")
    if wandb.run is not None and wandb.run.url is not None:
      print(f"Training complete! View results at: {url}")
    else:
      print("Training complete!")

  finally:
    wandb.finish()


if __name__ == "__main__":
  main()
