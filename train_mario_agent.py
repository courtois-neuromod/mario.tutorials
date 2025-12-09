#!/usr/bin/env python3
"""
Train PPO agent for Super Mario Bros (NES).

Based on mario_generalization PPO implementation.
Training takes ~2 hours on CPU, ~30min on GPU.

Usage:
    python train_mario_agent.py --steps 5000000 --gpu

Saves model to: models/mario_ppo_agent.pth
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import retro
from tqdm import tqdm
import json

# Import from src
import sys

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from rl_utils import SimpleCNN
from utils import get_sourcedata_path


class PPOAgent:
    """PPO agent for training."""

    def __init__(
        self,
        n_actions=12,
        lr=1e-4,
        gamma=0.9,
        gae_lambda=0.95,
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        device="cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Model
        self.model = SimpleCNN(n_actions=n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        """Select action using current policy."""
        state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            action_logits, value = self.model(state_tensor)
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return advantages

    def update(self, states, actions, old_log_probs, returns, advantages):
        """PPO update step."""
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.tensor(actions).long().to(self.device)
        old_log_probs = torch.tensor(old_log_probs).float().to(self.device)
        returns = torch.tensor(returns).float().to(self.device)
        advantages = torch.tensor(advantages).float().to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass
        action_logits, values = self.model(states)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # PPO loss
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns)

        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item(),
        }


def action_to_buttons(action):
    """
    Convert action index to button array.

    COMPLEX_MOVEMENT action space (12 actions):
    0: NOOP
    1: right
    2: right + A (jump)
    3: right + B (run)
    4: right + A + B (run + jump)
    5: A (jump)
    6: left
    7: left + A
    8: left + B
    9: left + A + B
    10: down
    11: up

    Based on mario_generalization:
    - Intermediate format: [run, up, down, left, right, jump]
    - Final NES format: [B/run, unused, unused, unused, up, down, left, right, A/jump]
    """
    # First create 6-button format: [run, up, down, left, right, jump]
    buttons_6 = [0] * 6

    if action in (1, 2, 3, 4):  # right actions
        buttons_6[4] = 1  # right
    if action in (6, 7, 8, 9):  # left actions
        buttons_6[3] = 1  # left
    if action in (2, 4, 5, 7, 9):  # jump actions (A button)
        buttons_6[5] = 1  # jump
    if action in (3, 4, 8, 9):  # run actions (B button)
        buttons_6[0] = 1  # run
    if action == 10:  # down
        buttons_6[2] = 1
    if action == 11:  # up
        buttons_6[1] = 1

    # Convert to 9-button NES format: [run, 0, 0, 0, up, down, left, right, jump]
    buttons_9 = [buttons_6[0], 0, 0, 0] + buttons_6[1:]

    return buttons_9


def preprocess_frame(frame):
    """Preprocess frame: grayscale + resize to 84x84."""
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def create_frame_stack(frames):
    """Create 4-frame stack."""
    return np.stack(frames, axis=0)


def compute_shaped_reward(env_data, prev_data):
    """
    Compute shaped reward based on game state.

    Reward function inspired by gym-super-mario-bros with added score term:
    - v (velocity): x_pos difference - rewards rightward movement
    - c (clock): time difference - penalizes standing still
    - d (death): constant penalty when losing a life
    - s (score): scaled in-game score variations - rewards coins/enemies

    Formula: r = v + c + d + s, clipped to (-15, 15)

    Based on: https://github.com/Kautenja/gym-super-mario-bros
    Score scaling inspired by: https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch

    Parameters
    ----------
    env_data : dict
        Current step data from env.data.lookup_all()
    prev_data : dict
        Previous step data

    Returns
    -------
    float
        Shaped reward
    """
    reward = 0.0

    # v: Velocity (x_pos difference)
    # x position is split across player_x_posHi (high byte) and player_x_posLo (low byte)
    curr_x = 256 * int(env_data.get("player_x_posHi", 0)) + int(
        env_data.get("player_x_posLo", 0)
    )
    prev_x = 256 * int(prev_data.get("player_x_posHi", 0)) + int(
        prev_data.get("player_x_posLo", 0)
    )
    v = curr_x - prev_x
    reward += v

    # c: Clock penalty (time difference)
    # Time decreases as game progresses, so time_diff will be negative
    c = env_data.get("time", 0) - prev_data.get("time", 0)
    reward += c

    # d: Death penalty (constant value when losing a life)
    d = 0
    if env_data.get("lives", 2) < prev_data.get("lives", 2):
        d = -15
    reward += d

    # s: Score variations (scaled)
    # Scale down the score since it can be large (100s-1000s for coins/enemies)
    score_diff = env_data.get("score", 0) - prev_data.get("score", 0)
    s = score_diff / 40.0  # Scaling factor similar to vietnh1009's implementation
    reward += s

    # Clip to range (-15, 15) as in gym-super-mario-bros
    reward = max(min(reward, 15), -15)

    # Scale reward by 0.1 to normalize magnitude for stable training
    # This keeps relative importance but makes values more NN-friendly
    return reward * 0.1


def train_ppo(
    n_steps=5_000_000,
    rollout_length=128,
    n_epochs=4,
    batch_size=32,
    eval_interval=10000,
    save_path="models/mario_ppo_agent.pth",
    device="cpu",
    levels=None,
    render=False,
    log_path="models/training_log.json",
):
    """Train PPO agent on multiple levels."""

    # Default levels (comprehensive training set across multiple worlds)
    if levels is None:
        levels = [
            # World 1
            "Level1-1",
            "Level1-2",
            "Level1-3",
            # World 2
            "Level2-1",
        ]

    print("=" * 80)
    print("PPO Training for Super Mario Bros")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Total steps: {n_steps:,}")
    print(f"Rollout length: {rollout_length}")
    print(f"Levels: {', '.join(levels)}")
    print(f"Save path: {save_path}")
    print(f"Render: {render}")
    print("=" * 80 + "\n")

    # Import ROM from custom path
    sourcedata_path = get_sourcedata_path()
    rom_path = sourcedata_path / "mario" / "stimuli"

    if rom_path.exists():
        print(f"Importing ROM from: {rom_path}")
        retro.data.Integrations.add_custom_path(str(rom_path))
        print("✓ ROM path added to retro integrations\n")
    else:
        print(f"⚠️  ROM path not found: {rom_path}")
        print("   Falling back to default retro ROM location\n")

    # Create agent
    agent = PPOAgent(n_actions=12, device=device)

    # Environment will be recreated for each episode with random level
    render_mode = "human" if render else None
    current_level = np.random.choice(levels)

    env = retro.make(
        game="SuperMarioBros-Nes",
        state=current_level,
        inttype=retro.data.Integrations.ALL,
        render_mode=render_mode,
    )

    # Training loop
    obs, info = env.reset()  # stable-retro returns (obs, info)
    frame_stack = [preprocess_frame(obs)] * 4
    prev_data = env.data.lookup_all()  # Track previous game data for reward shaping

    episode_rewards = []
    episode_reward = 0
    step = 0
    episode = 0
    level_episodes = {level: 0 for level in levels}

    # Rollout buffers
    states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

    # Training log
    training_log = {
        "config": {
            "n_steps": n_steps,
            "rollout_length": rollout_length,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "eval_interval": eval_interval,
            "levels": levels,
            "device": device,
        },
        "progress": [],
    }

    # Progress bar
    pbar = tqdm(total=n_steps, desc="Training", unit="steps")

    while step < n_steps:
        # Collect rollout
        for _ in range(rollout_length):
            # Get state
            state = create_frame_stack(frame_stack)

            # Select action
            action, log_prob, value = agent.select_action(state)

            # Convert action to button array
            button_array = action_to_buttons(action)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(button_array)
            done = terminated or truncated

            # Get game data and compute shaped reward (CRITICAL FIX!)
            curr_data = env.data.lookup_all()
            shaped_reward = compute_shaped_reward(curr_data, prev_data)
            prev_data = curr_data

            # Store transition
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(shaped_reward)  # Use shaped reward instead of raw reward
            values.append(value)
            dones.append(done)

            # Update state
            frame_stack = frame_stack[1:] + [preprocess_frame(next_obs)]
            episode_reward += shaped_reward  # Track shaped reward
            step += 1
            pbar.update(1)

            if done:
                episode_rewards.append(episode_reward)
                episode += 1
                level_episodes[current_level] += 1

                # Switch to random level for next episode
                env.close()
                current_level = np.random.choice(levels)
                env = retro.make(
                    game="SuperMarioBros-Nes",
                    state=current_level,
                    inttype=retro.data.Integrations.ALL,
                    render_mode=render_mode,
                )

                # Reset
                obs, info = env.reset()  # stable-retro returns (obs, info)
                frame_stack = [preprocess_frame(obs)] * 4
                prev_data = env.data.lookup_all()  # Reset prev_data for new episode
                episode_reward = 0

            # Evaluation
            if step % eval_interval == 0:
                mean_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                median_reward = (
                    np.median(episode_rewards[-100:]) if episode_rewards else 0
                )
                max_reward = np.max(episode_rewards[-100:]) if episode_rewards else 0

                # Log progress
                log_entry = {
                    "step": step,
                    "episode": episode,
                    "mean_reward": float(mean_reward),
                    "median_reward": float(median_reward),
                    "max_reward": float(max_reward),
                    "n_episodes": len(episode_rewards),
                    "current_level": current_level,
                }
                training_log["progress"].append(log_entry)

                pbar.set_postfix(
                    {
                        "episode": episode,
                        "mean_reward": f"{mean_reward:.2f}",
                        "level": current_level,
                    }
                )

        # Compute returns and advantages
        advantages = agent.compute_gae(rewards, values, dones)
        returns = [adv + val for adv, val in zip(advantages, values)]

        # Update policy (multiple epochs)
        n_samples = len(states)
        indices = np.arange(n_samples)

        for epoch in range(n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_states = [states[i] for i in batch_indices]
                batch_actions = [actions[i] for i in batch_indices]
                batch_log_probs = [log_probs[i] for i in batch_indices]
                batch_returns = [returns[i] for i in batch_indices]
                batch_advantages = [advantages[i] for i in batch_indices]

                losses = agent.update(
                    batch_states,
                    batch_actions,
                    batch_log_probs,
                    batch_returns,
                    batch_advantages,
                )

        # Clear buffers
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

    # Close progress bar
    pbar.close()

    # Final evaluation
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Total episodes: {episode}")
    print(f"Final mean reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print("\nEpisodes per level:")
    for level in sorted(level_episodes.keys()):
        print(f"  {level}: {level_episodes[level]} episodes")
    print("=" * 80 + "\n")

    # Save model
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": agent.model.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "step": step,
            "episode": episode,
            "mean_reward": np.mean(episode_rewards[-100:]),
        },
        save_path,
    )

    print(f"✓ Model saved to: {save_path}\n")

    # Save training log
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    training_log["final_stats"] = {
        "total_episodes": episode,
        "final_mean_reward": float(np.mean(episode_rewards[-100:])),
        "level_episodes": level_episodes,
    }

    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"✓ Training log saved to: {log_path}\n")

    env.close()

    return agent


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for Mario")
    parser.add_argument(
        "--steps",
        type=int,
        default=5_000_000,
        help="Total training steps (default: 5M)",
    )
    parser.add_argument(
        "--rollout", type=int, default=128, help="Rollout length (default: 128)"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10000,
        help="Evaluation interval (default: 10k)",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument(
        "--render", action="store_true", help="Render gameplay during training"
    )
    parser.add_argument(
        "--levels",
        type=str,
        nargs="+",
        default=None,
        help="Mario levels to train on (default: 1-1, 1-2, 4-1, 4-2, 5-1, 5-2)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/mario_ppo_agent.pth",
        help="Where to save model",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="models/training_log.json",
        help="Where to save training log",
    )

    args = parser.parse_args()

    # Device
    if args.gpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Convert level format if provided (e.g., "1-1" -> "Level1-1")
    if args.levels is not None:
        levels = []
        for level in args.levels:
            if not level.startswith("Level"):
                # Convert "1-1" to "Level1-1"
                levels.append(f"Level{level}")
            else:
                levels.append(level)
    else:
        levels = None  # Use default

    # Train
    train_ppo(
        n_steps=args.steps,
        rollout_length=args.rollout,
        eval_interval=args.eval_interval,
        save_path=args.save_path,
        log_path=args.log_path,
        device=device,
        levels=levels,
        render=args.render,
    )


if __name__ == "__main__":
    main()
