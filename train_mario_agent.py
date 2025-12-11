#!/usr/bin/env python3
"""
Integrated PPO Trainer for Super Mario Bros (NES).

Features:
- Custom PPO implementation (from Code B)
- Robust CLI and Level Parsing (from Code A)
- Dynamic file naming and logging
- Frame Skipping & Shaping
- Replay GIF generation
"""

import argparse
import os
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import retro
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import time

try:
    import imageio
except ImportError:
    imageio = None
    print(
        "Warning: imageio not installed. Install with 'pip install imageio' for GIF replay support."
    )

# Import from src (assuming existing project structure)
# If running standalone, ensure these imports work or paste utils here
try:
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    from rl import SimpleCNN
    from utils import get_sourcedata_path
except ImportError:
    print(
        "Warning: Could not import 'src' modules. Ensure project structure is correct."
    )

    # Fallback/Placeholder if modules missing (for standalone testing)
    def get_sourcedata_path():
        return Path("data")

    class SimpleCNN(nn.Module):  # Placeholder SimpleCNN if import fails
        def __init__(self, input_shape=(4, 84, 84), n_actions=12):
            super().__init__()
            self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            self.fc1 = nn.Linear(3136, 512)
            self.actor = nn.Linear(512, n_actions)
            self.critic = nn.Linear(512, 1)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.reshape(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.actor(x), self.critic(x)


ALL_LEVELS = [
    "w1l1",
    "w1l2",
    "w1l3",
    "w2l1",
    "w2l3",
    "w3l1",
    "w3l2",
    "w3l3",
    "w4l1",
    "w4l2",
    "w4l3",
    "w5l1",
    "w5l2",
    "w5l3",
    "w6l1",
    "w6l2",
    "w6l3",
    "w7l1",
    "w7l3",
    "w8l1",
    "w8l2",
    "w8l3",
]


def parse_levels(levels_arg):
    """Parse levels argument into list of level names."""
    if levels_arg.lower() == "all":
        return ALL_LEVELS
    return [level.strip() for level in levels_arg.split(",")]


def convert_level_name(lvl):
    """
    Convert shorthand 'w1l1' to Retro format 'Level1-1'.
    """
    if lvl.startswith("Level"):
        return lvl  # Already correct
    try:
        # Assumes format wXlY -> LevelX-Y
        world = lvl[1]
        level = lvl[3]
        return f"Level{world}-{level}"
    except IndexError:
        return lvl  # Return as is if format doesn't match


def create_model_filename(levels, total_timesteps):
    """Create unique filename based on training parameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if len(levels) == 1:
        level_str = levels[0]
    elif len(levels) == len(ALL_LEVELS):
        level_str = "all_levels"
    else:
        level_str = f"{len(levels)}_levels"

    timesteps_str = f"{total_timesteps//1000}k"
    filename = f"ppo_mario_{level_str}_{timesteps_str}_{timestamp}"

    return filename


class PPOAgent:
    """PPO agent for training."""

    def __init__(
        self,
        n_actions=12,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.02,
        device="cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.lr = lr

        self.model = SimpleCNN(n_actions=n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

    def select_action(self, state):
        state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            action_logits, value = self.model(state_tensor)
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        next_val = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            next_val = values[t]
        return advantages

    def update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.tensor(actions).long().to(self.device)
        old_log_probs = torch.tensor(old_log_probs).float().to(self.device)
        returns = torch.tensor(returns).float().to(self.device)
        advantages = torch.tensor(advantages).float().to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        action_logits, values = self.model(states)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values.squeeze(), returns)

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        return loss.item()

    def update_learning_rate(self, current_step, total_steps):
        frac = 1.0 - (current_step / total_steps)
        new_lr = self.lr * frac
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr


def action_to_buttons(action):
    buttons_6 = [0] * 6
    if action in (1, 2, 3, 4):
        buttons_6[4] = 1  # right
    if action in (6, 7, 8, 9):
        buttons_6[3] = 1  # left
    if action in (2, 4, 5, 7, 9):
        buttons_6[5] = 1  # jump
    if action in (3, 4, 8, 9):
        buttons_6[0] = 1  # run
    if action == 10:
        buttons_6[2] = 1  # down
    if action == 11:
        buttons_6[1] = 1  # up
    return [buttons_6[0], 0, 0, 0] + buttons_6[1:]


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def create_frame_stack(frames):
    return np.stack(frames, axis=0)


def compute_shaped_reward(env_data, prev_data):
    reward = 0.0
    curr_x = 256 * int(env_data.get("player_x_posHi", 0)) + int(
        env_data.get("player_x_posLo", 0)
    )
    prev_x = 256 * int(prev_data.get("player_x_posHi", 0)) + int(
        prev_data.get("player_x_posLo", 0)
    )
    reward += curr_x - prev_x
    reward += env_data.get("time", 0) - prev_data.get("time", 0)
    if env_data.get("lives", 2) < prev_data.get("lives", 2):
        reward += -15
    reward += (env_data.get("score", 0) - prev_data.get("score", 0)) / 40.0
    return max(min(reward * 0.1, 15), -15)


def save_replay_gif(agent, level, gif_path, device="cpu", max_steps=2000):
    """
    Record a replay of the trained agent and save as GIF.

    Args:
        agent: Trained PPOAgent
        level: Level code (e.g., 'w1l1')
        gif_path: Path to save GIF file
        device: Device for inference
        max_steps: Maximum steps to record
    """
    if imageio is None:
        print("Cannot save replay: imageio not installed")
        return

    print(f"\nRecording replay for {level}...")

    # Convert level name
    state = convert_level_name(level)

    # Create environment
    try:
        env = retro.make(
            game="SuperMarioBros-Nes",
            state=state,
            inttype=retro.data.Integrations.ALL,
            render_mode=None,
        )
    except FileNotFoundError:
        print(f"Error: Could not load state '{state}' for replay")
        return

    obs, info = env.reset()
    frame_stack = [preprocess_frame(obs)] * 4

    frames = []
    done = False
    step = 0
    frame_counter = 0  # Track frames to skip every other one

    agent.model.eval()

    with torch.no_grad():
        while not done and step < max_steps:
            # Get action
            state = create_frame_stack(frame_stack)
            state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(device)
            action_logits, _ = agent.model(state_tensor)
            dist = Categorical(logits=action_logits)
            action = dist.sample().item()

            button_array = action_to_buttons(action)

            # Step with frame skip - record every other frame only
            for _ in range(4):
                next_obs, _, terminated, truncated, _ = env.step(button_array)
                done = terminated or truncated
                # Record only every other frame (skip odd frames)
                if frame_counter % 2 == 0:
                    frames.append(next_obs)
                frame_counter += 1
                frame_stack = frame_stack[1:] + [preprocess_frame(next_obs)]
                if done:
                    break

            step += 1

    env.close()

    # Save GIF at 60 FPS - since we skip every other frame, this gives 2x speedup
    print(f"Saving replay GIF with {len(frames)} frames to {gif_path}")
    imageio.mimsave(gif_path, frames, duration=1000 / 60)
    print(f"Replay saved successfully!")


def train_ppo(
    levels,  # Now accepts list of levels
    model_path,  # Exact path to save model
    log_path,  # Exact path to save logs
    n_steps=1_000_000,
    rollout_length=2048,
    n_epochs=10,
    batch_size=64,
    device="cpu",
    render_freq=0,  # Render every N episodes (0 = disabled)
    gif_freq=0,  # Save GIF every N episodes (0 = disabled, except final)
    save_checkpoints=False,
    checkpoint_freq=100000,
):
    print("=" * 80)
    print(f"PPO Training | Device: {device} | Steps: {n_steps:,}")
    print(f"Levels ({len(levels)}): {', '.join(levels)}")
    print(f"Model Path: {model_path}")
    if render_freq > 0:
        print(f"Visual Rendering: Every {render_freq} episodes")
    if gif_freq > 0:
        print(f"Saving GIF replays: Every {gif_freq} episodes")
    else:
        print(f"Saving final GIF replay only")
    print("=" * 80 + "\n")

    # Load ROM
    sourcedata_path = get_sourcedata_path()
    rom_path = sourcedata_path / "mario" / "stimuli"
    if rom_path.exists():
        retro.data.Integrations.add_custom_path(str(rom_path))

    agent = PPOAgent(n_actions=12, device=device)

    # Init Environment
    # Start without rendering for performance
    render_mode = None

    # Select random initial level and convert to Retro format
    current_lvl_code = np.random.choice(levels)
    current_state = convert_level_name(current_lvl_code)

    print(f"Initializing Environment with state: {current_state}")

    try:
        env = retro.make(
            game="SuperMarioBros-Nes",
            state=current_state,
            inttype=retro.data.Integrations.ALL,
            render_mode=render_mode,
        )
    except FileNotFoundError:
        print(
            f"Error: State '{current_state}' not found. Check your retro integrations."
        )
        return

    obs, info = env.reset()
    frame_stack = [preprocess_frame(obs)] * 4
    prev_data = env.data.lookup_all()

    episode_rewards = []
    episode_reward = 0
    step = 0
    episode = 0

    # Initialize Log
    training_log = {
        "config": {
            "n_steps": n_steps,
            "rollout_length": rollout_length,
            "levels": levels,
            "timestamp": datetime.now().isoformat(),
        },
        "progress": [],
    }

    # Buffers
    states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
    pbar = tqdm(total=n_steps, desc="Training")

    # For recording episodes as GIFs
    episode_frames = []
    is_recording = False
    should_display = False  # Separate flag for visual rendering
    frame_counter = 0  # Track frames to skip every other one

    while step < n_steps:
        agent.update_learning_rate(step, n_steps)

        # --- ROLLOUT ---
        for _ in range(rollout_length):
            state = create_frame_stack(frame_stack)
            action, log_prob, value = agent.select_action(state)
            button_array = action_to_buttons(action)

            accumulated_reward = 0
            for skip_idx in range(4):  # Frame skip
                next_obs, r, terminated, truncated, info = env.step(button_array)
                done = terminated or truncated

                curr_data = env.data.lookup_all()
                shaped_r = compute_shaped_reward(curr_data, prev_data)
                prev_data = curr_data
                accumulated_reward += shaped_r

                frame_stack = frame_stack[1:] + [preprocess_frame(next_obs)]

                # Record only every other frame if we're recording (for 2x speedup)
                if is_recording:
                    if frame_counter % 2 == 0:
                        episode_frames.append(next_obs)
                    frame_counter += 1

                if done:
                    break

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(accumulated_reward)
            values.append(value)
            dones.append(done)

            episode_reward += accumulated_reward
            step += 1
            pbar.update(1)

            if done:
                episode_rewards.append(episode_reward)
                episode += 1

                # Save GIF if we were recording this episode
                if is_recording and len(episode_frames) > 0:
                    replay_dir = Path(model_path).parent / "replays"
                    replay_dir.mkdir(parents=True, exist_ok=True)
                    gif_filename = (
                        f"episode_{episode}_step_{step}_{current_lvl_code}.gif"
                    )
                    gif_path = replay_dir / gif_filename

                    if imageio is not None:
                        print(
                            f"\nSaving GIF for episode {episode} ({len(episode_frames)} frames)"
                        )
                        # 60 FPS with half the frames = 2x speedup
                        imageio.mimsave(gif_path, episode_frames, duration=1000 / 60)

                    episode_frames = []
                    frame_counter = 0  # Reset frame counter for next episode

                # Properly close the environment
                env.close()
                del env

                # If we were rendering, give the window time to close
                if should_display:
                    time.sleep(0.1)

                # Pick new random level
                current_lvl_code = np.random.choice(levels)
                current_state = convert_level_name(current_lvl_code)

                # Determine if we should render (visual display) this episode
                should_display = render_freq > 0 and episode % render_freq == 0
                # Determine if we should record (save GIF) this episode
                is_recording = gif_freq > 0 and episode % gif_freq == 0

                current_render_mode = "human" if should_display else None

                env = retro.make(
                    game="SuperMarioBros-Nes",
                    state=current_state,
                    inttype=retro.data.Integrations.ALL,
                    render_mode=current_render_mode,
                )
                obs, info = env.reset()
                frame_stack = [preprocess_frame(obs)] * 4
                prev_data = env.data.lookup_all()
                episode_reward = 0

            # Logging
            if step % 10000 == 0:
                mean_r = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0
                max_r = np.max(episode_rewards[-100:]) if episode_rewards else 0.0
                current_lr = agent.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"Mean R": f"{mean_r:.1f}", "LR": f"{current_lr:.6f}"})

                training_log["progress"].append(
                    {
                        "step": step,
                        "mean_reward": float(mean_r),
                        "max_reward": float(max_r),
                        "lr": float(current_lr),
                    }
                )

            # Checkpoint
            if save_checkpoints and step % checkpoint_freq == 0:
                ckpt_dir = Path(model_path).parent / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"{Path(model_path).stem}_{step // 1000}k.pth"
                torch.save(agent.model.state_dict(), ckpt_path)

        # --- UPDATE ---
        advantages = agent.compute_gae(rewards, values, dones)
        returns = [adv + val for adv, val in zip(advantages, values)]

        n_samples = len(states)
        indices = np.arange(n_samples)

        for epoch in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                agent.update(
                    [states[i] for i in batch_idx],
                    [actions[i] for i in batch_idx],
                    [log_probs[i] for i in batch_idx],
                    [returns[i] for i in batch_idx],
                    [advantages[i] for i in batch_idx],
                )

        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

        # Periodic Save (Model & Log)
        if step % (n_steps // 10) == 0:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(agent.model.state_dict(), model_path)
            with open(log_path, "w") as f:
                json.dump(training_log, f, indent=2)

    pbar.close()
    env.close()
    del env

    # Allow final window to close if it was open
    time.sleep(0.2)

    # Final Save
    torch.save(agent.model.state_dict(), model_path)
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"Training Complete. Saved to {model_path}")

    # Always save a final replay GIF
    gif_filename = f"{Path(model_path).stem}_final_replay.gif"
    gif_path = Path(model_path).parent / gif_filename
    # Use first level for replay
    replay_level = levels[0]
    save_replay_gif(agent, replay_level, gif_path, device=device)


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agent on Super Mario Bros (Custom Implementation)"
    )

    # Code A Arguments
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000000,
        help="Total training timesteps (default: 1000000)",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="w1l1",
        help='Levels to train on: comma-separated (e.g., "w1l1,w1l2") or "all"',
    )
    parser.add_argument(
        "--save_checkpoints",
        action="store_true",
        help="Save model checkpoints during training",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=100000,
        help="Checkpoint frequency in timesteps",
    )

    # Code B Arguments (merged)
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument(
        "--render_freq",
        type=int,
        default=0,
        help="Display gameplay every N episodes (0 = no visual rendering). Example: --render_freq 10",
    )
    parser.add_argument(
        "--gif_freq",
        type=int,
        default=0,
        help="Save GIF replay every N episodes (0 = final replay only). Example: --gif_freq 50",
    )

    args = parser.parse_args()

    levels = parse_levels(args.levels)
    print(f"Training Configured for {len(levels)} level(s): {', '.join(levels)}")

    model_filename = create_model_filename(levels, args.timesteps)
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"{model_filename}.pth")
    log_path = os.path.join(output_dir, f"{model_filename}.json")

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    train_ppo(
        levels=levels,
        model_path=model_path,
        log_path=log_path,
        n_steps=args.timesteps,
        device=device,
        render_freq=args.render_freq,
        gif_freq=args.gif_freq,
        save_checkpoints=args.save_checkpoints,
        checkpoint_freq=args.checkpoint_freq,
    )


if __name__ == "__main__":
    main()
