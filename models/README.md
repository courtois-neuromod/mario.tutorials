# Trained Models Directory

This directory stores trained RL agent weights.

## Files

- `mario_ppo_agent.pth` - Trained PPO agent (created by scripts/train_mario_agent.py)

## Training

To train an agent:

```bash
python scripts/train_mario_agent.py --steps 5000000  # Full training (~2 hours)
python scripts/train_mario_agent.py --steps 10000    # Quick demo (~2 minutes)
```

The script will save weights here automatically.
