#!/usr/bin/env python3
"""
Generate PPO Agent CNN architecture diagram using PlotNeuralNet.

This script creates a beautiful LaTeX/TikZ visualization of the SimpleCNN
architecture used for the Mario RL agent.

Usage:
    python generate_architecture_diagram.py

This will generate:
    - ppo_architecture.tex (LaTeX source)

To compile to PDF:
    bash PlotNeuralNet/tikzmake.sh ppo_architecture
"""

import sys
sys.path.append('PlotNeuralNet/')
from pycore.tikzeng import *

# Define PPO CNN architecture
arch = [
    to_head('PlotNeuralNet'),
    to_cor(),
    to_begin(),

    # Input: 4 stacked frames (84x84) - reduced height for wider aspect ratio
    to_Conv("input", s_filer=84, n_filer=4, offset="(0,0,0)", to="(0,0,0)",
            height=30, depth=30, width=1, caption="Input"),

    # Conv1: 4 -> 32 channels, kernel 3x3, stride 2 -> 42x42
    to_Conv("conv1", s_filer=42, n_filer=32, offset="(3,0,0)", to="(input-east)",
            height=24, depth=24, width=3, caption="Conv1"),
    to_connection("input", "conv1"),

    # Pool1: MaxPool 2x2 -> 21x21
    to_Pool("pool1", offset="(0.5,0,0)", to="(conv1-east)",
            height=24, depth=24, width=1, opacity=0.5, caption="Pool"),

    # Conv2: 32 -> 32 channels, kernel 3x3, stride 2 -> 21x21
    to_Conv("conv2", s_filer=21, n_filer=32, offset="(3,0,0)", to="(pool1-east)",
            height=18, depth=18, width=3, caption="Conv2"),
    to_connection("pool1", "conv2"),

    # Pool2: MaxPool 2x2 -> 11x11
    to_Pool("pool2", offset="(0.5,0,0)", to="(conv2-east)",
            height=18, depth=18, width=1, opacity=0.5, caption="Pool"),

    # Conv3: 32 -> 32 channels, kernel 3x3, stride 2 -> 11x11
    to_Conv("conv3", s_filer=11, n_filer=32, offset="(3,0,0)", to="(pool2-east)",
            height=12, depth=12, width=3, caption="Conv3"),
    to_connection("pool2", "conv3"),

    # Pool3: MaxPool 2x2 -> 6x6
    to_Pool("pool3", offset="(0.5,0,0)", to="(conv3-east)",
            height=12, depth=12, width=1, opacity=0.5, caption="Pool"),

    # Conv4: 32 -> 32 channels, kernel 3x3, stride 2 -> 6x6
    to_Conv("conv4", s_filer=6, n_filer=32, offset="(3,0,0)", to="(pool3-east)",
            height=8, depth=8, width=3, caption="Conv4"),
    to_connection("pool3", "conv4"),

    # Pool4: MaxPool 2x2 -> 3x3
    to_Pool("pool4", offset="(0.5,0,0)", to="(conv4-east)",
            height=8, depth=8, width=1, opacity=0.5, caption="Pool"),

    # Flatten: 32*3*3 = 1152
    to_SoftMax("flatten", s_filer=1152, offset="(3,0,0)", to="(pool4-east)",
               width=1.5, height=6, depth=6, opacity=0.8, caption="Flatten"),
    to_connection("pool4", "flatten"),

    # Linear: 1152 -> 512
    to_SoftMax("linear", s_filer=512, offset="(3,0,0)", to="(flatten-east)",
               width=2, height=8, depth=8, opacity=0.8, caption="Linear"),
    to_connection("flatten", "linear"),

    # Actor head: 512 -> 12 actions
    to_SoftMax("actor", s_filer=12, offset="(3.5,2,0)", to="(linear-east)",
               width=1.5, height=5, depth=5, opacity=0.8, caption="Actor"),
    to_connection("linear", "actor"),

    # Critic head: 512 -> 1 value
    to_SoftMax("critic", s_filer=1, offset="(3.5,-2,0)", to="(linear-east)",
               width=1.5, height=4, depth=4, opacity=0.8, caption="Critic"),
    to_connection("linear", "critic"),

    to_end()
]

def main():
    # Generate the LaTeX file
    to_generate(arch, 'ppo_architecture.tex')
    print("✓ Generated ppo_architecture.tex")
    print("\nTo compile to PDF:")
    print("  bash PlotNeuralNet/tikzmake.sh ppo_architecture")
    print("\nOr manually:")
    print("  pdflatex ppo_architecture.tex")
    print("\nThe PDF will be: ppo_architecture.pdf")

if __name__ == '__main__':
    main()
