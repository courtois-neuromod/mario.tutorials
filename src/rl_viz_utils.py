"""
Visualization utilities for RL section of Mario fMRI tutorial.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_pca_variance_per_layer(pca_results, layer_configs):
    """
    Plot PCA variance explained for each layer.

    Parameters
    ----------
    pca_results : dict
        Dictionary with layer names as keys, containing 'variance_explained'
    layer_configs : dict
        Dictionary with layer names and feature counts

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, layer_name in enumerate(layer_configs.keys()):
        ax = axes[idx]

        variance = pca_results[layer_name]['variance_explained']
        cumsum_var = np.cumsum(variance)

        # Bar plot
        ax.bar(range(len(variance)), variance, alpha=0.7, color='steelblue')

        # Cumulative line
        ax2 = ax.twinx()
        ax2.plot(range(len(cumsum_var)), cumsum_var,
                color='orangered', linewidth=2.5, marker='o', markersize=4)
        ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax2.set_ylim([0, 1.05])
        ax2.set_ylabel('Cumulative', fontsize=11, color='orangered', fontweight='bold')

        ax.set_xlabel('Component', fontsize=11, fontweight='bold')
        ax.set_ylabel('Variance', fontsize=11, color='steelblue', fontweight='bold')
        ax.set_title(f'{layer_name.upper()}', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

        # Total variance text
        total = cumsum_var[-1]
        ax.text(0.95, 0.95, f'{total*100:.1f}%',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
               fontsize=12, fontweight='bold')

    axes[-1].axis('off')

    plt.suptitle('PCA Variance Explained per Layer',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    return fig


def plot_layer_activations_sample(layer_activations, layer_name, n_trs=200, n_features=10):
    """
    Plot sample activations from a layer.

    Parameters
    ----------
    layer_activations : dict
        Dictionary mapping layer names to activation arrays
    layer_name : str
        Which layer to plot
    n_trs : int
        Number of TRs to show
    n_features : int
        Number of features to show

    Returns
    -------
    matplotlib.figure.Figure
    """
    acts = layer_activations[layer_name][:n_trs, :n_features]

    fig, ax = plt.subplots(figsize=(14, 6))

    im = ax.imshow(acts.T, aspect='auto', cmap='RdBu_r',
                   vmin=-2, vmax=2, interpolation='nearest')

    ax.set_xlabel('Time (TRs)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title(f'{layer_name.upper()} Activations (HRF-convolved)',
                 fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Activation (z-scored)')
    plt.tight_layout()

    return fig


def plot_agent_gameplay(frames, actions, rewards, max_frames=300):
    """
    Visualize agent gameplay with frames, actions, and rewards.

    Parameters
    ----------
    frames : list
        List of game frames (RGB images)
    actions : list
        List of action indices
    rewards : list
        List of rewards
    max_frames : int
        Maximum number of frames to plot

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_frames = min(len(frames), max_frames)

    fig, axes = plt.subplots(3, 1, figsize=(16, 9))

    # Action timeline
    ax = axes[0]
    action_array = np.array(actions[:n_frames])
    ax.plot(action_array, marker='.', linestyle='-', linewidth=0.5, markersize=3)
    ax.set_ylabel('Action', fontweight='bold')
    ax.set_title('Agent Gameplay', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    # Reward timeline
    ax = axes[1]
    reward_array = np.array(rewards[:n_frames])
    ax.fill_between(range(n_frames), reward_array, alpha=0.6, color='green')
    ax.set_ylabel('Reward', fontweight='bold')
    ax.grid(alpha=0.3)

    # Sample frames
    ax = axes[2]
    ax.axis('off')

    # Show 6 sample frames
    n_samples = 6
    sample_indices = np.linspace(0, n_frames-1, n_samples, dtype=int)

    for i, idx in enumerate(sample_indices):
        ax_img = fig.add_subplot(3, n_samples, 2*n_samples + i + 1)
        ax_img.imshow(frames[idx])
        ax_img.set_title(f'Frame {idx}', fontsize=9)
        ax_img.axis('off')

    plt.tight_layout()

    return fig
