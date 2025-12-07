"""
Visualization utilities for brain encoding section of Mario fMRI tutorial.
"""

import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting


def plot_r2_brainmap(r2_map, layer_name, threshold=0.01, vmax=0.2):
    """
    Plot R² brain map for a layer.

    Parameters
    ----------
    r2_map : Nifti1Image
        R² map as brain image
    layer_name : str
        Layer name for title
    threshold : float
        Threshold for display
    vmax : float
        Maximum value for colormap

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(16, 10))

    # Glass brain
    ax1 = plt.subplot(2, 1, 1)
    plotting.plot_glass_brain(
        r2_map,
        threshold=threshold,
        colorbar=True,
        cmap='hot',
        vmax=vmax,
        title=f'{layer_name.upper()} - Encoding Quality (R²)',
        display_mode='lyrz',
        axes=ax1
    )

    # Stat map
    ax2 = plt.subplot(2, 1, 2)
    plotting.plot_stat_map(
        r2_map,
        threshold=threshold,
        cmap='hot',
        vmax=vmax,
        colorbar=True,
        cut_coords=8,
        display_mode='z',
        title=f'{layer_name.upper()} - Axial Slices',
        axes=ax2
    )

    plt.tight_layout()

    return fig


def plot_encoding_comparison_table(comparison_df):
    """
    Plot encoding comparison as a formatted table.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison dataframe from compare_layer_performance

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=comparison_df.values,
        colLabels=comparison_df.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header
    for i in range(len(comparison_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(comparison_df) + 1):
        for j in range(len(comparison_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')

    plt.title('Layer Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    return fig


def plot_prediction_examples(bold_data, predictions, voxel_indices, test_indices):
    """
    Plot example predictions vs actual BOLD for selected voxels.

    Parameters
    ----------
    bold_data : np.ndarray
        Actual BOLD data (n_samples, n_voxels)
    predictions : np.ndarray
        Predicted BOLD data (n_samples, n_voxels)
    voxel_indices : list
        Indices of voxels to plot
    test_indices : np.ndarray
        Test set indices

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_voxels = len(voxel_indices)
    fig, axes = plt.subplots(n_voxels, 1, figsize=(14, 3*n_voxels))

    if n_voxels == 1:
        axes = [axes]

    for i, voxel_idx in enumerate(voxel_indices):
        ax = axes[i]

        actual = bold_data[test_indices, voxel_idx]
        predicted = predictions[test_indices, voxel_idx]

        # Compute R²
        from sklearn.metrics import r2_score
        r2 = r2_score(actual, predicted)

        # Plot
        ax.plot(actual, label='Actual BOLD', alpha=0.7, linewidth=1.5)
        ax.plot(predicted, label='Predicted BOLD', alpha=0.7, linewidth=1.5)

        ax.set_xlabel('Time (TRs)', fontweight='bold')
        ax.set_ylabel('BOLD', fontweight='bold')
        ax.set_title(f'Voxel {voxel_idx} (R² = {r2:.3f})',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)

    plt.tight_layout()

    return fig


def plot_layer_comparison_bars(encoding_results, layer_order):
    """
    Plot bar chart comparing layer performance.

    Parameters
    ----------
    encoding_results : dict
        Results from encoding analysis
    layer_order : list
        Order to display layers

    Returns
    -------
    matplotlib.figure.Figure
    """
    mean_r2_values = []
    for layer in layer_order:
        mean_r2 = encoding_results[layer].get('mean_r2_test',
                  encoding_results[layer].get('mean_r2_overall', 0))
        mean_r2_values.append(mean_r2)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot
    bars = ax.bar(range(len(layer_order)), mean_r2_values, color='steelblue', alpha=0.8)

    # Styling
    ax.set_xticks(range(len(layer_order)))
    ax.set_xticklabels([l.upper() for l in layer_order], fontweight='bold')
    ax.set_ylabel('Mean R² (test)', fontsize=12, fontweight='bold')
    ax.set_xlabel('CNN Layer', fontsize=12, fontweight='bold')
    ax.set_title('Encoding Performance by Layer', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

    # Add value labels on bars
    for bar, value in zip(bars, mean_r2_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Highlight best
    best_idx = np.argmax(mean_r2_values)
    bars[best_idx].set_color('orangered')
    bars[best_idx].set_alpha(1.0)

    plt.tight_layout()

    return fig
