"""
Visualization utilities for Mario fMRI tutorial.

This module consolidates all visualization functions from:
- GLM analysis (from glm_utils.py)
- RL agent (from rl_viz_utils.py)
- Encoding models (from encoding_viz_utils.py and atlas_encoding_utils.py)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting
from pathlib import Path



# ==============================================================================
# GLM VISUALIZATIONS
# ==============================================================================

# Visualization functions from glm_utils.py will be added here

# ==============================================================================
# RL VISUALIZATIONS
# ==============================================================================

# Visualization functions from rl_viz_utils.py will be added here

# ==============================================================================
# ENCODING VISUALIZATIONS
# ==============================================================================

# Visualization functions from encoding_viz_utils.py and atlas_encoding_utils.py will be added here


# ==============================================================================
# GLM VISUALIZATIONS (from glm_utils.py)
# ==============================================================================

def plot_event_frequencies(session_events, replay_metadata, subject, session, figsize=(16, 6)):
    """
    Plot event frequency breakdown with colored bars and replay statistics.

    Parameters
    ----------
    session_events : pd.DataFrame
        All events across session
    replay_metadata : list of dict
        Replay metadata from mario.replays for this session
    subject : str
        Subject ID
    session : str
        Session ID
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    import pandas as pd

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Define event categories
    button_events = ['A', 'B', 'LEFT', 'RIGHT', 'UP', 'DOWN']
    game_events = ['Kill/stomp', 'Kill/kick', 'Kill/impact', 'Hit/life_lost',
                   'Powerup_collected', 'Coin_collected', 'Flag_reached', 'brick_smashes']

    # Filter out gym-retro_game events
    filtered_events = session_events[session_events['trial_type'] != 'gym-retro_game']

    # Event frequencies with colored bars
    event_counts = filtered_events['trial_type'].value_counts().head(15)

    # Assign colors based on event type
    colors = []
    for event in event_counts.index:
        if event in button_events:
            colors.append('#3498db')  # Blue for buttons
        elif event in game_events:
            colors.append('#e74c3c')  # Red for game events
        else:
            colors.append('#e74c3c')  # Red for any other game-related events

    event_counts.plot(kind='barh', ax=ax1, color=colors)
    ax1.set_xlabel('Count', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Event Type', fontsize=13, fontweight='bold')
    ax1.set_title('Top 15 Event Types', fontsize=15, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', label='Player Actions'),
                      Patch(facecolor='#e74c3c', label='Game Events')]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Replay statistics
    if replay_metadata:
        df_replays = pd.DataFrame(replay_metadata)

        # Calculate statistics
        levels_played = len(df_replays)
        cleared = df_replays['Cleared'].sum()
        failed = (~df_replays['Cleared']).sum()
        enemies = df_replays['Enemies_killed'].sum()
        powerups = df_replays['Powerups_collected'].sum()
        coins = df_replays['CoinsGained'].sum()

        # Create bar chart with stacked Cleared/Failed
        categories = ['Levels\nPlayed', 'Enemies\nKilled', 'Powerups', 'Coins']
        x_pos = range(len(categories))

        # Stack Cleared (bottom, green) and Failed (top, red) for first bar
        ax2.bar(0, cleared, color='#27ae60', alpha=0.8, width=0.6, label='Cleared')
        ax2.bar(0, failed, bottom=cleared, color='#e74c3c', alpha=0.8, width=0.6, label='Failed')

        # Other statistics
        other_values = [enemies, powerups, coins]
        other_colors = ['#f39c12', '#3498db', '#f1c40f']
        for i, (value, color) in enumerate(zip(other_values, other_colors), start=1):
            ax2.bar(i, value, color=color, alpha=0.8, width=0.6)

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(categories)
        ax2.set_ylabel('Count', fontsize=13, fontweight='bold')
        ax2.set_title('Session Statistics', fontsize=15, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels
        # Levels Played: show total, cleared, and failed
        ax2.text(0, levels_played, f'{levels_played}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax2.text(0, cleared/2, f'{int(cleared)}',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        ax2.text(0, cleared + failed/2, f'{int(failed)}',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Other stats
        for i, value in enumerate([enemies, powerups, coins], start=1):
            ax2.text(i, value, f'{int(value)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Add legend for the stacked bar
        ax2.legend(loc='upper left', fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No replay metadata available',
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=14, color='gray')
        ax2.set_title('Session Statistics', fontsize=15, fontweight='bold')

    plt.suptitle(f'Session Event Summary - {subject} {session}',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_event_timeline(events_df, run_label, figsize=(16, 8)):
    """
    Plot event timeline for a single run with replay boundaries.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events for one run
    run_label : str
        Run identifier
    run_replays : list of dict, optional
        Replay metadata for this run (level, cleared, duration, etc.)
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Define all event types (buttons + game events)
    button_events = ['A', 'B', 'LEFT', 'RIGHT', 'UP', 'DOWN']
    game_events = ['Kill/stomp', 'Kill/kick', 'Kill/impact', 'Hit/life_lost',
                   'Powerup_collected', 'Coin_collected', 'Flag_reached', 'brick_smashes']
    all_events_list = button_events + game_events

    # Filter out events that don't exist in this run
    all_events_list = [e for e in all_events_list if e in events_df['trial_type'].values]

    # Event timeline with replay backgrounds
    ax1 = axes[0]

    # Draw replay backgrounds using gym-retro_game events
    game_markers = events_df[events_df['trial_type'] == 'gym-retro_game']

    if len(game_markers) > 0:
        colors = plt.cm.Set3(np.linspace(0, 1, len(game_markers)))
        for i, (idx, game_event) in enumerate(game_markers.iterrows()):
            onset = game_event['onset']
            duration = game_event['duration']

            # Draw semi-transparent rectangle for this replay
            ax1.axvspan(onset, onset + duration, alpha=0.15, color=colors[i], zorder=0)

            # Add level text from the event data itself
            # Get the level from the game_event row (which has the correct timing)
            level = game_event.get('level', 'unknown')
            # Position text just outside the top of the plot
            ax1.text(onset + duration/2, len(all_events_list) + 0.3,
                    level, ha='center', va='bottom', fontsize=9,
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
                                                  facecolor=colors[i], alpha=0.3))

    # Plot all events (buttons + game events)
    for idx, event_type in enumerate(all_events_list):
        event_data = events_df[events_df['trial_type'] == event_type]
        if len(event_data) > 0:
            ax1.scatter(event_data['onset'], [idx] * len(event_data),
                       alpha=0.6, s=30, zorder=2)

    ax1.set_ylabel('Events', fontsize=13, fontweight='bold')
    ax1.set_yticks(range(len(all_events_list)))
    ax1.set_yticklabels(all_events_list, fontsize=9)
    ax1.set_ylim(-0.5, len(all_events_list) + 1.2)  # Extend y-axis to show level labels
    ax1.set_title(f'Event Timeline - {run_label}', fontsize=15, fontweight='bold', pad=20)
    ax1.grid(alpha=0.3, zorder=1)

    # Button press density (bottom plot - only buttons, not game events)
    ax2 = axes[1]
    bin_size = 10  # seconds
    max_time = events_df['onset'].max()
    bins = np.arange(0, max_time + bin_size, bin_size)

    button_onsets = events_df[events_df['trial_type'].isin(button_events)]['onset']
    hist, _ = np.histogram(button_onsets, bins=bins)

    ax2.bar(bins[:-1], hist, width=bin_size*0.9, alpha=0.7, color='steelblue')
    ax2.set_xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax2.set_title('Button Presses per 10s', fontsize=15, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    return fig


def plot_confounds_structure(confounds_df, run_label, button_count=None, figsize=(14, 14)):
    """
    Visualize confound structure for a run.

    Parameters
    ----------
    confounds_df : pd.DataFrame
        Confounds dataframe
    run_label : str
        Run identifier
    button_count : np.ndarray, optional
        Button press counts to plot separately
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize)

    # Motion parameters
    motion_cols = [c for c in confounds_df.columns if 'trans' in c or 'rot' in c]
    if len(motion_cols) > 0:
        confounds_df[motion_cols].plot(ax=axes[0], alpha=0.7)
        axes[0].set_title('Motion Parameters (mm and radians)', fontweight='bold')
        axes[0].set_xlabel('TR')
        axes[0].legend(loc='upper right', ncol=3, fontsize=9)
        axes[0].grid(alpha=0.3)

    # Physiology confounds
    physio_cols = [c for c in confounds_df.columns
                   if 'csf' in c or 'white_matter' in c or 'global_signal' in c]
    if len(physio_cols) > 0:
        confounds_df[physio_cols].plot(ax=axes[1], alpha=0.7)
        axes[1].set_title('Physiological Confounds (WM, CSF, Global)', fontweight='bold')
        axes[1].set_xlabel('TR')
        axes[1].legend(loc='upper right', ncol=2, fontsize=9)
        axes[1].grid(alpha=0.3)

    # Low-level features (luminance, optical_flow, audio_envelope)
    lowlevel_cols = [c for c in confounds_df.columns
                     if c in ['luminance', 'optical_flow', 'audio_envelope']]
    if len(lowlevel_cols) > 0:
        confounds_df[lowlevel_cols].plot(ax=axes[2], alpha=0.7)
        axes[2].set_title('Low-level Features (Luminance, Optical Flow, Audio)', fontweight='bold')
        axes[2].set_xlabel('TR')
        axes[2].legend(loc='upper right', ncol=3, fontsize=9)
        axes[2].grid(alpha=0.3)

    # Button presses
    if button_count is not None:
        axes[3].bar(range(len(button_count)), button_count, alpha=0.7, color='steelblue')
        axes[3].set_title('Button Press Counts per TR', fontweight='bold')
        axes[3].set_ylabel('Count')

    axes[3].set_xlabel('TR')
    axes[3].grid(alpha=0.3)

    plt.suptitle(f'Confound Structure - {run_label}', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()

    return fig


def get_design_matrix_figure(fmri_glm, run_label=''):
    """
    Create figure showing design matrix.

    Parameters
    ----------
    fmri_glm : FirstLevelModel
        Fitted GLM model
    run_label : str, optional
        Label for the run (for title)

    Returns
    -------
    matplotlib.figure.Figure
        Design matrix figure
    """
    design_matrix = fmri_glm.design_matrices_[0]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot design matrix
    ax = plotting.plot_design_matrix(design_matrix, ax=ax)

    # Remove grid (it falls between columns which looks weird)
    ax.grid(False)

    if run_label:
        ax.set_title(f'Design Matrix - {run_label}', fontsize=14)
    else:
        ax.set_title('Design Matrix', fontsize=14)

    plt.tight_layout()

    return fig


def create_simple_action_models(events_df):
    """
    Create event dataframes for simple action models (one condition at a time).

    Parameters
    ----------
    events_df : pd.DataFrame
        Full events dataframe

    Returns
    -------
    dict
        Dictionary mapping model names to event dataframes
    """
    actions = ['A', 'B', 'LEFT', 'RIGHT', 'UP', 'DOWN']
    models = {}

    for action in actions:
        if action in events_df['trial_type'].values:
            models[f'simple_{action}'] = create_events_for_glm(events_df, [action])

    return models


def create_movement_model(events_df):
    """
    Create event dataframe for hand/thumb lateralization model.

    LEFT_THUMB: Direction buttons (LEFT, RIGHT, UP, DOWN) - left thumb
    RIGHT_THUMB: Action buttons (A=JUMP, B=RUN) - right thumb

    Parameters
    ----------
    events_df : pd.DataFrame
        Full events dataframe

    Returns
    -------
    pd.DataFrame
        Events for movement model with LEFT_THUMB and RIGHT_THUMB conditions
    """
    # Left thumb: directional pad
    left_thumb_actions = ['LEFT', 'RIGHT', 'UP', 'DOWN']
    # Right thumb: action buttons
    right_thumb_actions = ['A', 'B']

    # Filter events
    left_events = events_df[events_df['trial_type'].isin(left_thumb_actions)].copy()
    right_events = events_df[events_df['trial_type'].isin(right_thumb_actions)].copy()

    # Sanitize and relabel
    left_events['trial_type'] = 'LEFT_THUMB'
    right_events['trial_type'] = 'RIGHT_THUMB'

    # Combine
    combined = pd.concat([left_events, right_events], ignore_index=True)
    combined = combined.sort_values('onset').reset_index(drop=True)

    # Ensure required columns
    required_cols = ['onset', 'duration', 'trial_type']
    return combined[required_cols]


def create_full_actions_model(events_df):
    """
    Create event dataframe for full actions model (all buttons).

    Parameters
    ----------
    events_df : pd.DataFrame
        Full events dataframe

    Returns
    -------
    pd.DataFrame
        Events for full actions model
    """
    actions = ['A', 'B', 'LEFT', 'RIGHT', 'UP', 'DOWN']
    return create_events_for_glm(events_df, actions)


def create_game_events_model(events_df):
    """
    Create event dataframe for game events model.

    Parameters
    ----------
    events_df : pd.DataFrame
        Full events dataframe

    Returns
    -------
    pd.DataFrame
        Events for game events model
    """
    game_events = [
        'Kill/stomp',
        'Kill/kick',
        'Hit/life_lost',
        'Powerup_collected',
        'Coin_collected'
    ]

    # Filter for events that actually exist in this dataframe
    available_events = [e for e in game_events if e in events_df['trial_type'].values]

    if len(available_events) == 0:
        return None

    return create_events_for_glm(events_df, available_events)


def define_movement_contrasts():
    """
    Define contrasts for hand/thumb lateralization model.

    LEFT_THUMB: Direction buttons (LEFT, RIGHT, UP, DOWN) - left hand
    RIGHT_THUMB: Action buttons (A, B) - right hand

    Returns
    -------
    dict
        Contrast definitions
    """
    return {
        'LEFT_THUMB': 'LEFT_THUMB',
        'RIGHT_THUMB': 'RIGHT_THUMB',
        'LEFT_THUMB-RIGHT_THUMB': 'LEFT_THUMB - RIGHT_THUMB',
        'RIGHT_THUMB-LEFT_THUMB': 'RIGHT_THUMB - LEFT_THUMB'
    }


def define_game_event_contrasts():
    """
    Define contrasts for game events model.

    Note: Uses sanitized column names (slashes replaced with underscores).

    Contrasts:
    - Kill: Enemy kills (stomp + kick + impact)
    - Hit_life_lost: Player deaths
    - Kill-Hit: Positive outcome (defeating enemies) vs negative outcome (dying)

    Returns
    -------
    dict
        Contrast definitions
    """
    contrasts = {
        'Kill_stomp': 'Kill_stomp',
        'Kill_kick': 'Kill_kick',
        'Kill_impact': 'Kill_impact',
        'Hit_life_lost': 'Hit_life_lost',
    }

    # Positive outcome (killing enemies) vs negative outcome (dying)
    # Note: We combine all kill types into one regressor
    contrasts['Kill-Hit'] = 'Kill_stomp + Kill_kick + Kill_impact - Hit_life_lost'

    return contrasts


def plot_contrast_surfaces(z_map, contrast_name, stat_threshold=3.0,
                           cluster_threshold=10, figsize=(16, 10)):
    """
    Plot contrast z-map on brain surfaces (4 views: left/right × lateral/medial).

    Parameters
    ----------
    z_map : nibabel.Nifti1Image
        Z-score statistical map from GLM contrast
    contrast_name : str
        Name of the contrast (for title)
    stat_threshold : float, default=3.0
        Z-score threshold for display
    cluster_threshold : int, default=10
        Minimum cluster size in voxels
    figsize : tuple, default=(16, 10)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure with 4 surface plots
    """
    from nilearn.image import threshold_img
    from nilearn import datasets
    from nilearn.surface import vol_to_surf
    from nilearn.plotting import plot_surf_stat_map

    # Threshold the map
    z_thresh = threshold_img(
        z_map,
        threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        two_sided=True
    )

    print(f"✓ Contrast: {contrast_name}")
    print(f"  Threshold: |Z| > {stat_threshold}, cluster > {cluster_threshold} voxels")
    print(f"  (uncorrected for multiple comparisons)")

    # Fetch fsaverage surface
    print("\nFetching fsaverage surface...")
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')

    # Project volume to surface for each hemisphere
    print("Projecting volume to surface...")
    texture_left = vol_to_surf(z_thresh, fsaverage.infl_left)
    texture_right = vol_to_surf(z_thresh, fsaverage.infl_right)

    # Create figure with 4 subplots + space for colorbar
    fig = plt.figure(figsize=figsize)

    # Create GridSpec for better layout control
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05],
                  hspace=0.3, wspace=0.1)

    # Get min/max from both textures for symmetric colormap
    vmin = min(np.nanmin(texture_left), np.nanmin(texture_right))
    vmax = max(np.nanmax(texture_left), np.nanmax(texture_right))
    # Make symmetric around zero for diverging colormap
    abs_max = max(abs(vmin), abs(vmax))

    # Left hemisphere - lateral
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    plot_surf_stat_map(
        fsaverage.infl_left, texture_left, hemi='left', view='lateral',
        threshold=0.1, cmap='cold_hot', colorbar=False,
        bg_map=fsaverage.sulc_left, title='Left Hemisphere - Lateral',
        axes=ax1, vmin=-abs_max, vmax=abs_max
    )

    # Left hemisphere - medial
    ax2 = fig.add_subplot(gs[1, 0], projection='3d')
    plot_surf_stat_map(
        fsaverage.infl_left, texture_left, hemi='left', view='medial',
        threshold=0.1, cmap='cold_hot', colorbar=False,
        bg_map=fsaverage.sulc_left, title='Left Hemisphere - Medial',
        axes=ax2, vmin=-abs_max, vmax=abs_max
    )

    # Right hemisphere - lateral
    ax3 = fig.add_subplot(gs[0, 1], projection='3d')
    plot_surf_stat_map(
        fsaverage.infl_right, texture_right, hemi='right', view='lateral',
        threshold=0.1, cmap='cold_hot', colorbar=False,
        bg_map=fsaverage.sulc_right, title='Right Hemisphere - Lateral',
        axes=ax3, vmin=-abs_max, vmax=abs_max
    )

    # Right hemisphere - medial
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    plot_surf_stat_map(
        fsaverage.infl_right, texture_right, hemi='right', view='medial',
        threshold=0.1, cmap='cold_hot', colorbar=False,
        bg_map=fsaverage.sulc_right, title='Right Hemisphere - Medial',
        axes=ax4, vmin=-abs_max, vmax=abs_max
    )

    # Add single colorbar on the right
    from matplotlib import cm
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=-abs_max, vmax=abs_max)

    # Create colorbar axis
    cbar_ax = fig.add_subplot(gs[:, 2])
    cmap = cm.get_cmap('cold_hot')
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cbar_ax, label='Z-score')

    plt.suptitle(f'{contrast_name}', fontsize=16, fontweight='bold', y=0.98)

    return fig


def plot_contrast_glass_brain(z_map, contrast_name, stat_threshold=3.0,
                              cluster_threshold=10, figsize=(14, 4)):
    """
    Plot contrast z-map as glass brain (simple alternative to surface plots).

    Parameters
    ----------
    z_map : nibabel.Nifti1Image
        Z-score statistical map from GLM contrast
    contrast_name : str
        Name of the contrast (for title)
    stat_threshold : float, default=3.0
        Z-score threshold for display
    cluster_threshold : int, default=10
        Minimum cluster size in voxels
    figsize : tuple, default=(14, 4)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure with glass brain plot
    """
    from nilearn.image import threshold_img
    from nilearn.plotting import plot_glass_brain

    # Threshold the map
    z_thresh = threshold_img(
        z_map,
        threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        two_sided=True
    )

    print(f"✓ Contrast: {contrast_name}")
    print(f"  Threshold: |Z| > {stat_threshold}, cluster > {cluster_threshold} voxels")

    # Create glass brain plot
    fig = plt.figure(figsize=figsize)

    plot_glass_brain(
        z_thresh,
        colorbar=True,
        cmap='cold_hot',
        plot_abs=False,
        display_mode='lyrz',
        title=contrast_name
    )

    plt.tight_layout()

    return fig


def plot_contrast_stat_map(z_map, contrast_name, stat_threshold=3.0,
                           cluster_threshold=10, n_cuts=8, figsize=(14, 6)):
    """
    Plot contrast z-map as axial slices.

    Parameters
    ----------
    z_map : nibabel.Nifti1Image
        Z-score statistical map from GLM contrast
    contrast_name : str
        Name of the contrast (for title)
    stat_threshold : float, default=3.0
        Z-score threshold for display
    cluster_threshold : int, default=10
        Minimum cluster size in voxels
    n_cuts : int, default=8
        Number of axial slices
    figsize : tuple, default=(14, 6)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure with axial slice plot
    """
    from nilearn.image import threshold_img
    from nilearn.plotting import plot_stat_map

    # Threshold the map
    z_thresh = threshold_img(
        z_map,
        threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        two_sided=True
    )

    print(f"✓ Contrast: {contrast_name}")
    print(f"  Threshold: |Z| > {stat_threshold}, cluster > {cluster_threshold} voxels")

    # Create stat map plot
    fig = plt.figure(figsize=figsize)

    plot_stat_map(
        z_thresh,
        threshold=0.1,
        cmap='cold_hot',
        colorbar=True,
        cut_coords=n_cuts,
        display_mode='z',
        title=contrast_name
    )

    plt.tight_layout()

    return fig


# ==============================================================================
# RL VISUALIZATIONS (from rl_viz_utils.py)
# ==============================================================================

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


# ==============================================================================
# ENCODING VISUALIZATIONS (from encoding_viz_utils.py)
# ==============================================================================

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


# ==============================================================================
# ATLAS ENCODING VISUALIZATIONS (from atlas_encoding_utils.py)
# ==============================================================================

def plot_atlas_r2_surfaces(encoding_results, layer_name, atlas, r2_threshold=0.01,
                            figsize=(16, 10)):
    """
    Plot parcel R² values on brain surfaces.
    """
    from nilearn import datasets
    from nilearn.surface import vol_to_surf
    from nilearn.plotting import plot_surf_stat_map
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib import cm
    from matplotlib.colors import Normalize
    import nibabel as nib

    result = encoding_results[layer_name]
    r2_test = result['r2_test']

    print(f"Plotting R² map for layer: {layer_name}")
    print(f"  Mean R²: {r2_test.mean():.4f}")
    print(f"  Max R²: {r2_test.max():.4f}")
    print(f"  Threshold: R² > {r2_threshold}")

    # Load atlas data
    if isinstance(atlas['maps'], str):
        atlas_img = nib.load(atlas['maps'])
    else:
        atlas_img = atlas['maps']

    atlas_data = atlas_img.get_fdata()

    # Create R² volume
    # Initialize with NaN instead of 0 to distinguish "no data" from "zero R²"
    r2_volume = np.full_like(atlas_data, np.nan, dtype=np.float32)

    # Map parcel R² to volume
    # NOTE: BASC labels are integers 1..N
    # Schaefer labels are integers 1..N
    # We assume 'labels' list corresponds to indices 0..N-1 in r2_test
    # and map to atlas values 1..N

    unique_vals = np.unique(atlas_data)
    # Remove 0
    unique_vals = unique_vals[unique_vals != 0]
    unique_vals.sort()

    # Check if number of unique values matches r2_test length
    if len(unique_vals) != len(r2_test):
        print(f"Warning: Atlas has {len(unique_vals)} regions but results have {len(r2_test)} values.")
        print("Mapping by sorted index...")

    # Map ALL R² values to the volume (including negative ones)
    # The threshold will be applied during visualization, not here
    for i, region_id in enumerate(unique_vals):
        if i < len(r2_test):
            r2_val = r2_test[i]
            r2_volume[atlas_data == region_id] = r2_val

    # Create NIfTI image
    r2_img = nib.Nifti1Image(r2_volume, atlas_img.affine)

    # Fetch fsaverage surface
    print("\nFetching fsaverage surface...")
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')

    # Project volume to surface for each hemisphere
    print("Projecting volume to surface...")
    # Strategy: Try multiple radii and interpolations to maximize surface coverage
    # First try linear with large radius
    texture_left = vol_to_surf(
        r2_img,
        fsaverage.infl_left,
        radius=4.0,  # Larger radius for better coverage
        interpolation='linear',
        n_samples=None  # Use default
    )
    texture_right = vol_to_surf(
        r2_img,
        fsaverage.infl_right,
        radius=4.0,
        interpolation='linear',
        n_samples=None
    )

    # For vertices that are still NaN, try nearest neighbor with even larger radius
    nan_mask_left = np.isnan(texture_left)
    nan_mask_right = np.isnan(texture_right)

    if nan_mask_left.any():
        print(f"  Filling {nan_mask_left.sum()} NaN vertices on left hemisphere...")
        texture_left_nearest = vol_to_surf(
            r2_img,
            fsaverage.infl_left,
            radius=6.0,
            interpolation='linear',  # Use linear to avoid deprecation warning
            n_samples=None
        )
        texture_left[nan_mask_left] = texture_left_nearest[nan_mask_left]

    if nan_mask_right.any():
        print(f"  Filling {nan_mask_right.sum()} NaN vertices on right hemisphere...")
        texture_right_nearest = vol_to_surf(
            r2_img,
            fsaverage.infl_right,
            radius=6.0,
            interpolation='linear',  # Use linear to avoid deprecation warning
            n_samples=None
        )
        texture_right[nan_mask_right] = texture_right_nearest[nan_mask_right]

    # Keep NaN values as NaN - plot_surf_stat_map will handle them properly
    # This way, NaN regions will show the sulcal background instead of being colored

    print(f"  Left hemisphere: {np.sum(~np.isnan(texture_left) & (texture_left > r2_threshold))} vertices above threshold")
    print(f"  Right hemisphere: {np.sum(~np.isnan(texture_right) & (texture_right > r2_threshold))} vertices above threshold")

    # Create figure with 4 subplots + colorbar
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05],
                  hspace=0.05, wspace=0.05)

    # Get max R² for colormap
    vmax = max(np.nanmax(texture_left), np.nanmax(texture_right))
    if vmax == 0: vmax = 0.1 # Avoid error if all 0
    vmin = 0

    # Left hemisphere - lateral
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    plot_surf_stat_map(
        fsaverage.infl_left, texture_left, hemi='left', view='lateral',
        threshold=r2_threshold, cmap='hot', colorbar=False,
        bg_map=fsaverage.sulc_left, title='Left Hemisphere - Lateral',
        axes=ax1, vmin=vmin, vmax=vmax
    )
    ax1.dist = 7

    # Left hemisphere - medial
    ax2 = fig.add_subplot(gs[1, 0], projection='3d')
    plot_surf_stat_map(
        fsaverage.infl_left, texture_left, hemi='left', view='medial',
        threshold=r2_threshold, cmap='hot', colorbar=False,
        bg_map=fsaverage.sulc_left, title='Left Hemisphere - Medial',
        axes=ax2, vmin=vmin, vmax=vmax
    )
    ax2.dist = 7

    # Right hemisphere - lateral
    ax3 = fig.add_subplot(gs[0, 1], projection='3d')
    plot_surf_stat_map(
        fsaverage.infl_right, texture_right, hemi='right', view='lateral',
        threshold=r2_threshold, cmap='hot', colorbar=False,
        bg_map=fsaverage.sulc_right, title='Right Hemisphere - Lateral',
        axes=ax3, vmin=vmin, vmax=vmax
    )
    ax3.dist = 7

    # Right hemisphere - medial
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    plot_surf_stat_map(
        fsaverage.infl_right, texture_right, hemi='right', view='medial',
        threshold=r2_threshold, cmap='hot', colorbar=False,
        bg_map=fsaverage.sulc_right, title='Right Hemisphere - Medial',
        axes=ax4, vmin=vmin, vmax=vmax
    )
    ax4.dist = 7

    # Add colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    cbar_ax = fig.add_subplot(gs[:, 2])
    cmap_obj = cm.get_cmap('hot')
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_obj),
                      cax=cbar_ax, label='R²')

    plt.suptitle(f'Encoding Model Performance: {layer_name}',
                 fontsize=16, fontweight='bold', y=0.98)

    return fig


def plot_network_performance(encoding_results, layer_name, atlas, figsize=(12, 6)):
    """
    Plot encoding performance by Yeo network for a single layer.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    result = encoding_results[layer_name]
    r2_test = result['r2_test']
    labels = result['parcel_labels']

    # Handle label array length mismatch
    # Schaefer atlas labels include "Background" at index 0, but r2_test doesn't
    if len(labels) > len(r2_test):
        # Skip first label (Background)
        labels = labels[1:]

    # Ensure lengths match
    if len(labels) != len(r2_test):
        raise ValueError(f"Label and R² array length mismatch: {len(labels)} labels vs {len(r2_test)} R² values")

    # Extract network names (e.g., "Vis" from "7Networks_LH_Vis_1")
    networks = []

    for label in labels:
        if isinstance(label, bytes):
            label = label.decode('utf-8')

        parts = label.split('_')
        if len(parts) >= 3:
            # 7Networks_LH_Vis_1 -> Vis
            # 7Networks_RH_SomMot_3 -> SomMot
            net = parts[2]
            networks.append(net)
        else:
            networks.append('Unknown')

    df = pd.DataFrame({
        'Network': networks,
        'R2': r2_test
    })

    # Plot
    plt.figure(figsize=figsize)

    # Sort by median R2
    order = df.groupby('Network')['R2'].median().sort_values(ascending=False).index

    sns.boxplot(data=df, x='Network', y='R2', order=order, palette='viridis')
    plt.title(f'Performance by Functional Network ({layer_name})', fontsize=14)
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)

    return plt.gcf()


def plot_network_performance_grid(all_encoding_results, pca_dim, atlas,
                                  all_encoding_results_untrained=None,
                                  figsize=(20, 12)):
    """
    Create a grid plot showing network performance for all CNN layers.

    Optionally compare trained vs untrained network performance.

    Parameters
    ----------
    all_encoding_results : dict
        Nested dict: {pca_dim: {layer_name: results}} for trained network
    pca_dim : int
        Which PCA dimension to plot
    atlas : dict
        Atlas dictionary with 'labels'
    all_encoding_results_untrained : dict, optional
        Same structure as all_encoding_results but for untrained network.
        If provided, will show trained vs untrained comparison.
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    encoding_results = all_encoding_results[pca_dim]
    layers = list(encoding_results.keys())
    n_layers = len(layers)

    # Check if we're doing comparison
    compare_mode = all_encoding_results_untrained is not None

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    # Common network order (sorted by overall median R²)
    all_data = []
    for layer_name in layers:
        result = encoding_results[layer_name]
        r2_test = result['r2_test']
        labels = result['parcel_labels']

        # Handle label mismatch
        if len(labels) > len(r2_test):
            labels = labels[1:]

        # Extract networks
        networks = []
        for label in labels:
            if isinstance(label, bytes):
                label = label.decode('utf-8')
            parts = label.split('_')
            if len(parts) >= 3:
                networks.append(parts[2])
            else:
                networks.append('Unknown')

        if compare_mode:
            condition = 'Trained'
            all_data.extend(list(zip(networks, r2_test, [layer_name]*len(r2_test), [condition]*len(r2_test))))
        else:
            all_data.extend(list(zip(networks, r2_test, [layer_name]*len(r2_test))))

    # If comparing, add untrained data
    if compare_mode:
        encoding_results_untrained = all_encoding_results_untrained[pca_dim]
        for layer_name in layers:
            result = encoding_results_untrained[layer_name]
            r2_test = result['r2_test']
            labels = result['parcel_labels']

            if len(labels) > len(r2_test):
                labels = labels[1:]

            networks = []
            for label in labels:
                if isinstance(label, bytes):
                    label = label.decode('utf-8')
                parts = label.split('_')
                if len(parts) >= 3:
                    networks.append(parts[2])
                else:
                    networks.append('Unknown')

            all_data.extend(list(zip(networks, r2_test, [layer_name]*len(r2_test), ['Untrained']*len(r2_test))))

    # Create dataframe
    columns = ['Network', 'R2', 'Layer', 'Condition'] if compare_mode else ['Network', 'R2', 'Layer']
    all_df = pd.DataFrame(all_data, columns=columns)

    # Get network order (by overall median R² for trained network)
    if compare_mode:
        network_order = all_df[all_df['Condition'] == 'Trained'].groupby('Network')['R2'].median().sort_values(ascending=False).index.tolist()
    else:
        network_order = all_df.groupby('Network')['R2'].median().sort_values(ascending=False).index.tolist()

    # Plot each layer
    for idx, layer_name in enumerate(layers):
        ax = axes[idx]

        if compare_mode:
            # Prepare data for this layer (both trained and untrained)
            layer_data = all_df[all_df['Layer'] == layer_name]

            # Plot with hue for trained vs untrained
            sns.boxplot(data=layer_data, x='Network', y='R2', hue='Condition',
                       order=network_order, palette=['#1f77b4', '#ff7f0e'],
                       ax=ax)

            # Get mean R² for trained and untrained
            mean_r2_trained = layer_data[layer_data['Condition'] == 'Trained']['R2'].mean()
            mean_r2_untrained = layer_data[layer_data['Condition'] == 'Untrained']['R2'].mean()
            delta = mean_r2_trained - mean_r2_untrained

            ax.set_title(f'{layer_name.upper()}\nTrained={mean_r2_trained:.4f}, Untrained={mean_r2_untrained:.4f} (Δ={delta:.4f})',
                        fontsize=11, fontweight='bold')

            # Move legend to upper right, smaller
            if idx == 2:  # Only show legend on top-right subplot
                ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            else:
                ax.legend().set_visible(False)
        else:
            # Original single-condition plot
            result = encoding_results[layer_name]
            r2_test = result['r2_test']
            labels = result['parcel_labels']

            # Handle label mismatch
            if len(labels) > len(r2_test):
                labels = labels[1:]

            # Extract networks
            networks = []
            for label in labels:
                if isinstance(label, bytes):
                    label = label.decode('utf-8')
                parts = label.split('_')
                if len(parts) >= 3:
                    networks.append(parts[2])
                else:
                    networks.append('Unknown')

            df = pd.DataFrame({
                'Network': networks,
                'R2': r2_test
            })

            # Plot
            sns.boxplot(data=df, x='Network', y='R2', order=network_order,
                       palette='viridis', ax=ax)

            # Get mean R² for this layer
            mean_r2 = r2_test.mean()

            ax.set_title(f'{layer_name.upper()} (Mean R²={mean_r2:.4f})',
                        fontsize=13, fontweight='bold')

        ax.set_ylabel('R² Score' if idx % 3 == 0 else '')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Set consistent y-axis limits across all plots
        ax.set_ylim([all_df['R2'].min() - 0.01, all_df['R2'].max() + 0.01])

    # Hide extra subplot
    if n_layers < len(axes):
        for idx in range(n_layers, len(axes)):
            axes[idx].set_visible(False)

    if compare_mode:
        title = f'Network Performance: Trained vs Untrained ({pca_dim} PCA components)'
    else:
        title = f'Network Performance Across CNN Layers ({pca_dim} PCA components)'

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    return fig


def plot_glass_brain_r2(encoding_results, layer_name, atlas, r2_threshold=0.01,
                        encoding_results_untrained=None):
    """
    Plot transparent glass brain view of R² results.

    Optionally compare trained vs untrained networks.

    Parameters
    ----------
    encoding_results : dict
        Results for trained network
    layer_name : str
        Which layer to plot
    atlas : dict
        Atlas with 'maps' and 'labels'
    r2_threshold : float
        Display threshold
    encoding_results_untrained : dict, optional
        Results for untrained network

    Returns
    -------
    fig : Figure
    """
    from nilearn import plotting
    import nibabel as nib
    import matplotlib.pyplot as plt

    compare_mode = encoding_results_untrained is not None

    # Helper to create R² volume
    def create_r2_volume(r2_test, atlas_img, atlas_data):
        r2_volume = np.zeros_like(atlas_data)
        unique_vals = np.unique(atlas_data)
        unique_vals = unique_vals[unique_vals != 0]
        unique_vals.sort()

        for i, region_id in enumerate(unique_vals):
            if i < len(r2_test):
                r2_val = r2_test[i]
                if r2_val > r2_threshold:
                    r2_volume[atlas_data == region_id] = r2_val

        return nib.Nifti1Image(r2_volume, atlas_img.affine)

    # Load atlas
    if isinstance(atlas['maps'], str):
        atlas_img = nib.load(atlas['maps'])
    else:
        atlas_img = atlas['maps']
    atlas_data = atlas_img.get_fdata()

    if compare_mode:
        # Comparison mode: trained vs untrained
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))

        # Trained
        result_trained = encoding_results[layer_name]
        r2_img_trained = create_r2_volume(result_trained['r2_test'], atlas_img, atlas_data)

        plotting.plot_glass_brain(
            r2_img_trained,
            display_mode='lyrz',
            colorbar=True,
            threshold=r2_threshold,
            cmap='hot',
            plot_abs=False,
            title=f'Trained - {layer_name.upper()} (Mean R²={result_trained["r2_test"].mean():.4f})',
            axes=axes[0]
        )

        # Untrained
        result_untrained = encoding_results_untrained[layer_name]
        r2_img_untrained = create_r2_volume(result_untrained['r2_test'], atlas_img, atlas_data)

        plotting.plot_glass_brain(
            r2_img_untrained,
            display_mode='lyrz',
            colorbar=True,
            threshold=r2_threshold,
            cmap='hot',
            plot_abs=False,
            title=f'Untrained - {layer_name.upper()} (Mean R²={result_untrained["r2_test"].mean():.4f})',
            axes=axes[1]
        )

        fig.suptitle(f'Glass Brain Comparison: Trained vs Untrained',
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

    else:
        # Single network mode
        result = encoding_results[layer_name]
        r2_img = create_r2_volume(result['r2_test'], atlas_img, atlas_data)

        fig = plt.figure(figsize=(10, 4))
        plotting.plot_glass_brain(
            r2_img,
            display_mode='lyrz',
            colorbar=True,
            threshold=r2_threshold,
            cmap='hot',
            plot_abs=False,
            title=f'Glass Brain View: {layer_name}',
            figure=fig
        )

    return fig


def visualize_best_parcel_prediction(layer_activations, parcel_bold, atlas, 
                                     train_indices, test_indices, layer_name, 
                                     encoding_results, alphas=None):
    """
    Visualize time series prediction for the best parcel.
    """
    import matplotlib.pyplot as plt
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    
    if alphas is None:
        alphas = [0.1, 1, 10, 100, 1000, 10000, 100000]

    # Find best parcel
    result = encoding_results[layer_name]
    best_idx = np.argmax(result['r2_test'])
    best_r2 = result['r2_test'][best_idx]
    label = result['parcel_labels'][best_idx]
    
    if isinstance(label, bytes):
        label = label.decode('utf-8')
    
    print(f"Refitting best parcel: {label} (R² = {best_r2:.4f})")
    
    # Get data
    X = layer_activations[layer_name]
    y = parcel_bold[:, best_idx]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Refit
    ridge = RidgeCV(alphas=alphas, cv=3)
    ridge.fit(X_train_scaled, y_train)
    y_pred = ridge.predict(X_test_scaled)
    
    # Plot
    plt.figure(figsize=(14, 4))
    plt.plot(y_test, label='Actual BOLD', alpha=0.7)
    plt.plot(y_pred, label='Predicted BOLD', alpha=0.8, color='orange')
    plt.title(f'Actual vs Predicted Signal: {label}\nTest Set (n={len(y_test)}), R²={best_r2:.3f}')
    plt.xlabel('Time (TRs)')
    plt.ylabel('BOLD Signal (z-score)')
    plt.legend()
    plt.grid(alpha=0.3)

    return plt.gcf()


# ==============================================================================
# RL AGENT VISUALIZATIONS
# ==============================================================================

def show_reward_function_examples():
    """
    Print concrete examples of the reward function with different scenarios.

    Shows how different gameplay events translate to rewards, highlighting
    the asymmetric design that promotes cautious play.
    """
    print("Reward Function Examples:\n")
    print("=" * 70)

    # Example scenarios
    scenarios = [
        {
            'name': 'Normal forward movement',
            'x_diff': 3,
            'score_diff': 0,
            'lives_lost': False,
            'time_diff': 0
        },
        {
            'name': 'Collecting a coin',
            'x_diff': 2,
            'score_diff': 200,  # Coin value
            'lives_lost': False,
            'time_diff': 0
        },
        {
            'name': 'Defeating an enemy',
            'x_diff': 1,
            'score_diff': 100,  # Goomba stomp
            'lives_lost': False,
            'time_diff': 0
        },
        {
            'name': 'LOSING A LIFE (hit by enemy)',
            'x_diff': 0,
            'score_diff': 0,
            'lives_lost': True,
            'time_diff': 0
        },
        {
            'name': 'Risky play: coin + enemy hit',
            'x_diff': 2,
            'score_diff': 200,
            'lives_lost': True,
            'time_diff': 0
        }
    ]

    for scenario in scenarios:
        reward = 0.0

        # Movement
        if -5 <= scenario['x_diff'] <= 5:
            reward += scenario['x_diff']

        # Score (increased weight from /4.0 to /2.0)
        reward += min(scenario['score_diff'] / 2.0, 50)

        # Life loss
        if scenario['lives_lost']:
            reward -= 50

        # Time
        reward += scenario['time_diff']

        # Clip
        reward = max(min(reward, 15), -50)

        print(f"{scenario['name']:35s} → Reward: {reward:+6.1f}")
        if scenario['lives_lost']:
            print("  ⚠️  HEAVY PENALTY! Agent learns to avoid this!")

    print("=" * 70)
    print("\n💡 Key insight: The -50 life penalty dominates all positive rewards")
    print("   This teaches the agent that survival > score gains")


def plot_training_progress(training_log):
    """
    Plot agent training progress over time.

    Parameters
    ----------
    training_log : dict
        Training log dictionary with 'config' and 'progress' keys

    Returns
    -------
    matplotlib.figure.Figure
        Figure with training curve
    """
    print("Training Progress:\n")
    print(f"Configuration:")
    for key, val in training_log['config'].items():
        if key != 'levels':
            print(f"  {key}: {val}")
    print(f"  levels: {', '.join(training_log['config']['levels'])}")

    # Create plot if progress data exists
    if len(training_log['progress']) > 0:
        progress = training_log['progress']
        steps = [p['step'] for p in progress]
        mean_rewards = [p['mean_reward'] for p in progress]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, mean_rewards, linewidth=2)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Mean Reward (last 100 episodes)')
        ax.set_title('PPO Training Progress')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        print(f"\n✓ Agent trained for {steps[-1]:,} steps")
        return fig
    else:
        print("  No progress data available")
        return None


def plot_layer_performance_comparison(all_comparisons, pca_dims, figsize=(18, 5)):
    """
    Plot layer performance comparison across multiple PCA dimensions.

    Parameters
    ----------
    all_comparisons : dict
        Dictionary mapping PCA dimensions to comparison DataFrames
    pca_dims : list
        List of PCA dimensions tested
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, len(pca_dims), figsize=figsize)
    layer_order = ['conv1', 'conv2', 'conv3', 'conv4', 'linear']

    for idx, n_comp in enumerate(pca_dims):
        ax = axes[idx] if len(pca_dims) > 1 else axes
        comparison_df = all_comparisons[n_comp].copy()

        # Reorder by layer_order
        comparison_df['layer'] = pd.Categorical(
            comparison_df['layer'],
            categories=layer_order,
            ordered=True
        )
        comparison_df = comparison_df.sort_values('layer')

        # Bar plot
        bars = ax.bar(
            comparison_df['layer'],
            comparison_df['mean_r2'],
            color='steelblue',
            alpha=0.8,
            edgecolor='black'
        )

        # Highlight best layer
        best_idx = comparison_df['mean_r2'].argmax()
        bars[best_idx].set_color('darkorange')

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Mean R²', fontsize=12)
        ax.set_title(f'{n_comp} PCA Components', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, max(0.05, comparison_df['mean_r2'].max() * 1.2)])

    plt.tight_layout()
    return fig


def plot_r2_distribution_and_top_parcels(r2_test, best_layer, top_parcels,
                                          n_top=10, figsize=(14, 5)):
    """
    Plot R² distribution and top performing parcels.

    Parameters
    ----------
    r2_test : np.ndarray
        R² values for test set
    best_layer : str
        Name of the best performing layer
    top_parcels : pd.DataFrame
        DataFrame with top parcels (from get_top_parcels)
    n_top : int
        Number of top parcels to display
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # R² distribution histogram
    axes[0].hist(
        r2_test[r2_test > 0],
        bins=50,
        color='steelblue',
        alpha=0.7,
        edgecolor='black'
    )
    axes[0].axvline(
        np.mean(r2_test),
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Mean = {np.mean(r2_test):.4f}'
    )
    axes[0].axvline(
        np.median(r2_test),
        color='orange',
        linestyle='--',
        linewidth=2,
        label=f'Median = {np.median(r2_test):.4f}'
    )
    axes[0].set_xlabel('R²')
    axes[0].set_ylabel('Number of parcels')
    axes[0].set_title(f'{best_layer} - R² Distribution', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Top parcels bar plot
    top_n_for_plot = top_parcels.head(n_top)
    axes[1].barh(
        range(len(top_n_for_plot)),
        top_n_for_plot['r2'][::-1],
        color='steelblue',
        alpha=0.8,
        edgecolor='black'
    )
    axes[1].set_yticks(range(len(top_n_for_plot)))
    axes[1].set_yticklabels(top_n_for_plot['label'].tolist()[::-1], fontsize=9)
    axes[1].set_xlabel('R²')
    axes[1].set_title(f'Top {n_top} Parcels', fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig
