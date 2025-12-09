"""
GLM analysis utilities for Mario fMRI tutorial.
Adapted from shinobi_fmri methodology.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm import compute_fixed_effects
from nilearn import plotting
import matplotlib.pyplot as plt


def downsample_lowlevel_confounds(lowlevel_df, n_scans, tr, game_fps=60):
    """
    Downsample low-level confounds from game framerate to fMRI TR.

    Parameters
    ----------
    lowlevel_df : pd.DataFrame
        Low-level confounds at game framerate (luminance, optical_flow, audio_envelope)
    n_scans : int
        Number of fMRI scans/TRs
    tr : float
        Repetition time in seconds
    game_fps : int, default=60
        Game frames per second

    Returns
    -------
    pd.DataFrame
        Downsampled confounds matching fMRI timepoints
    """
    from scipy.ndimage import uniform_filter1d

    # Calculate number of game frames per TR
    frames_per_tr = int(game_fps * tr)

    downsampled = {}

    for col in lowlevel_df.columns:
        signal = lowlevel_df[col].values

        # Downsample by averaging over each TR window
        downsampled_signal = []
        for i in range(n_scans):
            start_idx = i * frames_per_tr
            end_idx = min((i + 1) * frames_per_tr, len(signal))

            if start_idx < len(signal):
                # Average over this TR's frames
                downsampled_signal.append(np.mean(signal[start_idx:end_idx]))
            else:
                # If we've run out of game frames, use the last value
                downsampled_signal.append(signal[-1])

        downsampled[col] = np.array(downsampled_signal)

    return pd.DataFrame(downsampled)


def prepare_confounds(confounds_df, strategy='full', lowlevel_confounds=None):
    """
    Prepare confounds for GLM analysis.

    Parameters
    ----------
    confounds_df : pd.DataFrame
        Raw confounds from fMRIPrep
    strategy : str, default='full'
        Confound strategy:
        - 'minimal': 6 motion parameters
        - 'basic': Motion + WM + CSF
        - 'full': Motion (24 params) + WM + CSF + global signal
    lowlevel_confounds : pd.DataFrame, optional
        Low-level confounds (luminance, optical_flow, audio_envelope)
        If provided, these will be added to the confounds

    Returns
    -------
    pd.DataFrame
        Selected and processed confounds
    """
    confounds_list = []

    # Motion parameters (6 basic)
    motion_params = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

    if strategy == 'minimal':
        confounds_list.extend(motion_params)

    elif strategy in ['basic', 'full']:
        # 24 motion parameters (original + derivatives + quadratic)
        confounds_list.extend(motion_params)

        # Derivatives
        for param in motion_params:
            deriv_col = f'{param}_derivative1'
            if deriv_col in confounds_df.columns:
                confounds_list.append(deriv_col)

        # Quadratic terms
        for param in motion_params:
            quad_col = f'{param}_power2'
            if quad_col in confounds_df.columns:
                confounds_list.append(quad_col)

        # Quadratic derivatives
        for param in motion_params:
            quad_deriv_col = f'{param}_derivative1_power2'
            if quad_deriv_col in confounds_df.columns:
                confounds_list.append(quad_deriv_col)

        # White matter and CSF
        if 'white_matter' in confounds_df.columns:
            confounds_list.append('white_matter')
        if 'csf' in confounds_df.columns:
            confounds_list.append('csf')

    if strategy == 'full':
        # Global signal
        if 'global_signal' in confounds_df.columns:
            confounds_list.append('global_signal')

    # Select only existing columns
    available_confounds = [c for c in confounds_list if c in confounds_df.columns]

    if len(available_confounds) == 0:
        raise ValueError("No confounds found in dataframe")

    confounds = confounds_df[available_confounds].copy()

    # Fill NaN values (typically first row for derivatives)
    confounds.fillna(0, inplace=True)

    # Add low-level confounds if provided
    if lowlevel_confounds is not None:
        # Ensure same number of rows
        if len(lowlevel_confounds) != len(confounds):
            raise ValueError(f"Low-level confounds ({len(lowlevel_confounds)} rows) must match "
                           f"fMRIPrep confounds ({len(confounds)} rows)")

        # Add each low-level confound column
        for col in lowlevel_confounds.columns:
            confounds[col] = lowlevel_confounds[col].values

    # Standardize confounds (z-score) for better numerical stability
    # This is important for physiological confounds (WM, CSF, GS) which have
    # large absolute values but small relative variance
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    confounds_scaled = pd.DataFrame(
        scaler.fit_transform(confounds),
        columns=confounds.columns,
        index=confounds.index
    )

    return confounds_scaled


def add_button_press_counts(confounds_df, events_df, tr, n_scans):
    """
    Add button press counts as psychophysics confounds.

    Parameters
    ----------
    confounds_df : pd.DataFrame
        Existing confounds
    events_df : pd.DataFrame
        Events dataframe with trial_type, onset, duration
    tr : float
        Repetition time in seconds
    n_scans : int
        Number of volumes

    Returns
    -------
    pd.DataFrame
        Confounds with added button press counts
    """
    # Initialize button press count array
    button_presses = np.zeros(n_scans)

    # Count button presses in each TR
    button_types = ['A', 'B', 'LEFT', 'RIGHT', 'UP', 'DOWN']
    button_events = events_df[events_df['trial_type'].isin(button_types)]

    for _, event in button_events.iterrows():
        onset = event['onset']
        # Find corresponding TR
        tr_idx = int(onset / tr)
        if tr_idx < n_scans:
            button_presses[tr_idx] += 1

    # Add to confounds
    confounds_df = confounds_df.copy()
    confounds_df['button_press_count'] = button_presses

    return confounds_df


def sanitize_trial_type(trial_type):
    """
    Sanitize trial type name to be a valid Python identifier.

    Replaces slashes and other invalid characters with underscores.

    Parameters
    ----------
    trial_type : str
        Original trial type name

    Returns
    -------
    str
        Sanitized trial type name
    """
    return trial_type.replace('/', '_').replace('-', '_').replace(' ', '_')


def create_events_for_glm(events_df, conditions, tr=None):
    """
    Create events dataframe for specific conditions.

    Parameters
    ----------
    events_df : pd.DataFrame
        Full events dataframe
    conditions : list of str
        Trial types to include
    tr : float, optional
        If provided, filter out events shorter than TR

    Returns
    -------
    pd.DataFrame
        Filtered events with onset, duration, trial_type columns
    """
    # Filter for specified conditions
    glm_events = events_df[events_df['trial_type'].isin(conditions)].copy()

    # Sanitize trial_type names for GLM compatibility
    glm_events['trial_type'] = glm_events['trial_type'].apply(sanitize_trial_type)

    # Ensure required columns
    required_cols = ['onset', 'duration', 'trial_type']
    if not all(col in glm_events.columns for col in required_cols):
        raise ValueError(f"Events must have columns: {required_cols}")

    # Filter by duration if TR provided
    if tr is not None:
        glm_events = glm_events[glm_events['duration'] >= tr/2].copy()

    # Keep only required columns
    glm_events = glm_events[required_cols].reset_index(drop=True)

    return glm_events


def fit_run_glm(bold_img, events, confounds, tr=1.49,
                hrf_model='spm', noise_model='ar1',
                smoothing_fwhm=None, mask_img=None,
                high_pass=1/128, drift_model='cosine'):
    """
    Fit first-level GLM to a single run.

    Parameters
    ----------
    bold_img : nibabel.Nifti1Image
        BOLD data
    events : pd.DataFrame
        Events with onset, duration, trial_type
    confounds : pd.DataFrame
        Confound regressors
    tr : float, default=1.49
        Repetition time in seconds
    hrf_model : str, default='spm'
        HRF model to use
    noise_model : str, default='ar1'
        Noise model ('ar1', 'ols')
    smoothing_fwhm : float, optional
        FWHM for spatial smoothing in mm
    mask_img : nibabel.Nifti1Image, optional
        Brain mask
    high_pass : float, default=1/128
        High-pass filter cutoff in Hz
    drift_model : str, default='cosine'
        Drift model type

    Returns
    -------
    FirstLevelModel
        Fitted GLM model
    """
    # Create model
    fmri_glm = FirstLevelModel(
        t_r=tr,
        hrf_model=hrf_model,
        noise_model=noise_model,
        smoothing_fwhm=smoothing_fwhm,
        mask_img=mask_img,
        high_pass=high_pass,
        drift_model=drift_model,
        standardize=False,
        minimize_memory=True
    )

    # Fit model
    fmri_glm.fit(bold_img, events=events, confounds=confounds)

    return fmri_glm


def threshold_map_clusters(stat_map, stat_threshold=3.0, cluster_threshold=10, two_sided=True):
    """
    Apply cluster-level inference to threshold statistical maps.

    Uses nilearn's thresholding to identify significant clusters and returns
    a thresholded map showing only robust clusters.

    Parameters
    ----------
    stat_map : Nifti1Image
        Statistical map (z-score or t-score)
    stat_threshold : float, default=3.0
        Cluster-forming threshold (z-score)
    cluster_threshold : int, default=10
        Minimum cluster size in voxels (for manual thresholding)
    two_sided : bool, default=True
        Whether to threshold both positive and negative clusters

    Returns
    -------
    Nifti1Image
        Thresholded statistical map with only significant clusters
    """
    from nilearn.image import threshold_img

    # Use threshold_img with cluster size parameter
    thresholded_map = threshold_img(
        stat_map,
        threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided
    )

    return thresholded_map


def compute_contrasts(fmri_glm, contrast_defs):
    """
    Compute contrasts from fitted GLM.

    Parameters
    ----------
    fmri_glm : FirstLevelModel
        Fitted GLM model
    contrast_defs : dict
        Dictionary mapping contrast names to contrast vectors/strings
        E.g., {'LEFT': 'LEFT', 'LEFT-RIGHT': 'LEFT - RIGHT'}

    Returns
    -------
    dict
        Dictionary mapping contrast names to stat maps
    """
    contrast_maps = {}

    for contrast_name, contrast_spec in contrast_defs.items():
        try:
            # Compute contrast
            z_map = fmri_glm.compute_contrast(
                contrast_spec,
                stat_type='t',
                output_type='z_score'
            )
            contrast_maps[contrast_name] = z_map
        except ValueError as e:
            print(f"Warning: Could not compute contrast '{contrast_name}': {e}")

    return contrast_maps


def aggregate_runs_fixed_effects(contrast_maps_list, variance_maps_list):
    """
    Aggregate multiple runs using fixed-effects model.

    Parameters
    ----------
    contrast_maps_list : list of nibabel.Nifti1Image
        Contrast maps from each run
    variance_maps_list : list of nibabel.Nifti1Image
        Variance maps from each run

    Returns
    -------
    tuple
        (fixed_effects_contrast, fixed_effects_variance, fixed_effects_stat)
    """
    if len(contrast_maps_list) == 0:
        raise ValueError("No contrast maps provided")

    if len(contrast_maps_list) == 1:
        # Single run, return as-is with None stat map
        return contrast_maps_list[0], variance_maps_list[0], None

    # Use nilearn's fixed effects
    fixed_fx = compute_fixed_effects(
        contrast_maps_list,
        variance_maps_list,
        precision_weighted=True
    )

    return fixed_fx


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
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)

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
        ax2.legend(loc='upper right', fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No replay metadata available',
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=14, color='gray')
        ax2.set_title('Session Statistics', fontsize=15, fontweight='bold')

    plt.suptitle(f'Session Event Summary - {subject} {session}',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_event_timeline(events_df, run_label, button_events_list, figsize=(16, 8)):
    """
    Plot event timeline for a single run.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events for one run
    run_label : str
        Run identifier
    button_events_list : list of str
        List of button event types
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Button timeline
    ax1 = axes[0]
    for idx, button in enumerate(button_events_list):
        button_data = events_df[events_df['trial_type'] == button]
        if len(button_data) > 0:
            ax1.scatter(button_data['onset'], [idx] * len(button_data),
                       label=button, alpha=0.6, s=30)

    ax1.set_ylabel('Button', fontsize=13, fontweight='bold')
    ax1.set_yticks(range(len(button_events_list)))
    ax1.set_yticklabels(button_events_list)
    ax1.set_title(f'Button Press Timeline - {run_label}', fontsize=15, fontweight='bold')
    ax1.legend(loc='upper right', ncol=6)
    ax1.grid(alpha=0.3)

    # Event density
    ax2 = axes[1]
    bin_size = 10  # seconds
    max_time = events_df['onset'].max()
    bins = np.arange(0, max_time + bin_size, bin_size)

    button_onsets = events_df[events_df['trial_type'].isin(button_events_list)]['onset']
    hist, _ = np.histogram(button_onsets, bins=bins)

    ax2.bar(bins[:-1], hist, width=bin_size*0.9, alpha=0.7, color='steelblue')
    ax2.set_xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Events per 10s', fontsize=13, fontweight='bold')
    ax2.set_title('Event Density', fontsize=15, fontweight='bold')
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
