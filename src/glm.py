"""
GLM analysis utilities for Mario fMRI tutorial.
Adapted from shinobi_fmri methodology.

This module contains only analysis functions.
Visualization functions are in visualization.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm import compute_fixed_effects


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


# ==============================================================================
# MODEL CREATION FUNCTIONS
# ==============================================================================

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
