"""
Mario fMRI Tutorial - Source Code Package

This package contains analysis modules for the Mario fMRI tutorial.

Modules:
    utils: General utilities for data loading and path management
    glm_utils: GLM analysis functions (confounds, design matrices, fitting)
    encoding_utils: Brain encoding model utilities (ridge regression, evaluation)
    rl_utils: Reinforcement learning utilities (activations, HRF, PCA)
"""

from .utils import (
    get_sourcedata_path,
    get_derivatives_path,
    load_events,
    get_bold_path,
    load_bold,
    load_brain_mask,
    load_confounds,
    get_session_runs,
    create_output_dir,
    save_stat_map,
    load_lowlevel_confounds
)

from .glm_utils import (
    prepare_confounds,
    add_button_press_counts,
    downsample_lowlevel_confounds,
    create_movement_model,
    create_game_events_model,
    define_movement_contrasts,
    define_game_event_contrasts,
    fit_run_glm,
    compute_contrasts,
    aggregate_runs_fixed_effects,
    threshold_map_clusters,
    get_design_matrix_figure
)

from .encoding_utils import (
    RidgeEncodingModel,
    load_and_prepare_bold,
    fit_encoding_model_per_layer,
    compare_layer_performance,
    create_encoding_summary_figure
)

from .rl_utils import (
    SimpleCNN,
    load_pretrained_model,
    create_simple_proxy_features,
    downsample_activations_to_tr,
    convolve_with_hrf,
    apply_pca
)

__version__ = '0.1.0'
__all__ = [
    # utils
    'get_sourcedata_path',
    'get_derivatives_path',
    'load_events',
    'get_bold_path',
    'load_bold',
    'load_brain_mask',
    'load_confounds',
    'get_session_runs',
    'create_output_dir',
    'save_stat_map',
    'load_lowlevel_confounds',
    # glm_utils
    'prepare_confounds',
    'add_button_press_counts',
    'downsample_lowlevel_confounds',
    'create_movement_model',
    'create_game_events_model',
    'define_movement_contrasts',
    'define_game_event_contrasts',
    'fit_run_glm',
    'compute_contrasts',
    'aggregate_runs_fixed_effects',
    'threshold_map_clusters',
    'get_design_matrix_figure',
    # encoding_utils
    'RidgeEncodingModel',
    'load_and_prepare_bold',
    'fit_encoding_model_per_layer',
    'compare_layer_performance',
    'create_encoding_summary_figure',
    # rl_utils
    'SimpleCNN',
    'load_pretrained_model',
    'create_simple_proxy_features',
    'downsample_activations_to_tr',
    'convolve_with_hrf',
    'apply_pca',
]
