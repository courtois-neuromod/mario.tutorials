"""
Mario fMRI Tutorial - Source Code Package

This package contains analysis modules for the Mario fMRI tutorial.

Modules:
    utils: General utilities (data loading, path management, environment setup)
    glm: GLM analysis functions (confounds, design matrices, fitting)
    parcellation: Parcellation and atlas utilities
    rl: Reinforcement learning utilities (CNN model, activations, HRF, PCA)
    encoding: Brain encoding models (ridge regression, evaluation)
    visualization: All visualization functions (GLM, RL, encoding)
"""

# Core utilities
# Backward compatibility aliases (deprecated)
import warnings

# Encoding models
from .encoding import (
    RidgeEncodingModel,
    compare_atlas_layer_performance,
    compare_layer_performance,
    compute_noise_ceiling,
    create_encoding_summary_figure,
    cross_validated_encoding,
    fit_atlas_encoding_per_layer,
    fit_encoding_model_per_layer,
    get_top_parcels,
    load_and_prepare_bold,
)

# GLM analysis
from .glm import (
    add_button_press_counts,
    add_scrub_regressors,
    aggregate_runs_fixed_effects,
    compute_contrasts,
    create_events_for_glm,
    create_full_actions_model,
    create_game_events_model,
    create_movement_model,
    create_simple_action_models,
    define_game_event_contrasts,
    define_movement_contrasts,
    downsample_lowlevel_confounds,
    fit_run_glm,
    prepare_confounds,
    sanitize_trial_type,
    threshold_map_clusters,
)

# Parcellation and atlases
from .parcellation import (
    create_kmeans_parcellation,
    create_random_parcellation,
    create_ward_parcellation,
    extract_parcel_bold,
    extract_parcel_bold_from_parcellation,
    get_parcel_labels,
    load_basc_atlas,
    load_parcellation,
    load_schaefer_atlas,
    save_parcellation,
)

# Reinforcement learning
from .rl import (
    SimpleCNN,
    action_to_buttons,
    align_activations_to_bold,
    apply_pca,
    apply_pca_with_nan_handling,
    convolve_with_hrf,
    create_frame_stack,
    create_simple_proxy_features,
    downsample_activations_to_tr,
    extract_activations_from_frames,
    extract_layer_activations,
    load_pretrained_model,
    play_agent_episode,
    preprocess_frame,
)
from .utils import (
    compute_dataset_statistics,
    # Output utilities
    create_output_dir,
    download_stimuli,
    get_bold_path,
    get_derivatives_path,
    get_project_root,
    get_session_runs,
    # Path management
    get_sourcedata_path,
    install_dependencies,
    load_bold,
    load_brain_mask,
    load_confounds,
    # Data loading
    load_events,
    load_lowlevel_confounds,
    load_replay_metadata,
    save_stat_map,
    # Setup functions
    setup_colab_environment,
    setup_datalad_datasets,
    verify_data,
)

# Visualizations
from .visualization import (
    get_design_matrix_figure,
    # Color helpers
    level_color,
    plot_agent_gameplay,
    plot_confounds_structure,
    plot_contrast_glass_brain,
    plot_contrast_stat_map,
    plot_contrast_surfaces,
    plot_encoding_comparison_table,
    # GLM viz
    plot_event_frequencies,
    plot_event_timeline,
    plot_glass_brain_r2,
    plot_layer_activations_sample,
    plot_layer_comparison_bars,
    # Atlas encoding viz
    plot_network_performance_grid,
    # RL viz
    plot_pca_variance_per_layer,
    plot_prediction_examples,
    # Encoding viz
    plot_r2_brainmap,
    visualize_best_parcel_prediction,
)


class DeprecatedModule:
    """Wrapper to show deprecation warnings for old module names."""
    def __init__(self, new_name, old_name):
        self.new_name = new_name
        self.old_name = old_name
        self._module = None

    def __getattr__(self, name):
        if self._module is None:
            warnings.warn(
                f"Importing from '{self.old_name}' is deprecated. "
                f"Use 'from {self.new_name} import ...' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            # Import the actual module
            import importlib
            self._module = importlib.import_module(f'.{self.new_name}', package='src')
        return getattr(self._module, name)

# Create deprecated aliases
import sys

sys.modules['src.glm_utils'] = DeprecatedModule('glm', 'glm_utils')
sys.modules['src.rl_utils'] = DeprecatedModule('rl', 'rl_utils')
sys.modules['src.encoding_utils'] = DeprecatedModule('encoding', 'encoding_utils')

__version__ = '0.2.0'

__all__ = [
    # Utils
    'get_sourcedata_path', 'get_derivatives_path', 'get_project_root',
    'load_events', 'get_bold_path', 'load_bold', 'load_brain_mask',
    'load_confounds', 'get_session_runs', 'load_lowlevel_confounds',
    'load_replay_metadata', 'create_output_dir', 'save_stat_map',
    'compute_dataset_statistics', 'setup_colab_environment', 'install_dependencies',
    'setup_datalad_datasets', 'download_stimuli', 'verify_data',
    # GLM
    'prepare_confounds', 'add_button_press_counts', 'add_scrub_regressors',
    'downsample_lowlevel_confounds', 'sanitize_trial_type', 'create_events_for_glm',
    'create_movement_model', 'create_game_events_model', 'create_full_actions_model',
    'create_simple_action_models', 'define_movement_contrasts',
    'define_game_event_contrasts', 'fit_run_glm', 'compute_contrasts',
    'aggregate_runs_fixed_effects', 'threshold_map_clusters',
    # Parcellation
    'create_random_parcellation', 'create_kmeans_parcellation', 'create_ward_parcellation',
    'save_parcellation', 'load_parcellation', 'extract_parcel_bold_from_parcellation',
    'get_parcel_labels', 'load_schaefer_atlas', 'load_basc_atlas', 'extract_parcel_bold',
    # RL
    'SimpleCNN', 'load_pretrained_model', 'preprocess_frame', 'create_frame_stack',
    'extract_activations_from_frames', 'downsample_activations_to_tr', 'convolve_with_hrf',
    'apply_pca', 'apply_pca_with_nan_handling', 'create_simple_proxy_features',
    'action_to_buttons', 'play_agent_episode', 'extract_layer_activations',
    'align_activations_to_bold',
    # Encoding
    'RidgeEncodingModel', 'load_and_prepare_bold', 'fit_encoding_model_per_layer',
    'cross_validated_encoding', 'compare_layer_performance', 'compute_noise_ceiling',
    'fit_atlas_encoding_per_layer', 'compare_atlas_layer_performance', 'get_top_parcels',
    'create_encoding_summary_figure',
    # Visualization
    'plot_event_frequencies', 'plot_event_timeline', 'plot_confounds_structure',
    'get_design_matrix_figure', 'plot_contrast_surfaces', 'plot_contrast_glass_brain',
    'plot_contrast_stat_map', 'plot_pca_variance_per_layer', 'plot_layer_activations_sample',
    'plot_agent_gameplay', 'plot_r2_brainmap', 'plot_encoding_comparison_table',
    'plot_prediction_examples', 'plot_layer_comparison_bars',
    'plot_network_performance_grid', 'plot_glass_brain_r2', 'visualize_best_parcel_prediction',
    'level_color',
]
