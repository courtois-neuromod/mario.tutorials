# Mario fMRI Tutorial - Source Code Modules

This directory contains the core analysis modules for the Mario fMRI tutorial.

**Version:** 0.2.0 (Reorganized module structure - January 2025)

## Module Overview

The codebase is organized into 6 core modules plus package initialization:

### `utils.py` - General Utilities & Setup
Core functions for data loading, path management, and environment setup.

**Setup functions (new in v0.2.0):**
- `setup_environment()` - Detect Colab/local environment, clone repo, setup paths
- `install_dependencies(requirements_file)` - Install packages from requirements file
- `setup_datalad_datasets(sourcedata_path)` - Install datasets with DataLad + fallback
- `download_stimuli(sourcedata_path)` - Download stimuli from Google Drive
- `verify_data(subject, session, sourcedata_path)` - Verify data availability

**Data loading functions:**
- `get_sourcedata_path()` - Locate source data directory
- `get_derivatives_path()` - Locate derivatives directory
- `load_events()` - Load behavioral annotations (onset, duration, trial_type)
- `load_bold()` - Load preprocessed fMRI data
- `load_brain_mask()` - Load brain mask
- `load_lowlevel_confounds()` - Load low-level confounds (luminance, optical flow, audio)
- `get_session_runs()` - Get list of runs for a session
- `get_bold_path()` - Get path to BOLD file

**File operations:**
- `create_output_dir()` - Create derivatives output directory
- `save_stat_map()` - Save statistical maps with BIDS-like naming

### `glm.py` - GLM Analysis
Functions for General Linear Model analysis following shinobi_fmri methodology.

**Confound processing:**
- `downsample_lowlevel_confounds()` - Downsample low-level confounds from 60Hz to TR
- `prepare_confounds()` - Extract and prepare confounds (motion, WM, CSF, global signal)
- `add_button_press_counts()` - Add button press counts as task confound

**Event model creation:**
- `sanitize_trial_type()` - Clean trial type names for nilearn
- `create_events_for_glm()` - Convert raw events to GLM format
- `create_simple_action_models()` - Build simple action event models
- `create_movement_model()` - Build movement event model (LEFT, RIGHT, etc.)
- `create_full_actions_model()` - Build complete action model
- `create_game_events_model()` - Build game event model (Kill, Hit, Powerup, etc.)

**Contrast definitions:**
- `define_movement_contrasts()` - Define movement contrasts (LEFT-RIGHT, etc.)
- `define_game_event_contrasts()` - Define game event contrasts (Reward-Punishment, etc.)

**GLM fitting:**
- `fit_run_glm()` - Fit GLM to a single run
- `compute_contrasts()` - Compute contrast maps
- `aggregate_runs_fixed_effects()` - Aggregate runs to session level
- `threshold_map_clusters()` - Apply cluster-level thresholding

**Design matrix structure:**
- Event regressors (convolved with HRF)
- Motion confounds (24 parameters: 6 motion + derivatives + quadratic)
- Physiological confounds (WM, CSF, global signal)
- Low-level confounds (luminance, optical flow, audio envelope)
- Task confounds (button press counts)
- Drift terms (cosine basis)

### `rl.py` - Reinforcement Learning Agent
Functions for RL agent activation extraction and processing.

**Model architecture:**
- `SimpleCNN` - PPO agent architecture (4-layer CNN + actor-critic)

**Model operations:**
- `load_pretrained_model()` - Load trained model weights
- `play_agent_episode()` - Play episode with agent
- `extract_layer_activations()` - Extract CNN activations during gameplay
- `load_replay_file()` - Load replay (.bk2) file

**Feature processing:**
- `create_simple_proxy_features()` - Create simplified features from annotations
- `align_activations_to_bold()` - Align RL activations to BOLD TRs
- `downsample_activations_to_tr()` - Downsample from 60Hz to fMRI TR
- `convolve_with_hrf()` - Convolve activations with hemodynamic response function
- `apply_pca()` - Dimensionality reduction with PCA

**CNN architecture:**
```
Input: 4 × 84 × 84 (stacked frames)
  ↓
conv1: 32 filters, 3×3, stride 2 → 32 × 42 × 42
  ↓
conv2: 32 filters, 3×3, stride 2 → 32 × 21 × 21
  ↓
conv3: 32 filters, 3×3, stride 2 → 32 × 11 × 11
  ↓
conv4: 32 filters, 3×3, stride 2 → 32 × 6 × 6
  ↓
linear: 1152 → 512 features
  ↓
Actor: 512 → 12 actions
Critic: 512 → 1 value
```

### `parcellation.py` - Brain Parcellation
Functions for creating and working with brain parcellations.

**Parcellation creation:**
- `create_random_parcellation()` - Create random parcellation from mask
- `get_parcel_labels()` - Get labels for parcellation

**Atlas loading:**
- `load_schaefer_atlas()` - Load Schaefer atlas (100-1000 ROIs)
- `load_basc_atlas()` - Load BASC atlas

**BOLD extraction:**
- `extract_parcel_bold_from_parcellation()` - Extract parcel-averaged BOLD

**Caching:**
- `save_complete_results()` - Save complete encoding results to disk
- `load_complete_results()` - Load cached encoding results
- `check_complete_results_exist()` - Check if cache exists
- `extract_parcel_bold()` - Extract BOLD from specific parcels

### `encoding.py` - Brain Encoding Models
Functions for predictive encoding analysis using ridge regression.

**Voxel-wise encoding:**
- `RidgeEncodingModel` - Ridge regression wrapper with cross-validation
- `load_and_prepare_bold()` - Load and clean BOLD data (deconfounding, detrending, standardization)
- `fit_encoding_model_per_layer()` - Fit ridge models for each CNN layer
- `compare_layer_performance()` - Compare R² across layers

**Parcel-wise encoding (atlas-based):**
- `fit_atlas_encoding_per_layer()` - Fit encoding models per parcel
- `compare_atlas_layer_performance()` - Compare layer performance across parcels
- `get_top_parcels()` - Get top parcels by R² score

**Workflow:**
1. Load BOLD data and apply brain mask
2. Deconfound (regress out nuisance variables)
3. Standardize (z-score)
4. Fit ridge regression: BOLD ~ RL features
5. Cross-validate to select regularization strength (alpha)
6. Evaluate on held-out test set
7. Create R² brain maps

### `visualization.py` - Visualization Functions
All plotting and visualization functions consolidated in one module.

**GLM visualizations:**
- `plot_event_frequencies()` - Session event summary with colored bars
- `plot_event_timeline()` - Single run timeline with replay backgrounds
- `plot_confounds_structure()` - 4-panel confound visualization
- `get_design_matrix_figure()` - Design matrix display
- `plot_contrast_surfaces()` - 4-view brain surface plots (fsaverage)
- `plot_contrast_glass_brain()` - Glass brain visualization
- `plot_contrast_stat_map()` - Axial slices visualization

**RL visualizations:**
- `plot_pca_variance_per_layer()` - PCA variance explained per layer
- `plot_layer_activations_sample()` - Sample activations heatmap
- `plot_agent_gameplay()` - Agent gameplay visualization

**Encoding visualizations:**
- `plot_r2_brainmap()` - R² brain map visualization
- `plot_encoding_comparison_table()` - Layer comparison table
- `plot_prediction_examples()` - Actual vs predicted BOLD traces
- `plot_layer_comparison_bars()` - Layer performance bars

**Atlas encoding visualizations:**
- `plot_atlas_r2_surfaces()` - R² on cortical surface (atlas-based)
- `plot_network_performance()` - Network-level performance bars
- `plot_network_performance_grid()` - Network performance across layers (grid)
- `plot_glass_brain_r2()` - Glass brain R² visualization (atlas-based)
- `visualize_best_parcel_prediction()` - Best parcel prediction trace

## Usage in Notebooks

All notebooks import from this package using the new module names:

```python
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))

# Import from new modules
from utils import get_sourcedata_path, load_events, load_bold
from glm import prepare_confounds, fit_run_glm, compute_contrasts
from rl import load_pretrained_model, extract_layer_activations, apply_pca
from parcellation import create_random_parcellation, load_schaefer_atlas
from encoding import fit_atlas_encoding_per_layer, compare_atlas_layer_performance
from visualization import plot_contrast_surfaces, plot_network_performance_grid
```

**Backward compatibility:** Old imports (e.g., `from glm_utils import ...`) still work with deprecation warnings in version 0.2.0.

## Dependencies

**Core:**
- numpy, scipy, pandas
- nibabel, nilearn
- scikit-learn
- matplotlib, seaborn

**RL (optional):**
- torch (PyTorch)
- gym-retro (for full RL training)

**Setup:**
- datalad (for dataset management)
- gdown (for Google Drive downloads)

## File Structure

```
src/
├── __init__.py           # Package initialization (v0.2.0, exports from all modules)
├── README.md            # This file
├── utils.py             # General utilities + setup functions (28K)
├── glm.py               # GLM analysis functions (17K)
├── rl.py                # RL agent utilities (40K)
├── parcellation.py      # Parcellation creation and atlas loading (15K)
├── encoding.py          # Brain encoding models (25K)
└── visualization.py     # All visualization functions (54K)
```

**Total:** 6 modules + __init__.py (179K total code)

## Migration from v0.1.0

If you have code using the old module names, update imports:

| Old Import | New Import |
|------------|------------|
| `from glm_utils import ...` | `from glm import ...` or `from visualization import ...` |
| `from rl_utils import ...` | `from rl import ...` |
| `from rl_viz_utils import ...` | `from visualization import ...` |
| `from encoding_utils import ...` | `from encoding import ...` |
| `from encoding_viz_utils import ...` | `from visualization import ...` |
| `from parcellation_utils import ...` | `from parcellation import ...` |
| `from atlas_encoding_utils import ...` | `from encoding import ...`, `from parcellation import ...`, or `from visualization import ...` |
| `from setup_utils import ...` | `from utils import ...` |

**Example:**
```python
# Old (v0.1.0)
from glm_utils import plot_contrast_surfaces, fit_run_glm
from encoding_utils import fit_encoding_model_per_layer
from rl_viz_utils import plot_pca_variance_per_layer

# New (v0.2.0)
from glm import fit_run_glm
from encoding import fit_encoding_model_per_layer
from visualization import plot_contrast_surfaces, plot_pca_variance_per_layer
```

## References

- **shinobi_fmri**: Session-level GLM methodology
- **mario_generalization**: RL training and encoding
- **nilearn**: [nilearn.github.io](https://nilearn.github.io) - Python neuroimaging
- **fMRIPrep**: [fmriprep.org](https://fmriprep.org) - Preprocessing pipeline

## Changelog

### v0.2.0 (January 2025)
- **Major reorganization:** 10 files → 6 modules for cleaner structure
- **New setup functions:** Automated environment setup, DataLad installation, stimuli download
- **Consolidated visualization:** All plot functions in single module
- **Parcel-wise encoding:** Added atlas-based encoding functions
- **Backward compatibility:** Old imports work with deprecation warnings
- **Improved documentation:** Complete function reference with examples

### v0.1.0 (Initial)
- Initial modular structure with separate utility files
