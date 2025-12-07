# Mario fMRI Tutorial - Source Code Modules

This directory contains the core analysis modules for the Mario fMRI tutorial.

## Module Overview

### `utils.py` - General Utilities
Core functions for data loading and path management.

**Key functions:**
- `get_sourcedata_path()` - Locate source data directory
- `get_derivatives_path()` - Locate derivatives directory
- `load_events()` - Load behavioral annotations (onset, duration, trial_type)
- `load_bold()` - Load preprocessed fMRI data
- `load_brain_mask()` - Load brain mask
- `load_confounds()` - Load fMRIPrep confounds
- `load_lowlevel_confounds()` - Load low-level confounds (luminance, optical flow, audio)
- `get_session_runs()` - Get list of runs for a session
- `create_output_dir()` - Create derivatives output directory
- `save_stat_map()` - Save statistical maps with BIDS-like naming

### `glm_utils.py` - GLM Analysis
Functions for General Linear Model analysis following shinobi_fmri methodology.

**Key functions:**
- `prepare_confounds()` - Extract and prepare confounds (motion, WM, CSF, global signal)
- `downsample_lowlevel_confounds()` - Downsample low-level confounds from 60Hz to TR
- `add_button_press_counts()` - Add button press counts as task confound
- `create_movement_model()` - Build movement event model (LEFT, RIGHT, etc.)
- `create_game_events_model()` - Build game event model (Kill, Hit, Powerup, etc.)
- `define_movement_contrasts()` - Define movement contrasts (LEFT-RIGHT, etc.)
- `define_game_event_contrasts()` - Define game event contrasts (Reward-Punishment, etc.)
- `fit_run_glm()` - Fit GLM to a single run
- `compute_contrasts()` - Compute contrast maps
- `aggregate_runs_fixed_effects()` - Aggregate runs to session level
- `threshold_map_clusters()` - Apply cluster-level thresholding
- `get_design_matrix_figure()` - Visualize design matrix

**Design matrix structure:**
- Event regressors (convolved with HRF)
- Motion confounds (24 parameters: 6 motion + derivatives + quadratic)
- Physiological confounds (WM, CSF, global signal)
- Low-level confounds (luminance, optical flow, audio envelope)
- Task confounds (button press counts)
- Drift terms (cosine basis)

### `encoding_utils.py` - Brain Encoding Models
Functions for predictive encoding analysis using ridge regression.

**Key functions:**
- `RidgeEncodingModel` - Ridge regression wrapper with cross-validation
- `load_and_prepare_bold()` - Load and clean BOLD data (deconfounding, detrending, standardization)
- `fit_encoding_model_per_layer()` - Fit ridge models for each CNN layer
- `compare_layer_performance()` - Compare R² across layers
- `create_encoding_summary_figure()` - Visualize layer comparison

**Workflow:**
1. Load BOLD data and apply brain mask
2. Deconfound (regress out nuisance variables)
3. Standardize (z-score)
4. Fit ridge regression: BOLD ~ RL features
5. Cross-validate to select regularization strength (alpha)
6. Evaluate on held-out test set
7. Create R² brain maps

### `rl_utils.py` - Reinforcement Learning Utilities
Functions for RL agent activation extraction and processing.

**Key functions:**
- `SimpleCNN` - PPO agent architecture (4-layer CNN + actor-critic)
- `load_pretrained_model()` - Load trained model weights
- `create_simple_proxy_features()` - Create simplified features from annotations
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

## Usage in Notebooks

All notebooks import from this package:

```python
import sys
from pathlib import Path

# Add code directory to path
code_dir = Path('..') / 'code'
sys.path.insert(0, str(code_dir))

# Import utilities
from utils import load_events, load_bold, get_sourcedata_path
from glm_utils import prepare_confounds, fit_run_glm
from encoding_utils import fit_encoding_model_per_layer
from rl_utils import convolve_with_hrf, apply_pca
```

Or import the entire package:

```python
import code  # After adding to path

# Use functions
events = code.load_events(subject, session, run)
confounds = code.prepare_confounds(confounds_raw)
```

## Dependencies

**Core:**
- numpy, scipy, pandas
- nibabel, nilearn
- scikit-learn
- matplotlib, seaborn

**RL (optional):**
- torch (PyTorch)
- gym-retro (for full RL training)

## File Structure

```
src/
├── __init__.py           # Package initialization, exports all functions
├── README.md            # This file
├── utils.py             # General utilities (paths, loading)
├── glm_utils.py         # GLM analysis functions
├── encoding_utils.py    # Brain encoding models
└── rl_utils.py          # RL agent utilities
```

## References

- **shinobi_fmri**: [GitHub link] - Session-level GLM methodology
- **mario_generalization**: [GitHub link] - RL training and encoding
- **nilearn**: [nilearn.github.io](https://nilearn.github.io) - Python neuroimaging
- **fMRIPrep**: [fmriprep.org](https://fmriprep.org) - Preprocessing pipeline
