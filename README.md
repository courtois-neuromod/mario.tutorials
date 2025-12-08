# Super Mario Bros. fMRI Dataset - Complete Tutorial

A comprehensive tutorial for analyzing naturalistic fMRI data from the CNeuromod Mario dataset, combining session-level GLM analysis, brain map visualization, RL agent training, and brain encoding models.

## Overview

This tutorial demonstrates a complete fMRI analysis pipeline from data exploration to brain encoding, following methodologies from:
- **shinobi_fmri**: Session-level GLM modeling and visualization
- **mario_generalization**: RL agent training and brain encoding

**Scope**: Single subject (sub-01), single session (ses-010) - optimized for laptop execution (~30-45 minutes)

## Installation

### Prerequisites

- **Operating System**: Linux, macOS, or Windows (via WSL)
- **Python**: 3.8 or higher
- **DataLad**: Required for dataset management
- **Storage**: ~8 GB for single session analysis

### Platform-Specific Setup

#### Linux

```bash
# Install DataLad (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install datalad

# Or using pip
pip install datalad

# Navigate to tutorial directory
cd mario.tutorials

# Run setup script
bash setup_environment.sh

# Install mario datasets
bash install_mario_datasets.sh

# Activate environment
source venv/bin/activate
```

#### macOS

```bash
# Install DataLad using Homebrew
brew install datalad

# Or using pip
pip install datalad

# Navigate to tutorial directory
cd mario.tutorials

# Run setup script
bash setup_environment.sh

# Install mario datasets
bash install_mario_datasets.sh

# Activate environment
source venv/bin/activate
```

#### Windows (via WSL)

**Note**: This tutorial requires Windows Subsystem for Linux (WSL). If you don't have WSL installed, follow the [official WSL installation guide](https://learn.microsoft.com/en-us/windows/wsl/install).

```bash
# Open WSL terminal (Ubuntu recommended)
# Install DataLad
sudo apt-get update
sudo apt-get install datalad

# Navigate to tutorial directory
cd mario.tutorials

# Run setup script
bash setup_environment.sh

# Install mario datasets
bash install_mario_datasets.sh

# Activate environment
source venv/bin/activate
```

### 2. Ensure Data Availability

Required data (should be in `../sourcedata/`):
- `mario/sub-01/ses-010/` - Raw BIDS data
- `mario.fmriprep/sub-01/ses-010/` - Preprocessed fMRI
- `mario.annotations/annotations/sub-01/ses-010/` - Behavioral annotations
- `cneuromod.processed/smriprep/sub-01/anat/` - Anatomical data

Total: ~7-8 GB for single session

### 3. Run Tutorial

Launch Jupyter and work through the notebooks sequentially:

```bash
jupyter notebook
```

Navigate to the `notebooks/` directory and open the tutorials in order.

## Tutorial Structure

### Notebooks

The tutorial is organized into four main notebooks:

| Notebook | Description | Duration |
|----------|-------------|----------|
| `00_dataset_overview.ipynb` | Dataset exploration and behavioral annotations | 15 min |
| `01_event_based_analysis.ipynb` | GLM analysis for actions and game events | 20 min |
| `02_reinforcement_learning.ipynb` | RL agent training and CNN activation extraction | 25 min |
| `03_brain_encoding.ipynb` | Ridge regression encoding models and layer comparison | 20 min |

**Total time**: ~80 minutes for complete pipeline

### What You'll Learn

- **Notebook 0**: Understand the CNeuromod Mario dataset structure, explore behavioral annotations, and visualize event timelines
- **Notebook 1**: Build GLM models, fit session-level analyses, compute contrasts (hand lateralization), and apply FWE correction
- **Notebook 2**: Train a PPO agent, extract CNN layer activations, apply PCA for dimensionality reduction, and visualize learned representations
- **Notebook 3**: Fit ridge regression encoding models, compare layer performance, and visualize brain prediction maps

### Source Code Modules

Analysis modules are organized in the `src/` directory:

- `src/utils.py` - General utilities (data loading, BIDS helpers, path management)
- `src/glm_utils.py` - GLM analysis functions (confounds, design matrices, fitting, contrasts)
- `src/rl_utils.py` - RL agent utilities (CNN architecture, activations, HRF, PCA)
- `src/encoding_utils.py` - Brain encoding models (ridge regression, evaluation, visualization)

See `src/README.md` for detailed documentation of each module.

## Analysis Pipeline

```
📊 Dataset Overview
    ↓
    Explore BIDS structure
    Load behavioral annotations
    Visualize event timelines

🧠 Event-Based Analysis (GLM)
    ↓
    Build design matrices
    Fit multi-run GLM models
    Compute statistical contrasts
    Apply FWE correction

🤖 Reinforcement Learning
    ↓
    Train PPO agent
    Extract CNN activations
    Apply PCA reduction
    Visualize representations

📈 Brain Encoding
    ↓
    Fit ridge regression models
    Compare layer performance
    Generate prediction maps
```

## Key Features

### GLM Analysis (shinobi_fmri style)
- **Confound strategy**: 24 motion params + WM/CSF + global signal
- **Models**:
  - Simple actions (one at a time)
  - Intermediate movement (LEFT vs RIGHT)
  - Full actions (all buttons)
  - Game events (reward/punishment)
- **Session-level aggregation**: Fixed-effects across runs
- **HRF**: SPM canonical, AR(1) noise model

### Brain Encoding (mario_generalization style)
- **RL Agent**: 4-layer CNN (PPO architecture)
- **Training options**:
  - Quick: Imitation learning (~5 min)
  - Recommended: Pre-trained weights (~1 min)
  - Advanced: Full PPO training (~2 hours)
- **Encoding**: Ridge regression with cross-validated α
- **Layer analysis**: Compare conv1-4 and linear layers
- **PCA reduction**: 50 components per layer for efficiency

### Visualization
- Surface projections (fsaverage)
- Glass brain views
- Statistical thresholding (FDR, cluster correction)
- R² brain maps for encoding models

## Dependencies

See `requirements.txt` for full list. Key packages:
- **Neuroimaging**: nilearn, nibabel
- **ML**: scikit-learn, torch
- **RL**: gym, stable-retro
- **Jupyter**: jupyter, notebook
- **Viz**: matplotlib, seaborn

## Dataset Information

**CNeuromod Mario Dataset**
- Task: Play Super Mario Bros (NES) naturally
- Subjects: 5 (sub-01, sub-02, sub-03, sub-05, sub-06)
- Sessions: 17-30 per subject
- Runs: ~5 per session, ~5 minutes each
- TR: 1.49 seconds
- Levels: 6 training (w1l1, w1l2, w4l1, w4l2, w5l1, w5l2) + 2 OOD (w2l1, w3l1)

**Annotations**:
- Actions: A, B, LEFT, RIGHT, UP, DOWN
- Events: Kills, hits, powerups, coins
- Scenes: Level segmentation
- Replays: .bk2 files with frame-by-frame state

## Resources

- **CNeuromod**: https://www.cneuromod.ca
- **CNeuromod Documentation**: https://docs.cneuromod.ca/
- **Nilearn Documentation**: https://nilearn.github.io
- **DataLad Handbook**: https://handbook.datalad.org/

## Troubleshooting

### Common Issues

**Issue**: DataLad installation fails
- **Solution**: Try using pip instead: `pip install datalad`

**Issue**: Notebooks can't find source modules
- **Solution**: Ensure you're running Jupyter from the `notebooks/` directory and that `src/` is in the parent directory

**Issue**: BOLD data not found
- **Solution**: Run `bash install_mario_datasets.sh` to download the required datasets

**Issue**: Out of memory errors
- **Solution**: The tutorial is designed for laptops with 8GB+ RAM. If issues persist, reduce the number of voxels by using a more restrictive mask.

---

**GL&HF!** 🧠🎮
