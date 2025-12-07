# Super Mario Bros. fMRI Dataset - Complete Tutorial

A comprehensive tutorial for analyzing naturalistic fMRI data from the CNeuromod Mario dataset, combining session-level GLM analysis, brain map visualization, RL agent training, and brain encoding models.

## Overview

This tutorial demonstrates a complete fMRI analysis pipeline from data exploration to brain encoding, following methodologies from:
- **shinobi_fmri**: Session-level GLM modeling and visualization
- **mario_generalization**: RL agent training and brain encoding

**Scope**: Single subject (sub-01), single session (ses-010) - optimized for laptop execution (~30-45 minutes)

## Quick Start

### 1. Setup Environment

```bash
# Navigate to tutorial directory
cd mario.tutorials

# Run setup script
bash setup_environment.sh

# Activate virtual environment
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

**Option A: RISE Presentation (1 hour overview)**
```bash
jupyter notebook notebooks/00_presentation_RISE.ipynb
```
Click the RISE button to start slideshow mode.

**Option B: Detailed Notebooks**
```bash
jupyter notebook
```
Navigate through notebooks 01-06 sequentially.

## Tutorial Structure

### Notebooks

| Notebook | Description | Duration |
|----------|-------------|----------|
| `00_presentation_RISE.ipynb` | ⭐ **RISE slideshow** - 1-hour overview of entire pipeline | 60 min |
| `01_dataset_exploration.ipynb` | BIDS organization, behavioral annotations, protocol | 10 min |
| `02_session_glm.ipynb` | Session-level GLM for actions and events | 15 min |
| `03_brain_visualization.ipynb` | Statistical brain maps and visualizations | 10 min |
| `04_rl_agent.ipynb` | RL agent training and activation extraction | 20 min |
| `05_brain_encoding.ipynb` | Ridge regression brain encoding models | 15 min |
| `06_summary.ipynb` | Summary and extensions | 5 min |

### Source Code Modules

Analysis modules are organized in the `src/` directory:

- `src/utils.py` - General utilities (data loading, BIDS helpers, path management)
- `src/glm_utils.py` - GLM analysis functions (confounds, design matrices, fitting, contrasts)
- `src/rl_utils.py` - RL agent utilities (CNN architecture, activations, HRF, PCA)
- `src/encoding_utils.py` - Brain encoding models (ridge regression, evaluation, visualization)

See `src/README.md` for detailed documentation of each module.

## Analysis Pipeline

```
📊 Dataset Exploration
    ↓
🧠 GLM Analysis (Actions + Events)
    ↓
🎨 Brain Visualization (Surface + Glass Brain)
    ↓
🤖 RL Agent (CNN activations)
    ↓
📈 Brain Encoding (Ridge regression)
    ↓
✅ Summary & Extensions
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
- Interactive HTML reports

## RISE Presentation

The `00_presentation_RISE.ipynb` notebook is a complete 1-hour slideshow covering all tutorial content:

### Structure (25 slides)
1. **Introduction** (10 min) - Dataset & pipeline overview
2. **Dataset** (8 min) - Annotations, timelines, replays
3. **GLM** (12 min) - Model fitting, movement contrasts, reward/punishment
4. **RL Agent** (15 min) - Architecture, training, activation extraction
5. **Encoding** (18 min) - Ridge models, layer comparison, brain maps
6. **Synthesis** (7 min) - GLM vs encoding, key findings

### How to Present
```bash
jupyter notebook notebooks/00_presentation_RISE.ipynb
```

1. Click the RISE icon (bar chart) in toolbar
2. Use spacebar to advance, shift+spacebar to go back
3. Press 'Esc' to exit slideshow
4. Execute cells during presentation (pre-cached for speed)

**Tips**:
- Pre-run computationally intensive cells before presenting
- Results cached in `derivatives/presentation_cache/`
- Adjust timing based on audience questions

## Dependencies

See `requirements.txt` for full list. Key packages:
- **Neuroimaging**: nilearn, nibabel, nistats
- **ML**: scikit-learn, torch, pytorch-lightning
- **RL**: gym, stable-retro
- **Presentation**: RISE, jupyter
- **Viz**: matplotlib, seaborn, plotly

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
- **Nilearn documentation**: https://nilearn.github.io
- **RISE documentation**: https://rise.readthedocs.io

---

**GL&HF!** 🧠🎮
