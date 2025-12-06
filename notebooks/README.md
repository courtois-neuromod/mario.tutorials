# Mario fMRI Tutorial Notebooks

This directory contains 6 Jupyter notebooks for a comprehensive Mario fMRI analysis tutorial.

## Notebook Overview

### 01_dataset_exploration.ipynb (10 min)
- BIDS organization walkthrough
- Load and visualize behavioral annotations
- Event frequency analysis and timelines
- Examine replay data structure

### 02_session_glm.ipynb (15 min)
- Load BOLD data and confounds
- Build design matrices for multiple models (actions, movement, events)
- Fit run-level GLMs
- Aggregate to session level with fixed-effects
- Generate design matrix visualizations

### 03_brain_visualization.ipynb (10 min)
- Load GLM results from notebook 02
- Apply statistical thresholding (raw, FDR, cluster)
- Create surface projections (fsaverage)
- Glass brain views and slice displays
- Comparison panels (LEFT vs RIGHT, reward vs punishment)
- Interactive HTML reports

### 04_rl_agent.ipynb (20 min, or 5 min with pre-trained model)
- Explain PPO architecture
- Option B: Load pre-trained model (recommended)
- Extract activations from all CNN layers
- Downsample to TR and convolve with HRF
- Apply PCA dimensionality reduction
- Visualize variance explained per layer

### 05_brain_encoding.ipynb (15 min)
- Load and prepare BOLD data (deconfounding)
- Fit ridge regression per layer
- Evaluate with cross-validation
- Compare layer performance (bar plot)
- Create R² brain maps for best layer
- Visualize encoding quality across brain regions

### 06_summary.ipynb (5 min)
- Recap key findings
- Compare GLM vs encoding approaches
- Show side-by-side brain maps
- List extensions (multi-subject, OOD, MVPA, temporal)
- Provide resources and links

## Data Requirements

**Subject**: sub-01  
**Session**: ses-010

**Required data** (in `../sourcedata/`):
- `mario.fmriprep/sub-01/ses-010/` - Preprocessed BOLD and masks
- `mario.annotations/annotations/sub-01/ses-010/` - Behavioral events
- `cneuromod.processed/smriprep/sub-01/anat/` - Anatomical templates (optional)
- `mario.replays/sub-01/ses-010/` - Game replays (optional)

## Running the Notebooks

1. **Setup environment**:
   ```bash
   cd ..
   bash setup_environment.sh
   conda activate mario_tutorial
   ```

2. **Launch Jupyter**:
   ```bash
   jupyter lab
   ```

3. **Run notebooks in order**: 01 → 02 → 03 → 04 → 05 → 06

4. **Expected runtime**: ~75 minutes total (or ~60 min with pre-trained model)

## Output Locations

All outputs are saved to `../derivatives/`:
- `glm_tutorial/` - GLM statistical maps
- `rl_agent/` - RL activations and models
- `encoding/` - Encoding R² maps and figures

## Dependencies

See `../requirements.txt` for full list. Key packages:
- nibabel, nilearn (fMRI analysis)
- numpy, scipy, pandas (data processing)
- scikit-learn (encoding models)
- matplotlib, seaborn (visualization)
- torch (RL models)

## Troubleshooting

**Missing data**: If sourcedata is not available, notebooks will show warnings and use dummy data where possible.

**Import errors**: Ensure scripts directory is in Python path (notebooks handle this automatically).

**Memory issues**: Reduce spatial resolution or use fewer runs if running on limited RAM.

**Slow execution**: Skip smoothing in GLM (set `smoothing_fwhm=None`), reduce PCA components.

## Support

- Tutorial issues: GitHub repository
- CNeuromod data: https://www.cneuromod.ca
- Nilearn documentation: https://nilearn.github.io

## Citation

If you use this tutorial or the CNeuromod dataset, please cite:
- CNeuromod dataset papers
- Tutorial repository (when published)
- Relevant software packages (nilearn, fMRIPrep, etc.)

---

**Total estimated time**: 75 minutes  
**Difficulty level**: Intermediate  
**Prerequisites**: Basic Python, fMRI concepts, linear regression
