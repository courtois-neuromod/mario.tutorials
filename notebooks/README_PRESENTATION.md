# Mario fMRI Tutorial - RISE Presentation

## Overview

This RISE slideshow provides a **1-hour overview** of the complete Mario fMRI analysis pipeline, from dataset exploration through GLM analysis, RL agent training, and brain encoding models.

## Files

- **00_presentation_RISE.ipynb** - Main presentation notebook (25 slides)
- **PRESENTATION_GUIDE.md** - Detailed presenter's guide with timing and tips
- **README_PRESENTATION.md** - This file

## Quick Start

### 1. Install RISE

```bash
pip install RISE
```

### 2. Launch Jupyter

```bash
cd /home/hyruuk/GitHub/neuromod/mario_analysis/mario.tutorials/notebooks/
jupyter notebook 00_presentation_RISE.ipynb
```

### 3. Start Slideshow

- Click the bar chart icon in toolbar, OR
- Press `Alt+R`, OR  
- View → Cell Toolbar → Slideshow → Click "Enter/Exit RISE Slideshow"

### 4. Navigate

- **Next:** Space, →, Page Down
- **Previous:** ←, Page Up
- **Overview:** O
- **Speaker notes:** S
- **Exit:** Esc

## Structure (25 slides, ~70 min)

### Section 1: Introduction (10 min, 4 slides)
- Title and overview
- CNeuromod Mario dataset
- Analysis pipeline flowchart
- Focus on sub-01, ses-010

### Section 2: Dataset Exploration (8 min, 3 slides)
- Behavioral annotations (with execution)
- Event visualization (with execution)
- Game replay data

### Section 3: GLM Analysis (12 min, 5 slides)
- GLM fundamentals
- Movement model: LEFT vs RIGHT (with execution)
- Brain maps showing motor lateralization
- Game events: Reward vs Punishment (with execution)
- Reward/punishment brain maps

### Section 4: RL Agent (15 min, 5 slides)
- Why RL for fMRI?
- PPO architecture diagram
- Training options (using proxy features)
- Feature creation and HRF convolution (with execution)
- PCA dimensionality reduction (with execution)

### Section 5: Brain Encoding (18 min, 5 slides)
- Encoding framework (ridge regression)
- BOLD data preparation (with execution)
- Fitting models per layer (with execution, ~4 min)
- Layer comparison (with execution)
- R² brain maps (with execution)

### Section 6: Synthesis (7 min, 3 slides)
- GLM vs Encoding comparison
- Key takeaways (5 main findings)
- Extensions and future directions
- Thank you + resources

## Timing Breakdown

| Component | Duration |
|-----------|----------|
| Speaking | ~52 min |
| Code execution | ~18 min |
| **Total** | **~70 min** |
| Buffer for Q&A | 10 min |

## Key Features

### RISE Configuration
- Theme: Simple
- Slide numbers: Enabled
- Scrolling: Enabled  
- Chalkboard: Enabled (press 'C')
- Header: "Mario fMRI Tutorial"
- Footer: "CNeuromod 2025"

### Code Execution
- Most cells execute in <1 min
- Longest: Encoding models (~4 min)
- All cells include `%%time` magic
- Graceful fallbacks if data unavailable

### Slide Types
- **slide**: Section transitions
- **subslide**: Content within sections
- **skip**: Hidden setup cells
- **notes**: Speaker notes (press 'S')

## Requirements

### Software
- Python 3.9+
- Jupyter Notebook
- RISE extension
- Standard scientific Python stack (numpy, pandas, matplotlib, seaborn)
- Neuroimaging libraries (nibabel, nilearn)
- scikit-learn, torch (for RL components)

### Data (optional)
- CNeuromod Mario dataset (sub-01, ses-010)
- Path: `sourcedata/mario.fmriprep/`, `mario.annotations/`, etc.
- Presentation works without data (explains expected results)

## Customization

### Change Subject/Session

Edit setup cell:
```python
SUBJECT = 'sub-01'
SESSION = 'ses-010'
```

### Shorten to 45 min
- Skip timeline visualization (slide 8)
- Reduce detailed interpretations
- Use fewer CV folds for encoding

### Extend to 90 min
- Add session-level aggregation
- Show multiple layer visualizations
- Include ROI analysis

## Tips for Success

### Before Presentation
1. Test run all cells (Cell → Run All)
2. Clear outputs (Cell → All Output → Clear)
3. Check data paths in setup cell
4. Review PRESENTATION_GUIDE.md for detailed tips

### During Presentation
1. Use speaker notes (press 'S')
2. Monitor `%%time` outputs for pacing
3. Explain concepts during long executions
4. Skip optional slides if behind schedule

### If Errors Occur
- Slides explain expected results
- Graceful degradation built-in
- Can skip code and continue narrative
- Reference detailed notebooks (01-06)

## Follow-up Materials

After presentation, direct attendees to:

1. **Detailed notebooks:**
   - 01_dataset_exploration.ipynb
   - 02_session_glm.ipynb
   - 03_brain_visualization.ipynb
   - 04_rl_agent.ipynb
   - 05_brain_encoding.ipynb
   - 06_summary.ipynb

2. **Design document:**
   - ../TUTORIAL_DESIGN.md

3. **Helper scripts:**
   - ../scripts/utils.py
   - ../scripts/glm_utils.py
   - ../scripts/rl_utils.py
   - ../scripts/encoding_utils.py

4. **Related projects:**
   - shinobi_fmri (production GLM)
   - mario_generalization (full RL pipeline)

## Resources

- **CNeuromod:** https://www.cneuromod.ca/
- **Documentation:** https://docs.cneuromod.ca/
- **RISE docs:** https://rise.readthedocs.io/

## Support

For issues or questions:
1. Check PRESENTATION_GUIDE.md for troubleshooting
2. Review detailed notebooks for working examples
3. Consult TUTORIAL_DESIGN.md for pipeline overview

---

**Ready to present!** 🎬 Good luck!
