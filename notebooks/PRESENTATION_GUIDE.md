# RISE Presentation Guide

## Quick Start

### Installation

First, ensure RISE is installed:

```bash
pip install RISE
```

Or with conda:

```bash
conda install -c conda-forge rise
```

### Launching the Presentation

1. Open the notebook:
   ```bash
   jupyter notebook 00_presentation_RISE.ipynb
   ```

2. Start the slideshow:
   - Click the bar chart icon in the toolbar, OR
   - Press `Alt+R`, OR
   - Use the menu: View → Cell Toolbar → Slideshow

3. Enter presentation mode:
   - Click the "Enter/Exit RISE Slideshow" button (appears after step 2)

### Navigation

- **Next slide:** Space, →, Page Down
- **Previous slide:** ←, Page Up
- **Toggle overview:** `O`
- **Speaker notes:** `S` (shows presenter view in new window)
- **Chalkboard:** `C` (draw on slides)
- **Fullscreen:** `F`
- **Exit:** `Esc` or click "X"

---

## Presentation Structure

### Timing Overview (70 minutes total)

| Section | Slides | Talk | Execution | Total |
|---------|--------|------|-----------|-------|
| 1. Introduction | 1-4 | 10 min | 0 min | 10 min |
| 2. Dataset | 5-7 | 5 min | 3 min | 8 min |
| 3. GLM | 8-12 | 7 min | 5 min | 12 min |
| 4. RL Agent | 13-17 | 11 min | 4 min | 15 min |
| 5. Encoding | 18-22 | 12 min | 6 min | 18 min |
| 6. Synthesis | 23-25 | 7 min | 0 min | 7 min |
| **Total** | **25 slides** | **52 min** | **18 min** | **70 min** |

*Leaves ~10 minutes for Q&A in a 1-hour slot*

---

## Slide-by-Slide Guide

### Section 1: Introduction (10 min)

**Slide 1: Title**
- Welcome, introduce tutorial scope
- Set expectations for live code execution

**Slide 2: CNeuromod Dataset**
- Emphasize naturalistic paradigm
- Highlight dataset richness (annotations, replays)
- Note: 5 subjects, focus on sub-01/ses-010

**Slide 3: Pipeline Overview**
- Walk through flowchart
- Explain GLM vs Encoding dichotomy
- Preview what's coming

**Slide 4: Today's Focus**
- Justify single-session approach
- Show BIDS structure
- Mention ~30-45 min runtime

---

### Section 2: Dataset Exploration (8 min)

**Slide 5: Behavioral Annotations**
- Explain three event types
- Transition to live code

**Slide 6: Load Events (CODE - 1 min execution)**
- Run cell to load session events
- Point out event counts (buttons vs game events)
- Expected output: ~200+ events total

**Slide 7: Event Visualization (CODE - 2 min execution)**
- Show frequency bar plot
- Highlight most common events (LEFT, RIGHT, A)
- Discuss category breakdown

**Slide 8: Timeline Visualization (CODE - optional)**
- If time permits, show temporal patterns
- Point out event clustering
- Skip if running behind

**Slide 9: Game Replay Data**
- Explain .bk2 files (no code execution)
- Mention uses for RL training
- Note we'll use proxy features

---

### Section 3: GLM Analysis (12 min)

**Slide 10: GLM Fundamentals**
- Quick refresher on GLM equation
- Walk through confound strategy
- Preview two models

**Slide 11: Fit Movement GLM (CODE - 2 min execution)**
- Run GLM fitting for LEFT vs RIGHT
- Show design matrix while waiting
- Expected: Design matrix figure

**Slide 12: LEFT-RIGHT Contrast (CODE - 1 min execution)**
- Compute and visualize contrast
- Point to motor cortex lateralization
- Explain contralateral control

**Slide 13: Movement Brain Maps**
- Interpret glass brain view
- Validate expected results
- Connect to neuroanatomy

**Slide 14: Game Events GLM (CODE - 1 min execution)**
- Fit Reward vs Punishment model
- Mention striatum/insula hypothesis

**Slide 15: Reward/Punishment Maps (CODE - 1 min execution)**
- Visualize contrast
- Identify reward system activation
- Link to RL value learning

---

### Section 4: RL Agent (15 min)

**Slide 16: Why RL?**
- Contrast with GLM limitations
- Explain data-driven approach
- Set up encoding motivation

**Slide 17: PPO Architecture**
- Walk through network layers
- Draw analogy to visual cortex
- Highlight hierarchical learning

**Slide 18: Training Options**
- Explain three options (A, B, C)
- Justify proxy features for demo
- Acknowledge trade-offs

**Slide 19: Create Proxy Features (CODE - 2 min execution)**
- Run feature creation
- Show layer dimensions
- Explain HRF convolution

**Slide 20: Apply PCA (CODE - 1 min execution)**
- Reduce dimensions to 50 components
- Show variance explained
- Prepare for encoding

**Slide 21: Variance Visualization (CODE - 1 min execution)**
- Display PCA plots per layer
- Point to 90% variance threshold
- Note differences across layers

---

### Section 5: Brain Encoding (18 min)

**Slide 22: Encoding Framework**
- Introduce ridge regression equation
- Explain voxel-wise fitting
- Preview key questions

**Slide 23: Load BOLD Data (CODE - 1 min execution)**
- Load and prepare fMRI
- Show data shapes
- Mention deconfounding

**Slide 24: Fit Encoding Models (CODE - 4 min execution)**
- This is the longest execution!
- Explain what's happening (5 layers × 50k voxels)
- Good time for Q&A or detailed explanation

**Slide 25: Layer Comparison (CODE - 1 min execution)**
- Show R² bar plot
- Identify best layer
- Discuss typical hierarchy

**Slide 26: Brain Maps (CODE - 1 min execution)**
- Visualize R² maps for best layer
- Point to hot spots
- Interpret regional encoding

---

### Section 6: Synthesis (7 min)

**Slide 27: GLM vs Encoding**
- Comparison table
- Convergent evidence in motor cortex
- Unique insights from each method

**Slide 28: Key Takeaways**
- Summarize 5 main findings
- Emphasize complementary approaches
- Reflect on tutorial arc

**Slide 29: Extensions**
- Preview possible directions
- Multi-subject, OOD, MVPA, temporal
- Invite exploration of detailed notebooks

**Slide 30: Thank You**
- Show resources (repos, data portal)
- List detailed notebooks (01-06)
- Open for questions

---

## Tips for Smooth Presentation

### Before Starting

1. **Test run:** Execute all cells before presentation (Cell → Run All)
2. **Clear outputs:** Cell → All Output → Clear (for clean demo)
3. **Check data:** Ensure sourcedata is accessible at expected paths
4. **Prepare fallbacks:** If data unavailable, slides explain expected results

### During Presentation

1. **Use speaker notes:** Press `S` for presenter view
2. **Manage time:** Use %%time magic outputs to track execution
3. **Skip cells if needed:** Some slides marked "optional" for time management
4. **Explain while executing:** Long cells (GLM, encoding) are good teaching moments
5. **Have backup figures:** Pre-run results in case of errors

### If Something Fails

- **Graceful degradation:** Most slides explain expected results
- **Skip and continue:** Not all code must execute for narrative flow
- **Use static examples:** Reference detailed notebooks for complex outputs

---

## Customization

### Adjust Timing

To shorten (45 min version):
- Skip slide 8 (timeline visualization)
- Skip detailed interpretations on slides 13, 21
- Reduce encoding wait time by using fewer alphas

To extend (90 min version):
- Add session-level aggregation (notebook 02)
- Show multiple layer visualizations (notebook 05)
- Include ROI analysis (notebook 05, slide 24)

### Modify Data

Edit setup cell (hidden) to change:
```python
SUBJECT = 'sub-01'  # Change subject
SESSION = 'ses-010'  # Change session
TR = 1.49          # Change TR if needed
```

### Theme Options

Modify notebook metadata (`rise` section) for different themes:
- `theme`: "simple", "black", "white", "league", "sky", "beige", "serif"
- `transition`: "fade", "slide", "convex", "concave", "zoom"

---

## Troubleshooting

### RISE not appearing
- Reinstall: `pip install --upgrade RISE`
- Enable extension: `jupyter nbextension enable rise --py --sys-prefix`

### Slides not advancing
- Check cell metadata (View → Cell Toolbar → Slideshow)
- Ensure cells are marked: slide, subslide, fragment, skip, or notes

### Code execution errors
- Verify sourcedata paths in setup cell
- Check data availability (sourcedata/mario.fmriprep/...)
- Review error messages - graceful fallbacks built in

### Timing issues
- Monitor execution times (%%time outputs)
- Skip optional slides if behind
- Pre-compute heavy cells before presentation

---

## Post-Presentation

### Share Materials

Attendees can access:
1. Full tutorial: notebooks/01-06
2. Detailed guide: TUTORIAL_DESIGN.md
3. Helper functions: scripts/*.py
4. This guide: PRESENTATION_GUIDE.md

### Follow-up Exercises

Point to:
- Notebook 06 for extensions
- shinobi_fmri for production GLM
- mario_generalization for full RL pipeline

---

## Questions?

Common Q&A:

**Q: How long to run full tutorial?**
A: ~30-45 minutes for single session (all notebooks)

**Q: Can I use my own fMRI data?**
A: Yes! Adapt GLM/encoding code to your BIDS dataset

**Q: Do I need GPU for RL?**
A: Proxy features work on CPU. Full RL training benefits from GPU.

**Q: What if data isn't available?**
A: Slides explain expected results; use for teaching without data

**Q: How to run on cluster?**
A: See shinobi_fmri for batch processing examples

---

Good luck with your presentation! 🎬
