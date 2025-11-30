# Super Mario Bros. fMRI Dataset - Complete Tutorial

This tutorial provides a comprehensive introduction to the Courtois NeuroMod Super Mario Bros. fMRI dataset.

## Tutorial Sections

1. **Dataset Overview** - How is the dataset built ? What does it contain ? What tools are available ?
1. **File Organization and Setup** - Download the CNeuromod data through datalad, understanding BIDS structure and derivatives.
2. **Rich Annotations** - Working with behavioral annotations from game replays  
3. **GLM Analysis** - Linking annotations to BOLD data
4. **MVPA Classification** - Decoding actions from brain activity (voxel-wise)
5. **RL Encoding Model** - Using reward-based variables to predict BOLD

**Focus**: Single session (sub-01, ses-010) for laptop compatibility

## Creating an environment to run the tutorial

To use this tutorial, you need Linux, MacOSX or Windows with WSL, and [Python>3.8](https://www.python.org/downloads/) installed.

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
chmod +x ./install_mario_datasets.sh
./install_mario_datasets.sh
```

(Run this script before the tutorial session, as the data can take up to an hour to download.)