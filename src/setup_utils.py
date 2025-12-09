import sys
import os
import subprocess
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Data requirements per notebook
# Format: 'dataset_name': ['path/to/files', 'another/path']
DATA_REQUIREMENTS = {
    "00_dataset_overview": {
        "mario": ["sub-01/ses-010/func/*.tsv", "sub-01/ses-010/gamelogs/*.bk2"],
        "mario.annotations": ["annotations/sub-01/ses-010"],
        "mario.fmriprep": [] # Just structure
    },
    "01_event_based_analysis": {
        "mario": ["sub-01/ses-010/func/*.tsv"],
        "mario.annotations": ["annotations/sub-01/ses-010"],
        "mario.fmriprep": [
            "sub-01/ses-010/func/*MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
            "sub-01/ses-010/func/*desc-confounds_timeseries.tsv",
            "sub-01/ses-010/func/*desc-brain_mask.nii.gz"
        ],
        "cneuromod.processed": ["smriprep/sub-01/anat"]
    },
    "02_reinforcement_learning": {
        # Note: ROM usually requires manual setup or AWS credentials
        # We fetch what we can (e.g. if the user provided credentials)
        "mario": ["stimuli"], 
    },
    "03_brain_encoding": {
        "mario.fmriprep": [
            "sub-01/ses-010/func/*MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
             "sub-01/ses-010/func/*desc-brain_mask.nii.gz"
        ],
        # Activations are assumed to be in the repo itself (derivatives/) or generated
    }
}

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def run_shell(cmd, check=True):
    """Run a shell command."""
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=check)

def install_system_deps():
    """Install system dependencies (Colab)."""
    if 'google.colab' in sys.modules:
        print("🛠 Installing system dependencies (git-annex)...")
        run_shell("apt-get update -qq")
        run_shell("apt-get install -y git-annex > /dev/null")

def install_python_deps():
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")
    # Install datalad explicitly first as it's needed for data fetching
    run_shell("pip install datalad")
    # Install the rest from requirements
    if os.path.exists("requirements.txt"):
        run_shell("pip install -r requirements.txt")
    else:
        # Fallback minimal install
        run_shell("pip install nilearn pandas numpy matplotlib seaborn scipy gym stable-retro")

def setup_datalad():
    """Configure DataLad."""
    # Ensure datalad is importable
    try:
        import datalad.api as dl
    except ImportError:
        run_shell("pip install datalad")
        import datalad.api as dl
    return dl

def fetch_data(notebook_id, sourcedata_path):
    """Fetch only the data required for the specific notebook."""
    import datalad.api as dl
    
    requirements = DATA_REQUIREMENTS.get(notebook_id)
    if not requirements:
        print(f"ℹ️ No specific data requirements found for {notebook_id}")
        return

    print(f"📥 Fetching data for {notebook_id}...")
    
    # 1. Install Dataset Structure (Lightweight)
    # ------------------------------------------
    datasets = {
        "mario": "https://github.com/courtois-neuromod/mario.git",
        "mario.fmriprep": "https://github.com/courtois-neuromod/mario.fmriprep.git",
        "cneuromod.processed": "https://github.com/courtois-neuromod/cneuromod.processed.git",
        "mario.annotations": "https://github.com/courtois-neuromod/mario.annotations.git"
    }
    
    for name, url in datasets.items():
        ds_path = sourcedata_path / name
        if name in requirements and not ds_path.exists():
            print(f"   Installing dataset structure: {name}...")
            try:
                dl.install(path=str(ds_path), source=url)
            except Exception as e:
                print(f"   Warning: Failed to install {name}: {e}")

    # 2. Get Specific Files (Heavyweight)
    # -----------------------------------
    for dataset_name, file_patterns in requirements.items():
        ds_path = sourcedata_path / dataset_name
        if not ds_path.exists():
            continue
            
        ds = dl.Dataset(str(ds_path))
        
        for pattern in file_patterns:
            print(f"   Downloading {dataset_name}/{pattern}...")
            try:
                # Use glob-like expansion or passing paths
                # Note: datalad get accepts lists of paths. 
                # We interpret the pattern relative to the dataset root.
                
                # Check if it's a glob
                if "*" in pattern:
                    # Simple glob handling via shell or python glob if needed, 
                    # but datalad get often handles paths. 
                    # A robust way is to use `ds.get(path)` directly.
                    # We'll pass the pattern directly to datalad get, it usually handles it.
                    ds.get(pattern)
                else:
                    ds.get(pattern)
            except Exception as e:
                print(f"   Warning: Failed to get {pattern}: {e}")
                
    print("✅ Data download complete.")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

def setup_notebook(notebook_id):
    """
    Main setup function to be called from the notebook. 
    
    Args:
        notebook_id (str): Identifier for the notebook (e.g., '01_event_based_analysis')
    """
    print(f"🚀 Starting setup for {notebook_id}")
    
    # 1. Environment Setup
    install_system_deps()
    install_python_deps()
    
    # 2. Project Paths
    # Assuming we are in the repo root or notebooks/ dir
    cwd = Path.cwd()
    if cwd.name == 'notebooks':
        repo_root = cwd.parent
    else:
        repo_root = cwd
        
    sourcedata_path = repo_root / "sourcedata"
    sourcedata_path.mkdir(exist_ok=True)
    
    # 3. Data Setup
    fetch_data(notebook_id, sourcedata_path)
    
    # 4. Final Configuration
    # Add src to path if not already there
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        
    print("\n✨ Setup finished successfully! You are ready to go.")