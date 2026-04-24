"""
General utility functions for Mario fMRI tutorial.

Button Mappings (NES Super Mario Bros):
---------------------------------------
- A = JUMP (short taps, ~0.3s mean duration)
- B = RUN/FIREBALL (held continuously, ~12s mean duration)
- LEFT/RIGHT = Move left/right
- UP = Enter pipe
- DOWN = Crouch
- START = Pause
- SELECT = (unused in SMB1)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
import json
from collections import defaultdict
import subprocess


def get_project_root():
    """Get the root directory of the tutorial project."""
    return Path(__file__).parent.parent


def get_sourcedata_path():
    """Get the path to sourcedata directory."""
    # Sourcedata is within the mario.tutorials directory
    root = get_project_root()
    return root / "sourcedata"


def get_derivatives_path():
    """Get the path to derivatives directory in tutorial."""
    return get_project_root() / "derivatives"


def get_bids_entities(filename):
    """
    Extract BIDS entities from a filename.

    Parameters
    ----------
    filename : str or Path
        BIDS-formatted filename

    Returns
    -------
    dict
        Dictionary of BIDS entities (sub, ses, task, run, etc.)
    """
    filename = Path(filename).name
    entities = {}

    for part in filename.split('_'):
        if '-' in part:
            key, value = part.split('-', 1)
            # Remove file extensions
            value = value.split('.')[0]
            entities[key] = value

    return entities


def load_events(subject, session, run, sourcedata_path=None):
    """
    Load annotated events file for a given subject/session/run.

    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-01')
    session : str
        Session ID (e.g., 'ses-010')
    run : str or int
        Run number (e.g., 'run-01' or 1)
    sourcedata_path : str or Path, optional
        Path to sourcedata directory

    Returns
    -------
    pd.DataFrame
        Events dataframe
    """
    if sourcedata_path is None:
        sourcedata_path = get_sourcedata_path()
    else:
        sourcedata_path = Path(sourcedata_path)

    # Ensure proper formatting
    if not subject.startswith('sub-'):
        subject = f'sub-{subject}'
    if not session.startswith('ses-'):
        session = f'ses-{session}'
    if isinstance(run, int):
        run = f'run-{run:02d}'
    elif run.startswith('run-'):
        # Extract number and reformat with zero padding
        run_num = run.split('-')[1]
        run = f'run-{int(run_num):02d}'
    else:
        run = f'run-{int(run):02d}'

    # Build path to annotated events (pre-shipped on mario@dev_replays under sub-XX/ses-YYY/func/)
    events_path = (sourcedata_path / "mario" /
                   subject / session / "func" /
                   f"{subject}_{session}_task-mario_{run}_desc-annotated_events.tsv")

    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    events = pd.read_csv(events_path, sep='\t')

    # dev_replays labels the two action buttons as JUMP and RUN/THROW. The tutorial
    # code + contrasts use the NES button names A (jump) and B (run/fireball), and
    # the contrast parser requires valid Python identifiers (so 'RUN/THROW' breaks).
    # Remap back to A / B for compatibility.
    events['trial_type'] = events['trial_type'].replace({'JUMP': 'A', 'RUN/THROW': 'B'})
    return events


def get_bold_path(subject, session, run, sourcedata_path=None, space='MNI152NLin2009cAsym'):
    """
    Get path to preprocessed BOLD data from fMRIPrep.

    Parameters
    ----------
    subject : str
        Subject ID
    session : str
        Session ID
    run : str or int
        Run number
    sourcedata_path : str or Path, optional
        Path to sourcedata directory
    space : str, default='MNI152NLin2009cAsym'
        Template space

    Returns
    -------
    Path
        Path to BOLD NIfTI file
    """
    if sourcedata_path is None:
        sourcedata_path = get_sourcedata_path()
    else:
        sourcedata_path = Path(sourcedata_path)

    # Ensure proper formatting
    if not subject.startswith('sub-'):
        subject = f'sub-{subject}'
    if not session.startswith('ses-'):
        session = f'ses-{session}'
    if isinstance(run, int):
        run = f'run-{run}'
    elif run.startswith('run-'):
        # Extract number (no zero padding for fMRIPrep files)
        run_num = run.split('-')[1]
        run = f'run-{int(run_num)}'
    else:
        run = f'run-{int(run)}'

    # Build path to preprocessed BOLD
    bold_path = (sourcedata_path / "mario.fmriprep" / subject / session / "func" /
                 f"{subject}_{session}_task-mario_{run}_space-{space}_desc-preproc_bold.nii.gz")

    if not bold_path.exists():
        raise FileNotFoundError(f"BOLD file not found: {bold_path}")

    return bold_path


def load_bold(subject, session, run, sourcedata_path=None, space='MNI152NLin2009cAsym'):
    """
    Load preprocessed BOLD data from fMRIPrep.

    Parameters
    ----------
    subject : str
        Subject ID
    session : str
        Session ID
    run : str or int
        Run number
    sourcedata_path : str or Path, optional
        Path to sourcedata directory
    space : str, default='MNI152NLin2009cAsym'
        Template space

    Returns
    -------
    nibabel.Nifti1Image
        BOLD image
    """
    bold_path = get_bold_path(subject, session, run, sourcedata_path, space)
    bold_img = nib.load(bold_path)
    return bold_img


def load_brain_mask(subject, session, run, sourcedata_path=None, space='MNI152NLin2009cAsym'):
    """
    Load brain mask from fMRIPrep.

    Parameters
    ----------
    subject : str
        Subject ID
    session : str
        Session ID
    run : str or int
        Run number
    sourcedata_path : str or Path, optional
        Path to sourcedata directory
    space : str, default='MNI152NLin2009cAsym'
        Template space

    Returns
    -------
    nibabel.Nifti1Image
        Brain mask image
    """
    if sourcedata_path is None:
        sourcedata_path = get_sourcedata_path()
    else:
        sourcedata_path = Path(sourcedata_path)

    # Ensure proper formatting
    if not subject.startswith('sub-'):
        subject = f'sub-{subject}'
    if not session.startswith('ses-'):
        session = f'ses-{session}'
    if isinstance(run, int):
        run = f'run-{run}'
    elif run.startswith('run-'):
        # Extract number (no zero padding for fMRIPrep files)
        run_num = run.split('-')[1]
        run = f'run-{int(run_num)}'
    else:
        run = f'run-{int(run)}'

    # Build path to brain mask
    mask_path = (sourcedata_path / "mario.fmriprep" / subject / session / "func" /
                 f"{subject}_{session}_task-mario_{run}_space-{space}_desc-brain_mask.nii.gz")

    if not mask_path.exists():
        raise FileNotFoundError(f"Brain mask not found: {mask_path}")

    mask_img = nib.load(mask_path)
    return mask_img


def load_confounds(subject, session, run, sourcedata_path=None, strategy='full'):
    """
    Load confounds from fMRIPrep using nilearn's load_confounds.

    Parameters
    ----------
    subject : str
        Subject ID
    session : str
        Session ID
    run : str or int
        Run number
    sourcedata_path : str or Path, optional
        Path to sourcedata directory
    strategy : str, default='full'
        Confound loading strategy (passed to prepare_confounds later)
        This parameter is kept for API consistency but the actual
        strategy is applied in prepare_confounds()

    Returns
    -------
    pd.DataFrame
        Confounds dataframe with all available confounds
    """
    if sourcedata_path is None:
        sourcedata_path = get_sourcedata_path()
    else:
        sourcedata_path = Path(sourcedata_path)

    # Ensure proper formatting
    if not subject.startswith('sub-'):
        subject = f'sub-{subject}'
    if not session.startswith('ses-'):
        session = f'ses-{session}'
    if isinstance(run, int):
        run = f'run-{run}'
    elif run.startswith('run-'):
        # Extract number (no zero padding for fMRIPrep files)
        run_num = run.split('-')[1]
        run = f'run-{int(run_num)}'
    else:
        run = f'run-{int(run)}'

    # Build path to confounds file
    confounds_path = (sourcedata_path / "mario.fmriprep" / subject / session / "func" /
                      f"{subject}_{session}_task-mario_{run}_desc-confounds_timeseries.tsv")

    if not confounds_path.exists():
        raise FileNotFoundError(f"Confounds file not found: {confounds_path}")

    # Load all confounds - selection happens in prepare_confounds()
    confounds = pd.read_csv(confounds_path, sep='\t')

    return confounds


def get_session_runs(subject, session, sourcedata_path=None):
    """
    Get list of run numbers for a given session.

    Parameters
    ----------
    subject : str
        Subject ID
    session : str
        Session ID
    sourcedata_path : str or Path, optional
        Path to sourcedata directory

    Returns
    -------
    list
        List of run numbers (as strings, e.g., ['run-01', 'run-02', ...])
    """
    if sourcedata_path is None:
        sourcedata_path = get_sourcedata_path()
    else:
        sourcedata_path = Path(sourcedata_path)

    # Ensure proper formatting
    if not subject.startswith('sub-'):
        subject = f'sub-{subject}'
    if not session.startswith('ses-'):
        session = f'ses-{session}'

    # Look in fMRIPrep directory
    func_dir = sourcedata_path / "mario.fmriprep" / subject / session / "func"

    if not func_dir.exists():
        raise FileNotFoundError(f"Functional directory not found: {func_dir}")

    # Find all BOLD files
    bold_files = sorted(func_dir.glob(f"{subject}_{session}_task-mario_run-*_*_bold.nii.gz"))

    # Extract run numbers (deduplicate)
    runs = []
    run_nums_seen = set()
    for bold_file in bold_files:
        entities = get_bids_entities(bold_file.name)
        if 'run' in entities and entities['run'] not in run_nums_seen:
            run_nums_seen.add(entities['run'])
            runs.append(f"run-{entities['run']}")

    return sorted(runs, key=lambda x: int(x.split('-')[1]))


def load_lowlevel_confounds(subject, session, run, sourcedata_path=None):
    """
    Load low-level confounds (luminance, optical flow, audio) from mario/sub-XX/ses-YYY/gamelogs/.

    Parameters
    ----------
    subject : str
        Subject ID
    session : str
        Session ID
    run : str or int
        Run number
    sourcedata_path : str or Path, optional
        Path to sourcedata directory

    Returns
    -------
    pd.DataFrame
        Low-level confounds dataframe with columns: luminance, optical_flow, audio_envelope
    """
    if sourcedata_path is None:
        sourcedata_path = get_sourcedata_path()
    else:
        sourcedata_path = Path(sourcedata_path)

    # Ensure proper formatting
    if not subject.startswith('sub-'):
        subject = f'sub-{subject}'
    if not session.startswith('ses-'):
        session = f'ses-{session}'

    # Get run number for file matching
    if isinstance(run, int):
        run_num = run
    elif run.startswith('run-'):
        run_num = int(run.split('-')[1])
    else:
        run_num = int(run)

    # Look for lowlevel files in mario/sub-XX/ses-YYY/gamelogs/ (branch: dev_replays)
    confs_dir = sourcedata_path / "mario" / subject / session / "gamelogs"

    if not confs_dir.exists():
        raise FileNotFoundError(f"Gamelogs directory not found: {confs_dir}")

    # Find matching lowlevel file(s) for this run
    # Files are named: sub-01_ses-010_task-mario_level-w1l1_rep-000_lowlevel.npy
    conf_files = sorted(confs_dir.glob(f"{subject}_{session}_task-mario_*_lowlevel.npy"))

    if len(conf_files) == 0:
        raise FileNotFoundError(f"No confounds files found in {confs_dir}")

    # Load all confounds and concatenate (one run may span multiple levels/replays)
    all_luminance = []
    all_optical_flow = []
    all_audio = []

    for conf_file in conf_files:
        confs = np.load(conf_file, allow_pickle=True).item()

        all_luminance.append(confs['luminance'])
        all_optical_flow.append(np.array(confs['optical_flow']))
        all_audio.append(confs['audio_envelope'])

    # Concatenate all segments
    luminance = np.concatenate(all_luminance)
    optical_flow = np.concatenate(all_optical_flow)
    audio_envelope = np.concatenate(all_audio)

    # Create dataframe
    lowlevel_df = pd.DataFrame({
        'luminance': luminance,
        'optical_flow': optical_flow,
        'audio_envelope': audio_envelope
    })

    return lowlevel_df


def create_output_dir(subject, session, analysis_type, derivatives_path=None):
    """
    Create output directory for a given analysis.

    Parameters
    ----------
    subject : str
        Subject ID
    session : str
        Session ID
    analysis_type : str
        Type of analysis ('glm_tutorial', 'rl_agent', 'encoding', etc.)
    derivatives_path : str or Path, optional
        Path to derivatives directory

    Returns
    -------
    Path
        Path to output directory
    """
    if derivatives_path is None:
        derivatives_path = get_derivatives_path()
    else:
        derivatives_path = Path(derivatives_path)

    # Ensure proper formatting
    if not subject.startswith('sub-'):
        subject = f'sub-{subject}'
    if not session.startswith('ses-'):
        session = f'ses-{session}'

    # Create directory structure
    output_dir = derivatives_path / analysis_type / subject / session / "func"
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def save_stat_map(img, subject, session, model_name, contrast_name,
                  stat_type='zmap', derivatives_path=None):
    """
    Save a statistical map with proper BIDS naming.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        Statistical map to save
    subject : str
        Subject ID
    session : str
        Session ID
    model_name : str
        Name of the GLM model (e.g., 'simple_actions', 'full_actions')
    contrast_name : str
        Name of the contrast (e.g., 'LEFT', 'LEFT-RIGHT')
    stat_type : str, default='zmap'
        Type of statistic ('zmap', 'beta', 'tmap', etc.)
    derivatives_path : str or Path, optional
        Path to derivatives directory

    Returns
    -------
    Path
        Path to saved file
    """
    output_dir = create_output_dir(subject, session, 'glm_tutorial', derivatives_path)

    # Create filename
    filename = (f"{subject}_{session}_task-mario_"
                f"model-{model_name}_contrast-{contrast_name}_"
                f"stat-{stat_type}.nii.gz")

    filepath = output_dir / filename
    nib.save(img, filepath)

    return filepath


def load_replay_metadata(sourcedata_path=None):
    """
    Load all replay metadata from *_summary.json sidecars in mario/sub-*/ses-*/gamelogs/.

    Parameters
    ----------
    sourcedata_path : str or Path, optional
        Path to sourcedata directory

    Returns
    -------
    pd.DataFrame
        Dataframe with all replay metadata including:
        - Subject, World, Level, Duration, Cleared, Phase
        - ScoreGained, Lives_lost, Hits_taken, Enemies_killed
        - Powerups_collected, CoinsGained, etc.
    """
    if sourcedata_path is None:
        sourcedata_path = get_sourcedata_path()
    else:
        sourcedata_path = Path(sourcedata_path)

    mario_path = sourcedata_path / "mario"

    if not mario_path.exists():
        raise FileNotFoundError(f"Mario dataset directory not found: {mario_path}")

    # Find all summary JSON files in gamelogs/
    json_files = sorted(mario_path.glob("sub-*/ses-*/gamelogs/*_summary.json"))

    if len(json_files) == 0:
        raise FileNotFoundError(f"No *_summary.json files found in {mario_path}/sub-*/ses-*/gamelogs/")

    # Load all JSONs
    all_data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            all_data.append(data)

    # Create dataframe
    df = pd.DataFrame(all_data)

    # Derive a boolean `Cleared` column from the `Outcome` string so that
    # existing statistics / visualisation helpers keep working.
    # Outcome values on dev_replays: 'cleared', 'failed/timeout', 'failed/fall', 'failed/killed'.
    if 'Outcome' in df.columns and 'Cleared' not in df.columns:
        df['Cleared'] = df['Outcome'] == 'cleared'

    return df


def compute_dataset_statistics(replay_data=None, sourcedata_path=None):
    """
    Compute and display global statistics across the entire Mario dataset.

    Parameters
    ----------
    replay_data : pd.DataFrame, optional
        Replay metadata dataframe. If None, will load from sourcedata_path.
    sourcedata_path : str or Path, optional
        Path to sourcedata directory
    """
    print("Loading dataset statistics...\n")

    if replay_data is None:
        df = load_replay_metadata(sourcedata_path)
    else:
        df = replay_data

    # Compute per-subject statistics
    subject_stats = []
    for subject in sorted(df['Subject'].unique()):
        sub_df = df[df['Subject'] == subject]

        # Extract unique sessions for this subject
        unique_sessions = sub_df['Bk2File'].str.extract(r'ses-(\d+)')[0].unique()

        stats = {
            'Subject': f'sub-{subject}',
            'Total Repetitions': len(sub_df),
            'Successful Completions': int(sub_df['Cleared'].sum()),
            'Failures': int((~sub_df['Cleared']).sum()),
            'Success Rate (%)': round(100 * sub_df['Cleared'].mean(), 1),
            'Total Duration (min)': round(sub_df['Duration'].sum() / 60, 1),
            'Sessions': len(unique_sessions)
        }
        subject_stats.append(stats)

    per_subject = pd.DataFrame(subject_stats)

    # Get unique levels
    unique_levels = sorted(df['LevelFullName'].unique())

    # Get levels grouped by world
    levels_by_world = defaultdict(list)
    for level in unique_levels:
        world = int(level[1])  # e.g., 'w1l1' -> 1
        levels_by_world[world].append(level)

    # Compute total statistics
    total_stats = {
        'Total Repetitions': len(df),
        'Total Successful Completions': int(df['Cleared'].sum()),
        'Total Failures': int((~df['Cleared']).sum()),
        'Overall Success Rate (%)': round(100 * df['Cleared'].mean(), 1),
        'Total Playtime (hours)': round(df['Duration'].sum() / 3600, 1),
        'Discovery Phase Reps': int((df['Phase'] == 'discovery').sum()),
        'Practice Phase Reps': int((df['Phase'] == 'practice').sum()),
    }

    # Display per-subject statistics
    print("=" * 80)
    print("PER-SUBJECT STATISTICS")
    print("=" * 80)
    print(per_subject.to_string(index=False))

    # Display total statistics
    print("\n" + "=" * 80)
    print("DATASET-WIDE STATISTICS")
    print("=" * 80)
    for key, value in total_stats.items():
        print(f"  {key:.<45} {value}")

    # Display phase breakdown
    print("\n" + "=" * 80)
    print("EXPERIMENTAL DESIGN")
    print("=" * 80)
    discovery_pct = 100 * total_stats['Discovery Phase Reps'] / total_stats['Total Repetitions']
    practice_pct = 100 * total_stats['Practice Phase Reps'] / total_stats['Total Repetitions']
    print(f"  Discovery Phase: {total_stats['Discovery Phase Reps']:>5} reps ({discovery_pct:.1f}%)")
    print(f"  Practice Phase:  {total_stats['Practice Phase Reps']:>5} reps ({practice_pct:.1f}%)")

    # Display levels
    print("\n" + "=" * 80)
    print("LEVELS PLAYED (22 total)")
    print("=" * 80)
    for world, levels in sorted(levels_by_world.items()):
        levels_str = ", ".join(levels)
        print(f"  World {world}: {levels_str}")

    print("\n✓ Dataset statistics loaded successfully!")


# ==============================================================================
# ENVIRONMENT SETUP FUNCTIONS
# ==============================================================================

def setup_colab_environment(IN_COLAB=False):
    """
    Make sure repo and paths are properly setup.
    """
    import sys
    from pathlib import Path
    import subprocess
    import os


    print("🚀 Detected Google Colab")
    PROJECT_PATH = Path("/content/mario.tutorials")

    # Install git-annex using datalad-installer (required for DataLad to work)
    print("📦 Installing git-annex for DataLad support...")
    
    # Install the missing 'netbase' dependency
    subprocess.run(["sudo", "apt-get", "update"], check=True, capture_output=True)
    subprocess.run(["sudo", "apt-get", "install", "-y", "netbase"], check=True)
    
    # Install git-annex
    subprocess.run(["pip", "install", "-q", "datalad", "datalad-installer"], check=True)
    subprocess.run(["datalad-installer", "--sudo", "ok", "git-annex", "-m", "datalad/git-annex:release"], check=True)
    
    # Configure git user (DataLad requires this)
    subprocess.run(["git", "config", "--global", "user.email", "colab@example.com"], check=True)
    subprocess.run(["git", "config", "--global", "user.name", "Colab User"], check=True)
    
    print("✓ git-annex installed and configured")

    # Change to project directory
    os.chdir(PROJECT_PATH)
    sys.path.insert(0, str(PROJECT_PATH / "src"))
    return


def install_dependencies(requirements_file):
    """
    Install Python dependencies from requirements file.

    Parameters
    ----------
    requirements_file : str or Path
        Path to requirements file (relative to project root)
    """

    print(f"📦 Installing dependencies from {requirements_file}...")
    subprocess.run(["pip", "install", "-q", "-r", str(requirements_file)], check=True)
    print("  ✓ Dependencies installed")


def setup_datalad_datasets(sourcedata_path, install_only=False):
    """
    Setup DataLad datasets with fallback instructions.

    Parameters
    ----------
    sourcedata_path : Path
        Path to sourcedata directory
    install_only : bool, default=False
        If True, only install dataset structure without fetching data

    Returns
    -------
    bool
        True if successful, False if DataLad failed
    """
    from pathlib import Path
    import subprocess

    sourcedata_path = Path(sourcedata_path)
    sourcedata_path.mkdir(parents=True, exist_ok=True)

    # Configure git for DataLad
    print("📥 Setting up DataLad datasets...")
    subprocess.run(["git", "config", "--global", "user.email", "notebook@example.com"], check=False)
    subprocess.run(["git", "config", "--global", "user.name", "Notebook User"], check=False)

    # Dataset URLs. `mario` on the `dev_replays` branch ships annotations
    # (func/*_desc-annotated_events.tsv) and replay derivatives (gamelogs/*) in-tree,
    # so we no longer install mario.annotations / mario.replays separately.
    datasets = {
        "mario": "https://github.com/courtois-neuromod/mario.git",
        "mario.fmriprep": "https://github.com/courtois-neuromod/mario.fmriprep.git",
        "cneuromod.processed": "https://github.com/courtois-neuromod/cneuromod.processed.git",
    }

    success = True
    for name, url in datasets.items():
        ds_path = sourcedata_path / name

        if ds_path.exists():
            print(f"  ✓ {name} already installed")
            if name == "mario":
                subprocess.run(["git", "-C", str(ds_path), "checkout", "dev_replays"], check=False)
            continue

        print(f"  Installing {name}...")
        try:
            # Try datalad install first
            result = subprocess.run(
                ["datalad", "install", "-s", url, str(ds_path)],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                # Fallback to git clone
                print(f"    DataLad install failed, trying git clone...")
                result = subprocess.run(
                    ["git", "clone", url, str(ds_path)],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    print(f"    ⚠️  Failed to install {name}")
                    success = False
                else:
                    print(f"    ✓ Cloned with git")
            else:
                print(f"    ✓ Installed with DataLad")

            # Pin mario to the dev_replays branch (annotations + gamelogs live there)
            if name == "mario" and ds_path.exists():
                subprocess.run(["git", "-C", str(ds_path), "checkout", "dev_replays"], check=False)

        except Exception as e:
            print(f"    ⚠️  Error installing {name}: {e}")
            success = False

    if not success:
        print("\n⚠️  Some datasets failed to install via DataLad")
        print("\nAlternative download options:")
        print("1. Manual download: Visit https://docs.cneuromod.ca/")
        print("2. Direct git clone for each dataset")
        print("3. Contact cneuromod team for access")
        return False

    print("  ✓ All datasets installed")
    return True


def download_stimuli(target_path=None):
    """
    Download Mario stimuli (ROM + save states). Prefers fetching via datalad
    from the `mario` dataset's `stimuli` subdataset (public conp-ria remote),
    and falls back to a Google Drive tarball if datalad isn't usable (e.g.
    Colab without the mario dataset installed).

    Parameters
    ----------
    target_path : Path, optional
        Target path for stimuli folder. If None, uses sourcedata/mario/stimuli
    """
    from pathlib import Path
    import subprocess

    if target_path is None:
        target_path = get_sourcedata_path() / "mario" / "stimuli"
    else:
        target_path = Path(target_path)

    rom_file = target_path / "SuperMarioBros-Nes" / "rom.nes"
    if rom_file.exists():
        print("✓ Stimuli ROM already present")
        return True

    # Preferred path: datalad get on the stimuli subdataset of the mario repo.
    mario_path = target_path.parent  # sourcedata/mario
    if (mario_path / ".datalad").exists() or (mario_path / ".git").exists():
        print("📥 Fetching mario/stimuli via datalad get...")
        try:
            import datalad.api as dl
            # Ensure the submodule is initialised before getting its content.
            # `dl.get` on a path inside an un-initialised submodule will fetch both.
            dl.get(path=str(target_path), dataset=str(mario_path))
            if rom_file.exists():
                print("  ✓ Stimuli downloaded via datalad")
                return True
            print("  ⚠️  datalad get completed but ROM still missing; trying Google Drive fallback...")
        except Exception as e:
            print(f"  ⚠️  datalad get failed ({e}); falling back to Google Drive...")

    print("📥 Downloading stimuli from Google Drive...")

    # Ensure gdown is installed
    subprocess.run(["pip", "install", "-q", "gdown"], check=False)

    # Google Drive file ID
    file_id = '17zaL1-6OOd3xxj4EIzCI6-o6sghwx7Qi'
    url = f'https://drive.google.com/uc?id={file_id}'

    # Create parent directory
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Download tar.gz file
    output_tar = target_path.parent / "stimuli.tar.gz"

    try:
        import gdown
        print(f"  Downloading to {output_tar}...")
        gdown.download(url, str(output_tar), quiet=False)

        print("  Extracting...")
        import tarfile
        with tarfile.open(output_tar, 'r:gz') as tar_ref:
            tar_ref.extractall(target_path.parent)

        # Cleanup
        import os
        os.remove(output_tar)

        print("  ✓ Stimuli downloaded and extracted")
        return True

    except Exception as e:
        print(f"  ⚠️  Failed to download stimuli: {e}")
        print(f"\nManual download:")
        print(f"1. Visit: https://drive.google.com/file/d/{file_id}/view")
        print(f"2. Download and extract to: {target_path}")
        return False


def _ensure_mario_installed(mario_path):
    """Install mario dataset (recursively, to initialise the stimuli submodule) and
    pin to the dev_replays branch."""
    import subprocess
    if not mario_path.exists():
        print("  Installing mario dataset (recursive — includes stimuli submodule)...")
        result = subprocess.run(
            ["datalad", "install", "--recursive",
             "-s", "https://github.com/courtois-neuromod/mario", str(mario_path)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ⚠️  DataLad install failed: {result.stderr}")
            return False
    # Pin to dev_replays (where annotations + gamelogs live)
    subprocess.run(["git", "-C", str(mario_path), "checkout", "dev_replays"], check=False,
                   capture_output=True)
    return True


def download_mario_replays(sourcedata_path=None):
    """
    Download replay summary JSON files from mario/sub-*/ses-*/gamelogs/ (branch: dev_replays).

    Parameters
    ----------
    sourcedata_path : Path, optional
        Path to sourcedata directory

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    from pathlib import Path
    import subprocess
    import os

    if sourcedata_path is None:
        sourcedata_path = get_sourcedata_path()
    else:
        sourcedata_path = Path(sourcedata_path)

    # Use absolute paths
    sourcedata_path = sourcedata_path.resolve()
    mario_path = sourcedata_path / "mario"

    # Check if data already exists
    replays_exists = mario_path.exists() and len(list(mario_path.glob("sub-*/ses-*/gamelogs/*_summary.json"))) > 0

    if replays_exists:
        print("✓ Replay summary metadata already downloaded!")
        return True

    print("📥 Downloading replay summary metadata from mario@dev_replays...")

    if not _ensure_mario_installed(mario_path):
        return False

    # Find all summary JSON files
    json_files = list(mario_path.glob("sub-*/ses-*/gamelogs/*_summary.json"))

    if not json_files:
        print("  ⚠️  No *_summary.json files found in dataset structure")
        return False

    print(f"  Found {len(json_files)} summary files to download...")

    # Get relative paths for datalad
    file_paths = [str(f.relative_to(mario_path)) for f in json_files]

    # Change to dataset directory
    original_dir = os.getcwd()
    os.chdir(mario_path)

    try:
        # Download in batches to avoid command line length limits
        batch_size = 100
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i+batch_size]
            result = subprocess.run(
                ["datalad", "get"] + batch,
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"  ⚠️  Batch download failed: {result.stderr}")
                os.chdir(original_dir)
                return False
            print(f"  Downloaded {min(i+batch_size, len(file_paths))}/{len(file_paths)} files...")

        os.chdir(original_dir)
        print("  ✓ All replay summary metadata downloaded!")
        return True

    except Exception as e:
        os.chdir(original_dir)
        print(f"  ⚠️  Error downloading replays: {e}")
        return False


def download_mario_annotations(subject, session, sourcedata_path=None):
    """
    Download annotated events TSV files for a specific subject/session
    from mario/sub-XX/ses-YYY/func/ (branch: dev_replays).

    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-01')
    session : str
        Session ID (e.g., 'ses-010')
    sourcedata_path : Path, optional
        Path to sourcedata directory

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    from pathlib import Path
    import subprocess
    import os

    if sourcedata_path is None:
        sourcedata_path = get_sourcedata_path()
    else:
        sourcedata_path = Path(sourcedata_path)

    # Ensure proper formatting
    if not subject.startswith('sub-'):
        subject = f'sub-{subject}'
    if not session.startswith('ses-'):
        session = f'ses-{session}'

    # Use absolute paths
    sourcedata_path = sourcedata_path.resolve()
    mario_path = sourcedata_path / "mario"

    # Check if data already exists for this subject/session
    session_path = mario_path / subject / session / "func"
    annotations_exist = session_path.exists() and len(list(session_path.glob(f"{subject}_{session}_*_desc-annotated_events.tsv"))) > 0

    if annotations_exist:
        print(f"✓ Annotated events already downloaded for {subject} {session}!")
        return True

    print(f"📥 Downloading annotated events for {subject} {session}...")

    if not _ensure_mario_installed(mario_path):
        return False

    # Find all TSV files for this subject/session
    tsv_files = list(mario_path.glob(f"{subject}/{session}/func/{subject}_{session}_task-mario_run-*_desc-annotated_events.tsv"))

    if not tsv_files:
        print(f"  ⚠️  No annotated event files found for {subject} {session}")
        return False

    print(f"  Found {len(tsv_files)} annotated event files to download...")

    # Get relative paths for datalad
    file_paths = [str(f.relative_to(mario_path)) for f in tsv_files]

    # Change to dataset directory
    original_dir = os.getcwd()
    os.chdir(mario_path)

    try:
        # Download all TSV files
        result = subprocess.run(
            ["datalad", "get"] + file_paths,
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ⚠️  Download failed: {result.stderr}")
            os.chdir(original_dir)
            return False

        os.chdir(original_dir)
        print(f"  ✓ Downloaded {len(file_paths)} files!")
        return True

    except Exception as e:
        os.chdir(original_dir)
        print(f"  ⚠️  Error downloading annotations: {e}")
        return False


def verify_data(subject, session, sourcedata_path=None, check_bold=False):
    """
    Verify that required data exists for a subject/session.

    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-01')
    session : str
        Session ID (e.g., 'ses-010')
    sourcedata_path : Path, optional
        Path to sourcedata directory
    check_bold : bool, default=False
        If True, also check for BOLD data (requires full download)

    Returns
    -------
    bool
        True if data exists, False otherwise
    """
    from pathlib import Path

    if sourcedata_path is None:
        sourcedata_path = get_sourcedata_path()
    else:
        sourcedata_path = Path(sourcedata_path)

    # Ensure proper formatting
    if not subject.startswith('sub-'):
        subject = f'sub-{subject}'
    if not session.startswith('ses-'):
        session = f'ses-{session}'

    print(f"📊 Verifying data for {subject} {session}...")

    # Check for annotations (on mario@dev_replays, under sub-XX/ses-YYY/func/)
    annot_dir = sourcedata_path / "mario" / subject / session / "func"
    if not annot_dir.exists():
        print(f"  ⚠️  Annotations not found: {annot_dir}")
        return False

    annot_files = list(annot_dir.glob("*_events.tsv"))
    if len(annot_files) == 0:
        print(f"  ⚠️  No event files found in {annot_dir}")
        return False

    print(f"  ✓ Found {len(annot_files)} event files")

    # Check for BOLD if requested
    if check_bold:
        bold_dir = sourcedata_path / "mario.fmriprep" / subject / session / "func"
        if not bold_dir.exists():
            print(f"  ⚠️  BOLD data not found: {bold_dir}")
            return False

        bold_files = list(bold_dir.glob("*_bold.nii.gz"))
        if len(bold_files) == 0:
            print(f"  ⚠️  No BOLD files found in {bold_dir}")
            return False

        print(f"  ✓ Found {len(bold_files)} BOLD files")

    print(f"  ✓ Data verified for {subject} {session}")
    return True


def download_cneuromod_data(
    dataset_name,
    subject=None,
    session=None,
    pattern=None,
    sourcedata_path=None
):
    """
    Download data from the CNeuromod datalad repositories with flexible filtering.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., 'mario', 'mario.fmriprep', 'cneuromod.processed').
        The 'mario' dataset is pinned to the `dev_replays` branch so that
        *_desc-annotated_events.tsv (in func/) and gamelogs/*  (summary / variables /
        lowlevel / recording) are available.
    subject : str or None
        Subject ID (e.g., 'sub-01'). If None, download for all subjects.
    session : str or None
        Session ID (e.g., 'ses-010'). If None, download for all sessions.
    pattern : str or None
        Filename pattern to match (e.g., '*events.tsv', '*.bk2', '*_summary.json').
        If None, download everything.
    sourcedata_path : Path or None
        Path to sourcedata directory. If None, uses get_sourcedata_path().

    Returns
    -------
    Path
        Path to the installed dataset

    Examples
    --------
    # Download all annotated events for sub-01, ses-010
    download_cneuromod_data('mario', 'sub-01', 'ses-010', '*_desc-annotated_events.tsv')

    # Download all replay .bk2 files for sub-01 (all sessions)
    download_cneuromod_data('mario', 'sub-01', None, '*.bk2')

    # Download all replay summary files for all subjects
    download_cneuromod_data('mario', None, None, '*_summary.json')

    # Install entire dataset
    download_cneuromod_data('mario.fmriprep')
    """
    import datalad.api as dl
    import subprocess
    from pathlib import Path

    if sourcedata_path is None:
        sourcedata_path = get_sourcedata_path()

    sourcedata_path = Path(sourcedata_path)
    sourcedata_path.mkdir(exist_ok=True, parents=True)

    dataset_path = sourcedata_path / dataset_name

    # Install dataset (idempotent: dl.install raises if path already has a dataset,
    # so we guard on existence). For mario, install recursively so the stimuli
    # submodule is initialised (its content is fetched separately by download_stimuli()).
    if not dataset_path.exists():
        print(f"📥 Installing {dataset_name}...")
        dl.install(
            source=f"https://github.com/courtois-neuromod/{dataset_name}",
            path=str(dataset_path),
            recursive=(dataset_name == "mario"),
        )
        print(f"✓ Installed to {dataset_path}")
    else:
        print(f"✓ {dataset_name} already installed at {dataset_path}")

    # mario needs the dev_replays branch (annotations in func/, replay derivatives in gamelogs/)
    if dataset_name == "mario":
        subprocess.run(
            ["git", "-C", str(dataset_path), "checkout", "dev_replays"],
            check=False,
            capture_output=True,
        )
    
    # Build search path based on subject/session
    search_paths = []
    
    if subject is None and session is None:
        # Search entire dataset
        search_paths = [dataset_path]
    elif subject is not None and session is None:
        # Search all sessions for this subject
        subject_path = dataset_path / subject
        if subject_path.exists():
            search_paths = [subject_path]
        else:
            print(f"⚠️  Subject {subject} not found in {dataset_name}")
            return dataset_path
    elif subject is not None and session is not None:
        # Search specific subject/session
        session_path = dataset_path / subject / session
        if session_path.exists():
            search_paths = [session_path]
        else:
            print(f"⚠️  {subject}/{session} not found in {dataset_name}")
            return dataset_path
    else:
        # session provided but not subject - not supported
        raise ValueError("Cannot specify session without subject")
    
    # Find matching files
    files_to_get = []
    for search_path in search_paths:
        if pattern is None:
            # Get everything recursively
            files_to_get.extend(search_path.rglob('*'))
        else:
            # Find files matching pattern at any depth
            files_to_get.extend(search_path.rglob(pattern))
    
    # Filter out directories, keep only files
    files_to_get = [f for f in files_to_get if f.is_file() or f.is_symlink()]
    
    if files_to_get:
        print(f"📥 Downloading {len(files_to_get)} files matching pattern '{pattern}'...")
        # Tolerate per-file failures (e.g. annexed files whose content isn't on any
        # reachable remote). DataLad raises IncompleteResultsError if any item fails;
        # we catch it, report the count of actually-downloaded files, and move on.
        try:
            from datalad.support.exceptions import IncompleteResultsError
        except ImportError:
            IncompleteResultsError = Exception
        try:
            dl.get(path=[str(f) for f in files_to_get])
            print(f"✓ Downloaded {len(files_to_get)} files")
        except IncompleteResultsError as exc:
            results = getattr(exc, 'failed', []) or []
            n_failed = len(results)
            n_ok = len(files_to_get) - n_failed
            print(f"⚠️  Downloaded {n_ok}/{len(files_to_get)} files; {n_failed} could not be fetched")
            if n_failed and n_failed <= 5:
                for r in results:
                    print(f"   - {r.get('path', '?')}")
    else:
        print(f"ℹ️  No files found matching pattern '{pattern}'")
    
    return dataset_path
