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

    # Build path to annotated events
    events_path = (sourcedata_path / "mario.annotations" /
                   subject / session / "func" /
                   f"{subject}_{session}_task-mario_{run}_desc-annotated_events.tsv")

    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    events = pd.read_csv(events_path, sep='\t')
    return events


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
    Load low-level confounds (luminance, optical flow, audio) from mario.replays.

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

    # Look for confounds files in mario.replays
    confs_dir = sourcedata_path / "mario.replays" / subject / session / "beh" / "confs"

    if not confs_dir.exists():
        raise FileNotFoundError(f"Low-level confounds directory not found: {confs_dir}")

    # Find matching confounds file(s) for this run
    # Files are named: sub-01_ses-010_task-mario_level-w1l1_rep-000_confs.npy
    conf_files = sorted(confs_dir.glob(f"{subject}_{session}_task-mario_*.npy"))

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
    Load all replay metadata from JSON sidecars in mario.replays dataset.

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

    replays_path = sourcedata_path / "mario.replays"

    if not replays_path.exists():
        raise FileNotFoundError(f"Replays directory not found: {replays_path}")

    # Find all JSON files
    json_files = sorted(replays_path.glob("*/*/beh/infos/*.json"))

    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON metadata files found in {replays_path}")

    # Load all JSONs
    all_data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            all_data.append(data)

    # Create dataframe
    df = pd.DataFrame(all_data)

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
