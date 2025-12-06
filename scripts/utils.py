"""
General utility functions for Mario fMRI tutorial.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib


def get_project_root():
    """Get the root directory of the tutorial project."""
    return Path(__file__).parent.parent


def get_sourcedata_path():
    """Get the path to sourcedata directory."""
    # Assume sourcedata is in parent of mario.tutorials
    root = get_project_root()
    return root.parent / "sourcedata"


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
    elif not run.startswith('run-'):
        run = f'run-{run}'

    # Build path to annotated events
    events_path = (sourcedata_path / "mario.annotations" / "annotations" /
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
        run = f'run-{run:02d}'
    elif not run.startswith('run-'):
        run = f'run-{run}'

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
        run = f'run-{run:02d}'
    elif not run.startswith('run-'):
        run = f'run-{run}'

    # Build path to brain mask
    mask_path = (sourcedata_path / "mario.fmriprep" / subject / session / "func" /
                 f"{subject}_{session}_task-mario_{run}_space-{space}_desc-brain_mask.nii.gz")

    if not mask_path.exists():
        raise FileNotFoundError(f"Brain mask not found: {mask_path}")

    mask_img = nib.load(mask_path)
    return mask_img


def load_confounds(subject, session, run, sourcedata_path=None):
    """
    Load confounds from fMRIPrep.

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
        Confounds dataframe
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
    elif not run.startswith('run-'):
        run = f'run-{run}'

    # Build path to confounds
    confounds_path = (sourcedata_path / "mario.fmriprep" / subject / session / "func" /
                      f"{subject}_{session}_task-mario_{run}_desc-confounds_timeseries.tsv")

    if not confounds_path.exists():
        raise FileNotFoundError(f"Confounds file not found: {confounds_path}")

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

    # Extract run numbers
    runs = []
    for bold_file in bold_files:
        entities = get_bids_entities(bold_file.name)
        if 'run' in entities and entities['run'] not in runs:
            runs.append(f"run-{entities['run']}")

    return sorted(runs)


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
