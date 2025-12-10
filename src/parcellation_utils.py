"""
Simple parcellation utilities for creating random parcellations.

Instead of using functional atlases (Schaefer, BASC) which have uneven
coverage and predefined regions, we create random parcellations that
ensure uniform coverage across the whole brain.
"""

import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from nilearn.regions import Parcellations
from sklearn.cluster import KMeans
from pathlib import Path


def create_random_parcellation(mask_img, n_parcels=400, random_state=42):
    """
    Create a random parcellation by assigning voxels to parcels randomly.

    This ensures uniform coverage across the whole brain, unlike functional
    atlases which may have uneven coverage.

    Parameters
    ----------
    mask_img : Niimg-like
        Brain mask defining which voxels to parcellate
    n_parcels : int
        Number of parcels to create
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    parcellation_img : Nifti1Image
        3D image where each voxel has a parcel label (1 to n_parcels)
    """
    # Load mask
    mask_data = mask_img.get_fdata()
    mask_bool = mask_data > 0

    # Get voxel coordinates
    voxel_coords = np.where(mask_bool)
    n_voxels = len(voxel_coords[0])

    print(f"Creating random parcellation:")
    print(f"  Total voxels: {n_voxels:,}")
    print(f"  Target parcels: {n_parcels}")
    print(f"  Voxels per parcel: ~{n_voxels // n_parcels}")

    # Random assignment
    np.random.seed(random_state)
    labels = np.random.randint(1, n_parcels + 1, size=n_voxels)

    # Create parcellation volume
    parcellation_data = np.zeros_like(mask_data, dtype=np.int32)
    parcellation_data[voxel_coords] = labels

    parcellation_img = nib.Nifti1Image(
        parcellation_data,
        mask_img.affine,
        mask_img.header
    )

    return parcellation_img


def create_kmeans_parcellation(bold_imgs, mask_img, n_parcels=400, random_state=42):
    """
    Create a spatially contiguous parcellation using K-Means clustering.

    This creates more realistic parcels (spatially contiguous) while still
    ensuring whole-brain coverage.

    Parameters
    ----------
    bold_imgs : list of Niimg-like
        BOLD images to use for creating parcellation
    mask_img : Niimg-like
        Brain mask
    n_parcels : int
        Number of parcels
    random_state : int
        Random seed

    Returns
    -------
    parcellation_img : Nifti1Image
        Parcellation image
    """
    from nilearn.regions import Parcellations

    print(f"Creating K-Means parcellation:")
    print(f"  Using {len(bold_imgs)} BOLD runs")
    print(f"  Target parcels: {n_parcels}")
    print(f"  This will take ~1-2 minutes...")

    # Use nilearn's Parcellations
    parcellator = Parcellations(
        method='kmeans',
        n_parcels=n_parcels,
        random_state=random_state,
        mask=mask_img,
        standardize=True,
        verbose=1
    )

    parcellator.fit(bold_imgs)
    parcellation_img = parcellator.labels_img_

    return parcellation_img


def create_ward_parcellation(bold_imgs, mask_img, n_parcels=400, connectivity='auto'):
    """
    Create a parcellation using Ward hierarchical clustering.

    This creates spatially contiguous parcels that respect local correlations.
    Generally produces the most anatomically plausible parcels.

    Parameters
    ----------
    bold_imgs : list of Niimg-like
        BOLD images
    mask_img : Niimg-like
        Brain mask
    n_parcels : int
        Number of parcels
    connectivity : str or array
        Connectivity constraint ('auto' for spatial adjacency)

    Returns
    -------
    parcellation_img : Nifti1Image
        Parcellation image
    """
    from nilearn.regions import Parcellations

    print(f"Creating Ward parcellation:")
    print(f"  Using {len(bold_imgs)} BOLD runs")
    print(f"  Target parcels: {n_parcels}")
    print(f"  Connectivity: {connectivity}")
    print(f"  This will take ~2-3 minutes...")

    parcellator = Parcellations(
        method='ward',
        n_parcels=n_parcels,
        mask=mask_img,
        standardize=True,
        verbose=1
    )

    parcellator.fit(bold_imgs)
    parcellation_img = parcellator.labels_img_

    return parcellation_img


def save_parcellation(parcellation_img, filepath):
    """Save parcellation to disk."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    nib.save(parcellation_img, filepath)
    print(f"Saved parcellation to: {filepath}")


def load_parcellation(filepath):
    """Load parcellation from disk."""
    return nib.load(filepath)


def extract_parcel_bold_from_parcellation(bold_imgs, parcellation_img,
                                          confounds_list=None,
                                          detrend=True, standardize=True,
                                          t_r=1.49):
    """
    Extract parcel-averaged BOLD using a custom parcellation image.

    Parameters
    ----------
    bold_imgs : list of Niimg-like
        BOLD images
    parcellation_img : Niimg-like
        Parcellation image (voxel values = parcel labels)
    confounds_list : list of DataFrames, optional
        Confounds for each run
    detrend : bool
        Whether to detrend
    standardize : bool
        Whether to standardize
    t_r : float
        Repetition time

    Returns
    -------
    parcel_bold : ndarray
        Shape (n_timepoints, n_parcels)
    """
    # Ensure list
    if not isinstance(bold_imgs, list):
        bold_imgs = [bold_imgs]

    # Create masker
    masker = NiftiLabelsMasker(
        labels_img=parcellation_img,
        standardize=standardize,
        detrend=detrend,
        low_pass=None,
        high_pass=None,
        t_r=t_r,
        resampling_target='data'
    )

    # Extract parcels for each run
    parcels_list = []

    for idx, bold_img in enumerate(bold_imgs):
        # Get confounds if provided
        if confounds_list is not None:
            confounds = confounds_list[idx]
        else:
            confounds = None

        # Extract parcel timeseries
        parcels = masker.fit_transform(bold_img, confounds=confounds)
        parcels_list.append(parcels)

    # Concatenate runs
    if len(parcels_list) > 1:
        parcels_concat = np.concatenate(parcels_list, axis=0)
    else:
        parcels_concat = parcels_list[0]

    return parcels_concat


def get_parcel_labels(parcellation_img):
    """
    Get parcel labels for a parcellation image.

    Returns
    -------
    labels : list of str
        Labels like "Parcel_001", "Parcel_002", etc.
    """
    data = parcellation_img.get_fdata()
    unique_vals = np.unique(data)
    unique_vals = unique_vals[unique_vals != 0]  # Remove background
    unique_vals = unique_vals.astype(int)
    unique_vals.sort()

    labels = [f"Parcel_{val:03d}" for val in unique_vals]
    return labels
