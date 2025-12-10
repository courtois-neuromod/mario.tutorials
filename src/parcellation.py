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


# ==============================================================================
# ATLAS LOADING FUNCTIONS
# ==============================================================================

def load_schaefer_atlas(n_rois=400, yeo_networks=7, resolution_mm=2):
    """
    Load Schaefer 2018 functional atlas.

    Parameters
    ----------
    n_rois : int, default=400
        Number of ROIs (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
    yeo_networks : int, default=7
        Number of Yeo networks (7 or 17)
    resolution_mm : int, default=2
        Resolution in mm (1 or 2)

    Returns
    -------
    dict
        Atlas with 'maps' and 'labels' keys
    """
    from nilearn import datasets

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois,
        yeo_networks=yeo_networks,
        resolution_mm=resolution_mm
    )
    return atlas


def load_basc_atlas(scale=444):
    """
    Load BASC 2015 multiscale atlas (Cam-CAN version).

    This atlas is defined in MNI152 space but often aligns well with
    standard fMRI prep outputs.

    Parameters
    ----------
    scale : int
        Scale of the parcellation (e.g., 64, 122, 197, 325, 444)

    Returns
    -------
    dict
        Atlas with 'maps' (path) and 'labels' (generated names)
    """
    from nilearn import datasets
    import nibabel as nib
    import numpy as np

    basc = datasets.fetch_atlas_basc_multiscale_2015(version='sym', resolution=scale)

    # Generate labels since BASC only provides ROI indices
    # We can inspect the image to find valid indices
    img = nib.load(basc['maps'])
    data = img.get_fdata()
    unique_labels = np.unique(data)
    # Remove background (0)
    unique_labels = unique_labels[unique_labels != 0]

    labels = [f"BASC_Region_{int(i)}" for i in unique_labels]

    return {
        'maps': basc['maps'],
        'labels': labels,
        'description': basc['description']
    }


# ==============================================================================
# CACHING FUNCTIONS
# ==============================================================================

def save_complete_results(parcel_bold, all_pca_results, all_encoding_results,
                          train_indices, test_indices, valid_mask, atlas,
                          filepath, pca_dims=None, subject=None, session=None):
    """
    Save ALL processing results to disk to avoid recomputation.

    Parameters
    ----------
    parcel_bold : ndarray
        Parcel-averaged BOLD timeseries
    all_pca_results : dict
        PCA results for each layer
    all_encoding_results : dict
        Encoding results for each layer
    train_indices : ndarray
        Training indices
    test_indices : ndarray
        Test indices
    valid_mask : ndarray
        Valid parcels mask
    atlas : dict
        Atlas with 'labels'
    filepath : str or Path
        Output filepath
    pca_dims : list, optional
        PCA dimensions used
    subject : str, optional
        Subject ID
    session : str, optional
        Session ID
    """
    import pickle
    from pathlib import Path

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Create metadata
    metadata = {
        'pca_dims': pca_dims,
        'subject': subject,
        'session': session,
        'n_parcels': parcel_bold.shape[1],
        'n_timepoints': parcel_bold.shape[0],
        'n_train': len(train_indices),
        'n_test': len(test_indices),
        'n_valid': valid_mask.sum()
    }

    # Package everything
    save_dict = {
        'parcel_bold': parcel_bold,
        'all_pca_results': all_pca_results,
        'all_encoding_results': all_encoding_results,
        'train_indices': train_indices,
        'test_indices': test_indices,
        'valid_mask': valid_mask,
        'atlas_labels': atlas['labels'],  # Save labels, not whole atlas
        'metadata': metadata
    }

    # Save with pickle
    print(f"\n{'='*80}")
    print("SAVING COMPLETE RESULTS TO DISK")
    print(f"{'='*80}")
    print(f"Saving to: {filepath}")

    with open(filepath, 'wb') as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = filepath.stat().st_size / 1024 / 1024
    print(f"\n✓ Complete results saved!")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"{'='*80}\n")


def load_complete_results(filepath):
    """
    Load ALL processing results from disk.

    Parameters
    ----------
    filepath : str or Path
        Path to saved results file

    Returns
    -------
    dict or None
        Saved results dictionary, or None if file doesn't exist
    """
    import pickle
    from pathlib import Path

    filepath = Path(filepath)

    if not filepath.exists():
        return None

    print(f"\n{'='*80}")
    print("LOADING COMPLETE RESULTS FROM CACHE")
    print(f"{'='*80}")
    print(f"Loading from: {filepath}")

    with open(filepath, 'rb') as f:
        results = pickle.load(f)

    metadata = results.get('metadata', {})

    print(f"\nLoaded cached results:")
    print(f"  Subject: {metadata.get('subject')}, Session: {metadata.get('session')}")
    print(f"  Parcels: {metadata.get('n_parcels')}")
    print(f"\n✓ All processing steps skipped - using cached results!")
    print(f"{'='*80}\n")

    return results


def check_complete_results_exist(filepath):
    """
    Check if complete results file exists.

    Parameters
    ----------
    filepath : str or Path
        Path to check

    Returns
    -------
    bool
        True if file exists
    """
    from pathlib import Path
    return Path(filepath).exists()


def extract_parcel_bold(bold_imgs, atlas, confounds_list=None,
                        detrend=True, standardize=True, t_r=1.49, mask_img=None):
    """
    Extract parcel-averaged BOLD timeseries using atlas.

    Parameters
    ----------
    bold_imgs : nibabel.Nifti1Image or list
        BOLD image(s)
    atlas : dict
        Atlas with 'maps' key
    confounds_list : list of DataFrame, optional
        Confounds for each run
    detrend : bool, default=True
        Whether to detrend
    standardize : bool, default=True
        Whether to standardize
    t_r : float, default=1.49
        Repetition time in seconds
    mask_img : nibabel.Nifti1Image, optional
        Brain mask

    Returns
    -------
    ndarray
        Parcel-averaged BOLD timeseries (timepoints × parcels)
    """
    from nilearn.image import resample_to_img
    from nilearn.maskers import NiftiLabelsMasker
    import nibabel as nib
    import numpy as np

    # Ensure list
    if not isinstance(bold_imgs, list):
        bold_imgs = [bold_imgs]

    # Load atlas image if it's a path
    if isinstance(atlas['maps'], str):
        atlas_img = nib.load(atlas['maps'])
    else:
        atlas_img = atlas['maps']

    # Explicitly resample atlas to mask_img if provided
    if mask_img is not None:
        print("  Explicitly resampling atlas to match mask_img affine...")
        atlas_img = resample_to_img(
            source_img=atlas_img,
            target_img=mask_img,
            interpolation='nearest'
        )

    # Create masker
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        mask_img=mask_img,
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
