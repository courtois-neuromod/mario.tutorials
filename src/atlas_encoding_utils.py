"""
Atlas-based encoding utilities for computational efficiency.

Instead of fitting 213k voxels independently, we:
1. Use Schaefer atlas to parcellate brain into ~400 regions
2. Average BOLD signal within each parcel
3. Fit ridge regression on parcels (much faster!)
"""

import numpy as np
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pickle
from pathlib import Path


def load_schaefer_atlas(n_rois=400, yeo_networks=7, resolution_mm=2):
    """
    Load Schaefer 2018 functional atlas.

    Parameters
    ----------
    n_rois : int
        Number of parcels (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
        Default: 400 (recommended balance)
    yeo_networks : int
        Number of Yeo networks (7 or 17)
    resolution_mm : int
        Spatial resolution in mm (1 or 2)

    Returns
    -------
    dict
        Atlas with 'maps' (NIfTI image) and 'labels' (region names)
    """
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois,
        yeo_networks=yeo_networks,
        resolution_mm=resolution_mm
    )

    return atlas


def save_complete_results(parcel_bold, all_pca_results, all_encoding_results,
                          train_indices, test_indices, valid_mask, atlas,
                          filepath, pca_dims=None, subject=None, session=None):
    """
    Save ALL processing results to disk to avoid recomputation.

    This saves the entire workflow: BOLD extraction, PCA, encoding.

    Parameters
    ----------
    parcel_bold : np.ndarray
        Parcel-averaged BOLD data
    all_pca_results : dict
        PCA results for all dimensions
    all_encoding_results : dict
        Encoding results for all PCA dimensions
    train_indices : np.ndarray
        Training indices
    test_indices : np.ndarray
        Test indices
    valid_mask : np.ndarray
        Valid timepoint mask
    atlas : dict
        Schaefer atlas
    filepath : str or Path
        Path to save results
    pca_dims : list, optional
        PCA dimensions used
    subject : str, optional
        Subject ID
    session : str, optional
        Session ID
    """
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
    print("This includes:")
    print(f"  - Parcel BOLD data ({parcel_bold.shape})")
    print(f"  - PCA results for {len(all_pca_results)} dimensions")
    print(f"  - Encoding results for {len(all_encoding_results)} dimensions")
    print(f"  - Train/test splits and valid mask")

    with open(filepath, 'wb') as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = filepath.stat().st_size / 1024 / 1024
    print(f"\n✓ Complete results saved!")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Next time you run this notebook, all processing will be skipped!")
    print(f"{'='*80}\n")


def load_complete_results(filepath):
    """
    Load ALL processing results from disk.

    Parameters
    ----------
    filepath : str or Path
        Path to saved results

    Returns
    -------
    dict or None
        Dictionary with all results, or None if file doesn't exist
        Keys: 'parcel_bold', 'all_pca_results', 'all_encoding_results',
              'train_indices', 'test_indices', 'valid_mask', 'atlas_labels', 'metadata'
    """
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
    print(f"  Timepoints: {metadata.get('n_timepoints')}")
    print(f"  PCA dimensions: {metadata.get('pca_dims')}")
    print(f"  Train samples: {metadata.get('n_train')}, Test samples: {metadata.get('n_test')}")
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
        True if file exists, False otherwise
    """
    return Path(filepath).exists()


def extract_parcel_bold(bold_imgs, atlas, confounds_list=None,
                        detrend=True, standardize=True, t_r=1.49):
    """
    Extract parcel-averaged BOLD timeseries using atlas.

    This replaces voxel-wise BOLD (n_timepoints × 213k voxels) with
    parcel-wise BOLD (n_timepoints × 400 parcels), dramatically reducing
    computation while improving SNR through averaging.

    Parameters
    ----------
    bold_imgs : list of nibabel images or paths
        BOLD images for each run
    atlas : dict
        Schaefer atlas from load_schaefer_atlas()
    confounds_list : list of DataFrames or None
        Confounds for each run (motion, WM, CSF, etc.)
    detrend : bool
        Whether to detrend
    standardize : bool
        Whether to z-score each parcel
    t_r : float
        Repetition time

    Returns
    -------
    np.ndarray
        Parcel-averaged BOLD (n_timepoints, n_parcels)
    """
    # Ensure list
    if not isinstance(bold_imgs, list):
        bold_imgs = [bold_imgs]

    # Create masker
    masker = NiftiLabelsMasker(
        labels_img=atlas['maps'],
        standardize=standardize,
        detrend=detrend,
        low_pass=None,
        high_pass=None,
        t_r=t_r
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


def fit_atlas_encoding_per_layer(layer_activations, parcel_bold, atlas,
                                  train_indices, test_indices, alphas=None,
                                  valid_mask=None):
    """
    Fit encoding models using atlas parcels.

    This is ~500x faster than voxel-wise encoding and often achieves
    better R² due to noise reduction from averaging.

    Parameters
    ----------
    layer_activations : dict
        Layer name → activation array
    parcel_bold : np.ndarray
        Parcel-averaged BOLD (n_timepoints, n_parcels)
    atlas : dict
        Schaefer atlas (for parcel labels)
    train_indices : np.ndarray
        Training indices
    test_indices : np.ndarray
        Test indices
    alphas : list, optional
        Ridge alpha values
    valid_mask : np.ndarray, optional
        Boolean mask of valid timepoints

    Returns
    -------
    dict
        Results for each layer:
        - 'r2_train', 'r2_test': R² per parcel
        - 'mean_r2_test': Average R² across parcels
        - 'parcel_labels': Names of parcels
        - 'best_alpha': Median selected alpha
    """
    if alphas is None:
        alphas = [0.1, 1, 10, 100, 1000, 10000, 100000]

    results = {}
    parcel_labels = atlas['labels']

    for layer_name, activations in layer_activations.items():
        print(f"Fitting encoding model for layer: {layer_name}")

        # Split data
        X_train = activations[train_indices]
        X_test = activations[test_indices]
        y_train = parcel_bold[train_indices]
        y_test = parcel_bold[test_indices]

        # Handle NaN masking if provided
        if valid_mask is not None:
            train_valid = valid_mask[train_indices]
            test_valid = valid_mask[test_indices]
            has_nan = np.isnan(X_train).any()

            if has_nan or not train_valid.all():
                X_train = X_train[train_valid]
                y_train = y_train[train_valid]
                X_test = X_test[test_valid]
                y_test = y_test[test_valid]
                print(f"  Using {train_valid.sum()}/{len(train_valid)} train, "
                      f"{test_valid.sum()}/{len(test_valid)} test")

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  ⚠ No valid samples, skipping...")
            continue

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit each parcel independently
        n_parcels = y_train.shape[1]
        r2_train_list = []
        r2_test_list = []
        alpha_list = []

        print(f"  Fitting {n_parcels} parcels...")

        for p_idx in range(n_parcels):
            if p_idx % 100 == 0 and p_idx > 0:
                print(f"    {p_idx}/{n_parcels} parcels...")

            # Fit ridge for this parcel
            ridge = RidgeCV(alphas=alphas, cv=3)
            ridge.fit(X_train_scaled, y_train[:, p_idx])

            # Evaluate
            y_pred_train = ridge.predict(X_train_scaled)
            y_pred_test = ridge.predict(X_test_scaled)

            r2_train_list.append(r2_score(y_train[:, p_idx], y_pred_train))
            r2_test_list.append(r2_score(y_test[:, p_idx], y_pred_test))
            alpha_list.append(ridge.alpha_)

        r2_train = np.array(r2_train_list)
        r2_test = np.array(r2_test_list)

        results[layer_name] = {
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mean_r2_train': r2_train.mean(),
            'mean_r2_test': r2_test.mean(),
            'median_r2_test': np.median(r2_test),
            'best_alpha': np.median(alpha_list),
            'parcel_labels': parcel_labels,
            'n_positive': (r2_test > 0).sum(),
            'n_significant': (r2_test > 0.01).sum()
        }

        print(f"  Median alpha: {np.median(alpha_list):.1f}")
        print(f"  Mean R² (train): {r2_train.mean():.4f}")
        print(f"  Mean R² (test): {r2_test.mean():.4f}")
        print(f"  Positive parcels: {(r2_test > 0).sum()}/{n_parcels} "
              f"({(r2_test > 0).sum()/n_parcels*100:.1f}%)")
        print()

    return results


def compare_atlas_layer_performance(encoding_results):
    """
    Compare encoding performance across layers (atlas version).

    Parameters
    ----------
    encoding_results : dict
        Results from fit_atlas_encoding_per_layer

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    import pandas as pd

    comparison = []

    for layer_name, result in encoding_results.items():
        row = {
            'layer': layer_name,
            'mean_r2': result['mean_r2_test'],
            'median_r2': result['median_r2_test'],
            'max_r2': result['r2_test'].max(),
            'n_positive': result['n_positive'],
            'pct_positive': result['n_positive'] / len(result['r2_test']) * 100,
            'n_significant': result['n_significant']
        }
        comparison.append(row)

    df = pd.DataFrame(comparison)
    df = df.sort_values('mean_r2', ascending=False)

    return df


def get_top_parcels(encoding_results, layer_name, n_top=20):
    """
    Get top N parcels for a given layer.

    Parameters
    ----------
    encoding_results : dict
        Results from fit_atlas_encoding_per_layer
    layer_name : str
        Layer to analyze
    n_top : int
        Number of top parcels to return

    Returns
    -------
    pd.DataFrame
        Top parcels with R² and labels
    """
    import pandas as pd

    result = encoding_results[layer_name]
    r2 = result['r2_test']
    labels = result['parcel_labels']

    # Get top indices
    top_indices = np.argsort(r2)[::-1][:n_top]

    # Create dataframe
    top_parcels = pd.DataFrame({
        'rank': np.arange(1, n_top + 1),
        'parcel_idx': top_indices,
        'r2': r2[top_indices],
        'label': [labels[i].decode('utf-8') if isinstance(labels[i], bytes)
                  else labels[i] for i in top_indices]
    })

    return top_parcels


def plot_atlas_r2_surfaces(encoding_results, layer_name, atlas, r2_threshold=0.01,
                            figsize=(16, 10)):
    """
    Plot parcel R² values on brain surfaces (4 views: left/right × lateral/medial).

    Similar to GLM contrast visualization, but for atlas-based encoding results.

    Parameters
    ----------
    encoding_results : dict
        Results from fit_atlas_encoding_per_layer
    layer_name : str
        Layer to visualize
    atlas : dict
        Schaefer atlas (contains 'maps')
    r2_threshold : float, default=0.01
        Minimum R² to display (parcels below this are shown as background)
    figsize : tuple, default=(16, 10)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure with 4 surface plots
    """
    from nilearn import datasets
    from nilearn.surface import vol_to_surf
    from nilearn.plotting import plot_surf_stat_map
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib import cm
    from matplotlib.colors import Normalize
    import nibabel as nib

    result = encoding_results[layer_name]
    r2_test = result['r2_test']

    print(f"Plotting R² map for layer: {layer_name}")
    print(f"  Mean R²: {r2_test.mean():.4f}")
    print(f"  Max R²: {r2_test.max():.4f}")
    print(f"  Threshold: R² > {r2_threshold}")

    # Create volume map from parcel R² values
    # Load atlas data (atlas['maps'] is a file path, not an image)
    if isinstance(atlas['maps'], str):
        atlas_img = nib.load(atlas['maps'])
    else:
        atlas_img = atlas['maps']

    atlas_data = atlas_img.get_fdata()

    # Create R² volume
    r2_volume = np.zeros_like(atlas_data)

    # Map parcel R² to volume (parcel indices start at 1, not 0)
    for parcel_idx, r2_val in enumerate(r2_test):
        # Threshold: only show parcels above threshold
        if r2_val > r2_threshold:
            r2_volume[atlas_data == (parcel_idx + 1)] = r2_val

    # Create NIfTI image
    r2_img = nib.Nifti1Image(r2_volume, atlas_img.affine)

    # Fetch fsaverage surface
    print("\nFetching fsaverage surface...")
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')

    # Project volume to surface for each hemisphere
    print("Projecting volume to surface...")
    texture_left = vol_to_surf(r2_img, fsaverage.infl_left)
    texture_right = vol_to_surf(r2_img, fsaverage.infl_right)

    # Create figure with 4 subplots + colorbar
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05],
                  hspace=0.3, wspace=0.1)

    # Get max R² for colormap (symmetric not needed for R²)
    vmax = max(np.nanmax(texture_left), np.nanmax(texture_right))
    vmin = 0  # R² is always >= 0

    # Left hemisphere - lateral
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    plot_surf_stat_map(
        fsaverage.infl_left, texture_left, hemi='left', view='lateral',
        threshold=r2_threshold, cmap='hot', colorbar=False,
        bg_map=fsaverage.sulc_left, title='Left Hemisphere - Lateral',
        axes=ax1, vmin=vmin, vmax=vmax
    )

    # Left hemisphere - medial
    ax2 = fig.add_subplot(gs[1, 0], projection='3d')
    plot_surf_stat_map(
        fsaverage.infl_left, texture_left, hemi='left', view='medial',
        threshold=r2_threshold, cmap='hot', colorbar=False,
        bg_map=fsaverage.sulc_left, title='Left Hemisphere - Medial',
        axes=ax2, vmin=vmin, vmax=vmax
    )

    # Right hemisphere - lateral
    ax3 = fig.add_subplot(gs[0, 1], projection='3d')
    plot_surf_stat_map(
        fsaverage.infl_right, texture_right, hemi='right', view='lateral',
        threshold=r2_threshold, cmap='hot', colorbar=False,
        bg_map=fsaverage.sulc_right, title='Right Hemisphere - Lateral',
        axes=ax3, vmin=vmin, vmax=vmax
    )

    # Right hemisphere - medial
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    plot_surf_stat_map(
        fsaverage.infl_right, texture_right, hemi='right', view='medial',
        threshold=r2_threshold, cmap='hot', colorbar=False,
        bg_map=fsaverage.sulc_right, title='Right Hemisphere - Medial',
        axes=ax4, vmin=vmin, vmax=vmax
    )

    # Add colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    cbar_ax = fig.add_subplot(gs[:, 2])
    cmap_obj = cm.get_cmap('hot')
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_obj),
                      cax=cbar_ax, label='R²')

    plt.suptitle(f'Encoding Model Performance: {layer_name}',
                 fontsize=16, fontweight='bold', y=0.98)

    return fig
