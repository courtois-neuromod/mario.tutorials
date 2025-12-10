"""
Atlas-based encoding utilities for computational efficiency.

Instead of fitting 213k voxels independently, we:
1. Use a functional atlas (Schaefer or BASC) to parcellate brain
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
import nibabel as nib


def load_schaefer_atlas(n_rois=400, yeo_networks=7, resolution_mm=2):
    """
    Load Schaefer 2018 functional atlas.
    """
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


def save_complete_results(parcel_bold, all_pca_results, all_encoding_results,
                          train_indices, test_indices, valid_mask, atlas,
                          filepath, pca_dims=None, subject=None, session=None):
    """
    Save ALL processing results to disk to avoid recomputation.
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
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = filepath.stat().st_size / 1024 / 1024
    print(f"\n✓ Complete results saved!")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"{'='*80}\n")


def load_complete_results(filepath):
    """
    Load ALL processing results from disk.
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
    print(f"\n✓ All processing steps skipped - using cached results!")
    print(f"{'='*80}\n")

    return results


def check_complete_results_exist(filepath):
    return Path(filepath).exists()


def extract_parcel_bold(bold_imgs, atlas, confounds_list=None,
                        detrend=True, standardize=True, t_r=1.49, mask_img=None):
    """
    Extract parcel-averaged BOLD timeseries using atlas.
    """
    from nilearn.image import resample_to_img
    import nibabel as nib

    # Ensure list
    if not isinstance(bold_imgs, list):
        bold_imgs = [bold_imgs]

    # Load atlas image if it's a path
    if isinstance(atlas['maps'], str):
        atlas_img = nib.load(atlas['maps'])
    else:
        atlas_img = atlas['maps']

    # Explicitly resample atlas to mask_img if provided (User requested explicit affine match)
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


def fit_atlas_encoding_per_layer(layer_activations, parcel_bold, atlas,
                                  train_indices, test_indices, alphas=None,
                                  valid_mask=None):
    """
    Fit encoding models using atlas parcels with constant signal check.
    """
    if alphas is None:
        alphas = [0.1, 1, 10, 100, 1000, 10000, 100000]

    results = {}
    parcel_labels = atlas['labels']

    # Identify valid parcels (non-constant signal)
    # Check signal on FULL dataset (before splitting) or train set?
    # Better to check on full or train. If train is constant, we can't fit.
    # If standard deviation is 0 (or very close), it's invalid.
    
    print("Checking for constant/zero signals in parcels...")
    parcel_std = np.std(parcel_bold, axis=0)
    valid_parcel_mask = parcel_std > 1e-6  # Threshold for "constant"
    n_valid_parcels = np.sum(valid_parcel_mask)
    n_total_parcels = len(parcel_labels)
    
    print(f"  Valid parcels: {n_valid_parcels}/{n_total_parcels}")
    if n_valid_parcels < n_total_parcels:
        print(f"  Dropped {n_total_parcels - n_valid_parcels} constant parcels.")

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
            # Check if this parcel is valid
            if not valid_parcel_mask[p_idx]:
                # Constant signal, skip fit, return 0 R2
                r2_train_list.append(0.0)
                r2_test_list.append(0.0)
                alpha_list.append(np.nan)
                continue

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
        
        # Calculate median alpha ignoring NaNs
        valid_alphas = [a for a in alpha_list if not np.isnan(a)]
        median_alpha = np.median(valid_alphas) if valid_alphas else 0

        results[layer_name] = {
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mean_r2_train': r2_train.mean(),
            'mean_r2_test': r2_test.mean(),
            'median_r2_test': np.median(r2_test),
            'best_alpha': median_alpha,
            'parcel_labels': parcel_labels,
            'n_positive': (r2_test > 0).sum(),
            'n_significant': (r2_test > 0.01).sum()
        }

        print(f"  Median alpha: {median_alpha:.1f}")
        print(f"  Mean R² (test): {r2_test.mean():.4f}")
        print(f"  Positive parcels: {(r2_test > 0).sum()}/{n_parcels} "
              f"({(r2_test > 0).sum()/n_parcels*100:.1f}%)")
        print()

    return results


def compare_atlas_layer_performance(encoding_results):
    """
    Compare encoding performance across layers.
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
    """
    import pandas as pd

    result = encoding_results[layer_name]
    r2 = result['r2_test']
    labels = result['parcel_labels']

    # Handle label array length mismatch
    # Schaefer atlas labels include "Background" at index 0, but r2_test doesn't
    if len(labels) > len(r2):
        # Skip first label (Background)
        labels = labels[1:]

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
    Plot parcel R² values on brain surfaces.
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

    # Load atlas data
    if isinstance(atlas['maps'], str):
        atlas_img = nib.load(atlas['maps'])
    else:
        atlas_img = atlas['maps']

    atlas_data = atlas_img.get_fdata()

    # Create R² volume
    # Initialize with NaN instead of 0 to distinguish "no data" from "zero R²"
    r2_volume = np.full_like(atlas_data, np.nan, dtype=np.float32)

    # Map parcel R² to volume
    # NOTE: BASC labels are integers 1..N
    # Schaefer labels are integers 1..N
    # We assume 'labels' list corresponds to indices 0..N-1 in r2_test
    # and map to atlas values 1..N

    unique_vals = np.unique(atlas_data)
    # Remove 0
    unique_vals = unique_vals[unique_vals != 0]
    unique_vals.sort()

    # Check if number of unique values matches r2_test length
    if len(unique_vals) != len(r2_test):
        print(f"Warning: Atlas has {len(unique_vals)} regions but results have {len(r2_test)} values.")
        print("Mapping by sorted index...")

    # Map ALL R² values to the volume (including negative ones)
    # The threshold will be applied during visualization, not here
    for i, region_id in enumerate(unique_vals):
        if i < len(r2_test):
            r2_val = r2_test[i]
            r2_volume[atlas_data == region_id] = r2_val

    # Create NIfTI image
    r2_img = nib.Nifti1Image(r2_volume, atlas_img.affine)

    # Fetch fsaverage surface
    print("\nFetching fsaverage surface...")
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')

    # Project volume to surface for each hemisphere
    print("Projecting volume to surface...")
    # Strategy: Try multiple radii and interpolations to maximize surface coverage
    # First try linear with large radius
    texture_left = vol_to_surf(
        r2_img,
        fsaverage.infl_left,
        radius=4.0,  # Larger radius for better coverage
        interpolation='linear',
        n_samples=None  # Use default
    )
    texture_right = vol_to_surf(
        r2_img,
        fsaverage.infl_right,
        radius=4.0,
        interpolation='linear',
        n_samples=None
    )

    # For vertices that are still NaN, try nearest neighbor with even larger radius
    nan_mask_left = np.isnan(texture_left)
    nan_mask_right = np.isnan(texture_right)

    if nan_mask_left.any():
        print(f"  Filling {nan_mask_left.sum()} NaN vertices on left hemisphere...")
        texture_left_nearest = vol_to_surf(
            r2_img,
            fsaverage.infl_left,
            radius=6.0,
            interpolation='linear',  # Use linear to avoid deprecation warning
            n_samples=None
        )
        texture_left[nan_mask_left] = texture_left_nearest[nan_mask_left]

    if nan_mask_right.any():
        print(f"  Filling {nan_mask_right.sum()} NaN vertices on right hemisphere...")
        texture_right_nearest = vol_to_surf(
            r2_img,
            fsaverage.infl_right,
            radius=6.0,
            interpolation='linear',  # Use linear to avoid deprecation warning
            n_samples=None
        )
        texture_right[nan_mask_right] = texture_right_nearest[nan_mask_right]

    # Keep NaN values as NaN - plot_surf_stat_map will handle them properly
    # This way, NaN regions will show the sulcal background instead of being colored

    print(f"  Left hemisphere: {np.sum(~np.isnan(texture_left) & (texture_left > r2_threshold))} vertices above threshold")
    print(f"  Right hemisphere: {np.sum(~np.isnan(texture_right) & (texture_right > r2_threshold))} vertices above threshold")

    # Create figure with 4 subplots + colorbar
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05],
                  hspace=0.05, wspace=0.05)

    # Get max R² for colormap
    vmax = max(np.nanmax(texture_left), np.nanmax(texture_right))
    if vmax == 0: vmax = 0.1 # Avoid error if all 0
    vmin = 0

    # Left hemisphere - lateral
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    plot_surf_stat_map(
        fsaverage.infl_left, texture_left, hemi='left', view='lateral',
        threshold=r2_threshold, cmap='hot', colorbar=False,
        bg_map=fsaverage.sulc_left, title='Left Hemisphere - Lateral',
        axes=ax1, vmin=vmin, vmax=vmax
    )
    ax1.dist = 7

    # Left hemisphere - medial
    ax2 = fig.add_subplot(gs[1, 0], projection='3d')
    plot_surf_stat_map(
        fsaverage.infl_left, texture_left, hemi='left', view='medial',
        threshold=r2_threshold, cmap='hot', colorbar=False,
        bg_map=fsaverage.sulc_left, title='Left Hemisphere - Medial',
        axes=ax2, vmin=vmin, vmax=vmax
    )
    ax2.dist = 7

    # Right hemisphere - lateral
    ax3 = fig.add_subplot(gs[0, 1], projection='3d')
    plot_surf_stat_map(
        fsaverage.infl_right, texture_right, hemi='right', view='lateral',
        threshold=r2_threshold, cmap='hot', colorbar=False,
        bg_map=fsaverage.sulc_right, title='Right Hemisphere - Lateral',
        axes=ax3, vmin=vmin, vmax=vmax
    )
    ax3.dist = 7

    # Right hemisphere - medial
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    plot_surf_stat_map(
        fsaverage.infl_right, texture_right, hemi='right', view='medial',
        threshold=r2_threshold, cmap='hot', colorbar=False,
        bg_map=fsaverage.sulc_right, title='Right Hemisphere - Medial',
        axes=ax4, vmin=vmin, vmax=vmax
    )
    ax4.dist = 7

    # Add colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    cbar_ax = fig.add_subplot(gs[:, 2])
    cmap_obj = cm.get_cmap('hot')
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_obj),
                      cax=cbar_ax, label='R²')

    plt.suptitle(f'Encoding Model Performance: {layer_name}',
                 fontsize=16, fontweight='bold', y=0.98)

    return fig


def plot_network_performance(encoding_results, layer_name, atlas, figsize=(12, 6)):
    """
    Plot encoding performance by Yeo network for a single layer.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    result = encoding_results[layer_name]
    r2_test = result['r2_test']
    labels = result['parcel_labels']

    # Handle label array length mismatch
    # Schaefer atlas labels include "Background" at index 0, but r2_test doesn't
    if len(labels) > len(r2_test):
        # Skip first label (Background)
        labels = labels[1:]

    # Ensure lengths match
    if len(labels) != len(r2_test):
        raise ValueError(f"Label and R² array length mismatch: {len(labels)} labels vs {len(r2_test)} R² values")

    # Extract network names (e.g., "Vis" from "7Networks_LH_Vis_1")
    networks = []

    for label in labels:
        if isinstance(label, bytes):
            label = label.decode('utf-8')

        parts = label.split('_')
        if len(parts) >= 3:
            # 7Networks_LH_Vis_1 -> Vis
            # 7Networks_RH_SomMot_3 -> SomMot
            net = parts[2]
            networks.append(net)
        else:
            networks.append('Unknown')

    df = pd.DataFrame({
        'Network': networks,
        'R2': r2_test
    })

    # Plot
    plt.figure(figsize=figsize)

    # Sort by median R2
    order = df.groupby('Network')['R2'].median().sort_values(ascending=False).index

    sns.boxplot(data=df, x='Network', y='R2', order=order, palette='viridis')
    plt.title(f'Performance by Functional Network ({layer_name})', fontsize=14)
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)

    return plt.gcf()


def plot_network_performance_grid(all_encoding_results, pca_dim, atlas,
                                  all_encoding_results_untrained=None,
                                  figsize=(20, 12)):
    """
    Create a grid plot showing network performance for all CNN layers.

    Optionally compare trained vs untrained network performance.

    Parameters
    ----------
    all_encoding_results : dict
        Nested dict: {pca_dim: {layer_name: results}} for trained network
    pca_dim : int
        Which PCA dimension to plot
    atlas : dict
        Atlas dictionary with 'labels'
    all_encoding_results_untrained : dict, optional
        Same structure as all_encoding_results but for untrained network.
        If provided, will show trained vs untrained comparison.
    figsize : tuple
        Figure size

    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    encoding_results = all_encoding_results[pca_dim]
    layers = list(encoding_results.keys())
    n_layers = len(layers)

    # Check if we're doing comparison
    compare_mode = all_encoding_results_untrained is not None

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    # Common network order (sorted by overall median R²)
    all_data = []
    for layer_name in layers:
        result = encoding_results[layer_name]
        r2_test = result['r2_test']
        labels = result['parcel_labels']

        # Handle label mismatch
        if len(labels) > len(r2_test):
            labels = labels[1:]

        # Extract networks
        networks = []
        for label in labels:
            if isinstance(label, bytes):
                label = label.decode('utf-8')
            parts = label.split('_')
            if len(parts) >= 3:
                networks.append(parts[2])
            else:
                networks.append('Unknown')

        if compare_mode:
            condition = 'Trained'
            all_data.extend(list(zip(networks, r2_test, [layer_name]*len(r2_test), [condition]*len(r2_test))))
        else:
            all_data.extend(list(zip(networks, r2_test, [layer_name]*len(r2_test))))

    # If comparing, add untrained data
    if compare_mode:
        encoding_results_untrained = all_encoding_results_untrained[pca_dim]
        for layer_name in layers:
            result = encoding_results_untrained[layer_name]
            r2_test = result['r2_test']
            labels = result['parcel_labels']

            if len(labels) > len(r2_test):
                labels = labels[1:]

            networks = []
            for label in labels:
                if isinstance(label, bytes):
                    label = label.decode('utf-8')
                parts = label.split('_')
                if len(parts) >= 3:
                    networks.append(parts[2])
                else:
                    networks.append('Unknown')

            all_data.extend(list(zip(networks, r2_test, [layer_name]*len(r2_test), ['Untrained']*len(r2_test))))

    # Create dataframe
    columns = ['Network', 'R2', 'Layer', 'Condition'] if compare_mode else ['Network', 'R2', 'Layer']
    all_df = pd.DataFrame(all_data, columns=columns)

    # Get network order (by overall median R² for trained network)
    if compare_mode:
        network_order = all_df[all_df['Condition'] == 'Trained'].groupby('Network')['R2'].median().sort_values(ascending=False).index.tolist()
    else:
        network_order = all_df.groupby('Network')['R2'].median().sort_values(ascending=False).index.tolist()

    # Plot each layer
    for idx, layer_name in enumerate(layers):
        ax = axes[idx]

        if compare_mode:
            # Prepare data for this layer (both trained and untrained)
            layer_data = all_df[all_df['Layer'] == layer_name]

            # Plot with hue for trained vs untrained
            sns.boxplot(data=layer_data, x='Network', y='R2', hue='Condition',
                       order=network_order, palette=['#1f77b4', '#ff7f0e'],
                       ax=ax)

            # Get mean R² for trained and untrained
            mean_r2_trained = layer_data[layer_data['Condition'] == 'Trained']['R2'].mean()
            mean_r2_untrained = layer_data[layer_data['Condition'] == 'Untrained']['R2'].mean()
            delta = mean_r2_trained - mean_r2_untrained

            ax.set_title(f'{layer_name.upper()}\nTrained={mean_r2_trained:.4f}, Untrained={mean_r2_untrained:.4f} (Δ={delta:.4f})',
                        fontsize=11, fontweight='bold')

            # Move legend to upper right, smaller
            if idx == 2:  # Only show legend on top-right subplot
                ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            else:
                ax.legend().set_visible(False)
        else:
            # Original single-condition plot
            result = encoding_results[layer_name]
            r2_test = result['r2_test']
            labels = result['parcel_labels']

            # Handle label mismatch
            if len(labels) > len(r2_test):
                labels = labels[1:]

            # Extract networks
            networks = []
            for label in labels:
                if isinstance(label, bytes):
                    label = label.decode('utf-8')
                parts = label.split('_')
                if len(parts) >= 3:
                    networks.append(parts[2])
                else:
                    networks.append('Unknown')

            df = pd.DataFrame({
                'Network': networks,
                'R2': r2_test
            })

            # Plot
            sns.boxplot(data=df, x='Network', y='R2', order=network_order,
                       palette='viridis', ax=ax)

            # Get mean R² for this layer
            mean_r2 = r2_test.mean()

            ax.set_title(f'{layer_name.upper()} (Mean R²={mean_r2:.4f})',
                        fontsize=13, fontweight='bold')

        ax.set_ylabel('R² Score' if idx % 3 == 0 else '')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Set consistent y-axis limits across all plots
        ax.set_ylim([all_df['R2'].min() - 0.01, all_df['R2'].max() + 0.01])

    # Hide extra subplot
    if n_layers < len(axes):
        for idx in range(n_layers, len(axes)):
            axes[idx].set_visible(False)

    if compare_mode:
        title = f'Network Performance: Trained vs Untrained ({pca_dim} PCA components)'
    else:
        title = f'Network Performance Across CNN Layers ({pca_dim} PCA components)'

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    return fig


def plot_glass_brain_r2(encoding_results, layer_name, atlas, r2_threshold=0.01,
                        encoding_results_untrained=None):
    """
    Plot transparent glass brain view of R² results.

    Optionally compare trained vs untrained networks.

    Parameters
    ----------
    encoding_results : dict
        Results for trained network
    layer_name : str
        Which layer to plot
    atlas : dict
        Atlas with 'maps' and 'labels'
    r2_threshold : float
        Display threshold
    encoding_results_untrained : dict, optional
        Results for untrained network

    Returns
    -------
    fig : Figure
    """
    from nilearn import plotting
    import nibabel as nib
    import matplotlib.pyplot as plt

    compare_mode = encoding_results_untrained is not None

    # Helper to create R² volume
    def create_r2_volume(r2_test, atlas_img, atlas_data):
        r2_volume = np.zeros_like(atlas_data)
        unique_vals = np.unique(atlas_data)
        unique_vals = unique_vals[unique_vals != 0]
        unique_vals.sort()

        for i, region_id in enumerate(unique_vals):
            if i < len(r2_test):
                r2_val = r2_test[i]
                if r2_val > r2_threshold:
                    r2_volume[atlas_data == region_id] = r2_val

        return nib.Nifti1Image(r2_volume, atlas_img.affine)

    # Load atlas
    if isinstance(atlas['maps'], str):
        atlas_img = nib.load(atlas['maps'])
    else:
        atlas_img = atlas['maps']
    atlas_data = atlas_img.get_fdata()

    if compare_mode:
        # Comparison mode: trained vs untrained
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))

        # Trained
        result_trained = encoding_results[layer_name]
        r2_img_trained = create_r2_volume(result_trained['r2_test'], atlas_img, atlas_data)

        plotting.plot_glass_brain(
            r2_img_trained,
            display_mode='lyrz',
            colorbar=True,
            threshold=r2_threshold,
            cmap='hot',
            plot_abs=False,
            title=f'Trained - {layer_name.upper()} (Mean R²={result_trained["r2_test"].mean():.4f})',
            axes=axes[0]
        )

        # Untrained
        result_untrained = encoding_results_untrained[layer_name]
        r2_img_untrained = create_r2_volume(result_untrained['r2_test'], atlas_img, atlas_data)

        plotting.plot_glass_brain(
            r2_img_untrained,
            display_mode='lyrz',
            colorbar=True,
            threshold=r2_threshold,
            cmap='hot',
            plot_abs=False,
            title=f'Untrained - {layer_name.upper()} (Mean R²={result_untrained["r2_test"].mean():.4f})',
            axes=axes[1]
        )

        fig.suptitle(f'Glass Brain Comparison: Trained vs Untrained',
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

    else:
        # Single network mode
        result = encoding_results[layer_name]
        r2_img = create_r2_volume(result['r2_test'], atlas_img, atlas_data)

        fig = plt.figure(figsize=(10, 4))
        plotting.plot_glass_brain(
            r2_img,
            display_mode='lyrz',
            colorbar=True,
            threshold=r2_threshold,
            cmap='hot',
            plot_abs=False,
            title=f'Glass Brain View: {layer_name}',
            figure=fig
        )

    return fig


def visualize_best_parcel_prediction(layer_activations, parcel_bold, atlas, 
                                     train_indices, test_indices, layer_name, 
                                     encoding_results, alphas=None):
    """
    Visualize time series prediction for the best parcel.
    """
    import matplotlib.pyplot as plt
    
    if alphas is None:
        alphas = [0.1, 1, 10, 100, 1000, 10000, 100000]

    # Find best parcel
    result = encoding_results[layer_name]
    best_idx = np.argmax(result['r2_test'])
    best_r2 = result['r2_test'][best_idx]
    label = result['parcel_labels'][best_idx]
    
    if isinstance(label, bytes):
        label = label.decode('utf-8')
    
    print(f"Refitting best parcel: {label} (R² = {best_r2:.4f})")
    
    # Get data
    X = layer_activations[layer_name]
    y = parcel_bold[:, best_idx]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Refit
    ridge = RidgeCV(alphas=alphas, cv=3)
    ridge.fit(X_train_scaled, y_train)
    y_pred = ridge.predict(X_test_scaled)
    
    # Plot
    plt.figure(figsize=(14, 4))
    plt.plot(y_test, label='Actual BOLD', alpha=0.7)
    plt.plot(y_pred, label='Predicted BOLD', alpha=0.8, color='orange')
    plt.title(f'Actual vs Predicted Signal: {label}\nTest Set (n={len(y_test)}), R²={best_r2:.3f}')
    plt.xlabel('Time (TRs)')
    plt.ylabel('BOLD Signal (z-score)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    return plt.gcf()
