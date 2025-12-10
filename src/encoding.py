"""
Brain encoding utilities for Mario fMRI tutorial.
Adapted from mario_generalization encoding approach.
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import nibabel as nib
from nilearn.masking import apply_mask, unmask
from nilearn.maskers import NiftiMasker
from nilearn.image import clean_img
from nilearn.interfaces.fmriprep import load_confounds
import warnings


class RidgeEncodingModel:
    """
    Ridge regression encoding model with cross-validation.

    Predicts brain activity from feature representations.
    """

    def __init__(self, alphas=None, cv=3, standardize=True):
        """
        Initialize encoding model.

        Parameters
        ----------
        alphas : array-like, optional
            Alpha values to test. Default: [0.1, 1, 10, 100, 1000, 10000, 100000]
        cv : int, default=3
            Number of cross-validation folds
        standardize : bool, default=True
            Whether to standardize features and targets
        """
        if alphas is None:
            alphas = [0.1, 1, 10, 100, 1000, 10000, 100000]

        self.alphas = alphas
        self.cv = cv
        self.standardize = standardize

        self.ridge = RidgeCV(alphas=alphas, cv=cv)
        self.feature_scaler = StandardScaler() if standardize else None
        self.target_scaler = StandardScaler() if standardize else None

        self.is_fitted = False

    def fit(self, features, targets):
        """
        Fit ridge regression model.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix (n_samples, n_features)
        targets : np.ndarray
            Target matrix (n_samples, n_targets) - e.g., BOLD voxels

        Returns
        -------
        self
        """
        # Standardize if requested
        if self.standardize:
            features = self.feature_scaler.fit_transform(features)
            targets = self.target_scaler.fit_transform(targets)

        # Fit ridge regression
        self.ridge.fit(features, targets)

        self.is_fitted = True

        return self

    def predict(self, features):
        """
        Predict targets from features.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predicted targets (n_samples, n_targets)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Standardize if needed
        if self.standardize:
            features = self.feature_scaler.transform(features)

        # Predict
        predictions = self.ridge.predict(features)

        # Inverse transform if standardized
        if self.standardize:
            predictions = self.target_scaler.inverse_transform(predictions)

        return predictions

    def score(self, features, targets):
        """
        Compute R² score.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix
        targets : np.ndarray
            Target matrix

        Returns
        -------
        float or np.ndarray
            R² score (overall or per target)
        """
        predictions = self.predict(features)

        # Compute R² per target (voxel)
        r2_per_target = np.array([
            r2_score(targets[:, i], predictions[:, i])
            for i in range(targets.shape[1])
        ])

        return r2_per_target

    def get_best_alpha(self):
        """Get the selected alpha value."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.ridge.alpha_


def load_and_prepare_bold(bold_imgs, mask_img, confounds_list=None,
                          detrend=True, standardize=True, low_pass=None,
                          high_pass=None, t_r=1.49, load_confounds_from_fmriprep=False):
    """
    Load and prepare BOLD data for encoding.

    This function performs confound regression to remove nuisance signals from BOLD data:

    **What is confound regression?**
    - BOLD signal contains both neural activity AND noise (head motion, physiological artifacts, scanner drift)
    - Confound regression removes these nuisance signals using linear regression:
      BOLD_clean = BOLD_raw - β₁·motion - β₂·WM_signal - β₃·CSF_signal - β₄·global_signal - ...
    - This is done voxel-by-voxel: for each voxel, we regress out the confound timeseries

    **What the masker does:**
    1. **Spatial masking**: Extract only brain voxels (exclude skull, air)
    2. **Confound regression**: Remove nuisance signals from each voxel's timeseries
    3. **Temporal filtering**: Apply high-pass/low-pass filters to remove slow drifts and high-freq noise
    4. **Detrending**: Remove linear/polynomial trends within each run
    5. **Standardization**: Z-score each voxel (mean=0, std=1) for comparable scales

    **Result:** Clean BOLD signal that better reflects neural activity, not artifacts

    Parameters
    ----------
    bold_imgs : nibabel.Nifti1Image, list of images, str, or list of str
        BOLD image(s) or path(s) to NIfTI files
    mask_img : nibabel.Nifti1Image
        Brain mask
    confounds_list : pd.DataFrame or list of DataFrames, optional
        Confounds to regress out (only used if load_confounds_from_fmriprep=False)
    detrend : bool, default=True
        Whether to detrend
    standardize : bool, default=True
        Whether to standardize (z-score)
    low_pass : float, optional
        Low-pass filter cutoff in Hz
    high_pass : float, optional
        High-pass filter cutoff in Hz
    t_r : float, default=1.49
        Repetition time
    load_confounds_from_fmriprep : bool, default=False
        If True, automatically load confounds from fMRIPrep outputs using nilearn.
        Requires bold_imgs to be file paths (not nibabel images).

    Returns
    -------
    np.ndarray
        Cleaned BOLD data (n_samples, n_voxels)
        - Each row is a timepoint (TR)
        - Each column is a voxel
        - Values are z-scored BOLD signal after confound regression
    """
    # Confound loading parameters for fMRIPrep
    # Note: Removed global_signal as it was too aggressive and destroyed temporal structure
    LOAD_CONFOUNDS_PARAMS = {
        "strategy": ["motion", "high_pass", "wm_csf"],
        "motion": "basic",
        "wm_csf": "basic",
    }

    # Ensure list
    if not isinstance(bold_imgs, list):
        bold_imgs = [bold_imgs]

    if confounds_list is not None and not isinstance(confounds_list, list):
        confounds_list = [confounds_list]

    # Create masker for cleaning and masking in one step
    masker = NiftiMasker(
        mask_img=mask_img,
        detrend=detrend,
        standardize=standardize,
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=t_r
    )

    bold_data = []

    for idx, bold_img in enumerate(bold_imgs):
        # Load confounds from fMRIPrep if requested
        if load_confounds_from_fmriprep:
            # bold_img must be a file path (str or Path)
            if isinstance(bold_img, str) or hasattr(bold_img, '__fspath__'):
                confounds, _ = load_confounds(bold_img, **LOAD_CONFOUNDS_PARAMS)
            else:
                raise ValueError(
                    "When load_confounds_from_fmriprep=True, bold_imgs must be file paths, "
                    "not nibabel images. Pass paths to NIfTI files instead."
                )
        else:
            # Use provided confounds
            confounds = confounds_list[idx] if confounds_list is not None else None

        # Transform: clean + mask in one step
        # This does:
        # 1. Load BOLD data (if path provided)
        # 2. Apply spatial mask (keep only brain voxels)
        # 3. Regress out confounds from each voxel's timeseries
        # 4. Apply temporal filtering (high-pass, low-pass)
        # 5. Detrend (remove linear drift)
        # 6. Standardize (z-score each voxel)
        masked_data = masker.fit_transform(bold_img, confounds=confounds)
        bold_data.append(masked_data)

    # Concatenate if multiple runs
    if len(bold_data) > 1:
        bold_data = np.concatenate(bold_data, axis=0)
    else:
        bold_data = bold_data[0]

    return bold_data


def fit_encoding_model_per_layer(layer_activations, bold_data, mask_img,
                                 train_indices, test_indices, alphas=None,
                                 valid_mask=None):
    """
    Fit encoding models for each layer, fitting each voxel independently.

    This matches the diagnostic approach: each voxel gets its own optimal alpha,
    avoiding the issue where multivariate ridge picks one alpha for all voxels.
    """
    if alphas is None:
        alphas = [0.1, 1, 10, 100, 1000, 10000, 100000]

    results = {}

    for layer_name, activations in layer_activations.items():
        print(f"Fitting encoding model for layer: {layer_name}")

        # Split data
        X_train = activations[train_indices]
        X_test = activations[test_indices]
        y_train = bold_data[train_indices]
        y_test = bold_data[test_indices]

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
                print(f"  Using {train_valid.sum()}/{len(train_valid)} train, {test_valid.sum()}/{len(test_valid)} test")

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  ⚠ No valid samples, skipping...")
            continue

        # Standardize features (matching diagnostics)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit each voxel independently
        n_voxels = y_train.shape[1]
        r2_train_list = []
        r2_test_list = []
        alpha_list = []

        print(f"  Fitting {n_voxels:,} voxels...")

        for v_idx in range(n_voxels):
            if v_idx % 50000 == 0 and v_idx > 0:
                print(f"    {v_idx:,}/{n_voxels:,} voxels...")

            # Fit ridge for this voxel
            ridge = RidgeCV(alphas=alphas, cv=3)
            ridge.fit(X_train_scaled, y_train[:, v_idx])

            # Evaluate
            y_pred_train = ridge.predict(X_train_scaled)
            y_pred_test = ridge.predict(X_test_scaled)

            r2_train_list.append(r2_score(y_train[:, v_idx], y_pred_train))
            r2_test_list.append(r2_score(y_test[:, v_idx], y_pred_test))
            alpha_list.append(ridge.alpha_)

        r2_train = np.array(r2_train_list)
        r2_test = np.array(r2_test_list)

        # Create R² map
        r2_map = unmask(r2_test, mask_img)

        results[layer_name] = {
            'r2_train': r2_train,
            'r2_test': r2_test,
            'r2_map': r2_map,
            'best_alpha': np.median(alpha_list),
            'mean_r2_train': r2_train.mean(),
            'mean_r2_test': r2_test.mean(),
            'median_alpha': np.median(alpha_list)
        }

        print(f"  Median alpha: {np.median(alpha_list):.1f}")
        print(f"  Mean R² (train): {r2_train.mean():.4f}")
        print(f"  Mean R² (test): {r2_test.mean():.4f}")
        print()

    return results


def cross_validated_encoding(layer_activations, bold_data, mask_img,
                              n_folds=3, alphas=None):
    """
    Perform k-fold cross-validation for encoding models.

    Parameters
    ----------
    layer_activations : dict
        Dictionary mapping layer names to activation arrays
    bold_data : np.ndarray
        BOLD data (n_samples, n_voxels)
    mask_img : nibabel.Nifti1Image
        Brain mask
    n_folds : int, default=3
        Number of CV folds
    alphas : array-like, optional
        Alpha values for ridge regression

    Returns
    -------
    dict
        Results for each layer with cross-validated R² scores
    """
    n_samples = bold_data.shape[0]
    kfold = KFold(n_splits=n_folds, shuffle=False)

    results = {layer: {'r2_folds': []} for layer in layer_activations.keys()}

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(np.arange(n_samples))):
        print(f"Fold {fold_idx + 1}/{n_folds}")

        fold_results = fit_encoding_model_per_layer(
            layer_activations, bold_data, mask_img,
            train_idx, test_idx, alphas
        )

        for layer_name, layer_result in fold_results.items():
            results[layer_name]['r2_folds'].append(layer_result['r2_test'])

    # Aggregate results
    for layer_name in results.keys():
        r2_folds = np.array(results[layer_name]['r2_folds'])
        results[layer_name]['mean_r2'] = r2_folds.mean(axis=0)
        results[layer_name]['std_r2'] = r2_folds.std(axis=0)
        results[layer_name]['mean_r2_overall'] = r2_folds.mean()

        # Create mean R² map
        results[layer_name]['r2_map'] = unmask(results[layer_name]['mean_r2'], mask_img)

    return results


def compare_layer_performance(encoding_results):
    """
    Compare encoding performance across layers.

    Parameters
    ----------
    encoding_results : dict
        Results from fit_encoding_model_per_layer or cross_validated_encoding

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    import pandas as pd

    comparison = []

    for layer_name, result in encoding_results.items():
        if 'mean_r2_test' in result:
            # Single split results
            row = {
                'layer': layer_name,
                'mean_r2': result['mean_r2_test'],
                'max_r2': result['r2_test'].max(),
                'median_r2': np.median(result['r2_test']),
                'n_positive_voxels': (result['r2_test'] > 0).sum()
            }
        elif 'mean_r2_overall' in result:
            # Cross-validated results
            row = {
                'layer': layer_name,
                'mean_r2': result['mean_r2_overall'],
                'max_r2': result['mean_r2'].max(),
                'median_r2': np.median(result['mean_r2']),
                'n_positive_voxels': (result['mean_r2'] > 0).sum()
            }
        else:
            continue

        comparison.append(row)

    df = pd.DataFrame(comparison)
    df = df.sort_values('mean_r2', ascending=False)

    return df


def compute_noise_ceiling(bold_data_list):
    """
    Compute noise ceiling from multiple runs.

    The noise ceiling is the theoretical upper bound on prediction performance,
    estimated from the correlation between repeated measurements.

    Parameters
    ----------
    bold_data_list : list of np.ndarray
        BOLD data from multiple runs (each n_samples_i, n_voxels)

    Returns
    -------
    np.ndarray
        Noise ceiling per voxel
    """
    if len(bold_data_list) < 2:
        warnings.warn("Need at least 2 runs to compute noise ceiling")
        return None

    # Ensure all runs have same number of voxels
    n_voxels = bold_data_list[0].shape[1]

    # Compute pairwise correlations between runs
    from scipy.stats import pearsonr

    noise_ceiling = np.zeros(n_voxels)

    for voxel_idx in range(n_voxels):
        correlations = []

        # Get all pairwise correlations
        for i in range(len(bold_data_list)):
            for j in range(i + 1, len(bold_data_list)):
                # Match lengths
                min_len = min(len(bold_data_list[i]), len(bold_data_list[j]))
                r, _ = pearsonr(
                    bold_data_list[i][:min_len, voxel_idx],
                    bold_data_list[j][:min_len, voxel_idx]
                )
                correlations.append(r ** 2)  # R²

        noise_ceiling[voxel_idx] = np.mean(correlations)

    return noise_ceiling


def create_encoding_summary_figure(encoding_results, layer_order=None):
    """
    Create summary figure comparing layer performance.

    Parameters
    ----------
    encoding_results : dict
        Results from encoding analysis
    layer_order : list, optional
        Order to display layers

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if layer_order is None:
        layer_order = sorted(encoding_results.keys())

    # Extract mean R² for each layer
    mean_r2_values = [encoding_results[layer].get('mean_r2_test',
                      encoding_results[layer].get('mean_r2_overall', 0))
                      for layer in layer_order]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot
    bars = ax.bar(range(len(layer_order)), mean_r2_values, color='steelblue')

    # Styling
    ax.set_xticks(range(len(layer_order)))
    ax.set_xticklabels(layer_order, rotation=45, ha='right')
    ax.set_ylabel('Mean R² (test)', fontsize=12)
    ax.set_xlabel('CNN Layer', fontsize=12)
    ax.set_title('Encoding Performance by Layer', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

    # Add value labels on bars
    for bar, value in zip(bars, mean_r2_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    return fig


# ==============================================================================
# ATLAS-BASED ENCODING FUNCTIONS
# ==============================================================================

def fit_atlas_encoding_per_layer(layer_activations, parcel_bold, atlas,
                                  train_indices, test_indices, alphas=None,
                                  valid_mask=None):
    """
    Fit encoding models using atlas parcels with constant signal check.

    Parameters
    ----------
    layer_activations : dict
        Dictionary of layer activations (layer_name -> activations array)
    parcel_bold : ndarray
        Parcel-averaged BOLD timeseries (timepoints × parcels)
    atlas : dict
        Atlas with 'labels' key
    train_indices : ndarray
        Training indices
    test_indices : ndarray
        Test indices
    alphas : list, optional
        Ridge regression alpha values
    valid_mask : ndarray, optional
        Valid samples mask

    Returns
    -------
    dict
        Encoding results for each layer
    """
    if alphas is None:
        alphas = [0.1, 1, 10, 100, 1000, 10000, 100000]

    results = {}
    parcel_labels = atlas['labels']

    # Identify valid parcels (non-constant signal)
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

    Parameters
    ----------
    encoding_results : dict
        Encoding results from fit_atlas_encoding_per_layer

    Returns
    -------
    pd.DataFrame
        Comparison table with performance metrics per layer
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
        Encoding results from fit_atlas_encoding_per_layer
    layer_name : str
        Layer name to get top parcels for
    n_top : int, default=20
        Number of top parcels to return

    Returns
    -------
    pd.DataFrame
        DataFrame with top parcels ranked by R²
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
