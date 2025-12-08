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
from nilearn.image import clean_img
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
                          high_pass=None, t_r=1.49):
    """
    Load and prepare BOLD data for encoding.

    Parameters
    ----------
    bold_imgs : nibabel.Nifti1Image or list of images
        BOLD image(s)
    mask_img : nibabel.Nifti1Image
        Brain mask
    confounds_list : pd.DataFrame or list of DataFrames, optional
        Confounds to regress out
    detrend : bool, default=True
        Whether to detrend
    standardize : bool, default=True
        Whether to standardize
    low_pass : float, optional
        Low-pass filter cutoff in Hz
    high_pass : float, optional
        High-pass filter cutoff in Hz
    t_r : float, default=1.49
        Repetition time

    Returns
    -------
    np.ndarray
        Cleaned BOLD data (n_samples, n_voxels)
    """
    # Ensure list
    if not isinstance(bold_imgs, list):
        bold_imgs = [bold_imgs]

    if confounds_list is not None and not isinstance(confounds_list, list):
        confounds_list = [confounds_list]

    bold_data = []

    for idx, bold_img in enumerate(bold_imgs):
        confounds = confounds_list[idx] if confounds_list is not None else None

        # Clean image
        cleaned_img = clean_img(
            bold_img,
            confounds=confounds,
            detrend=detrend,
            standardize=standardize,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=t_r,
            mask_img=mask_img
        )

        # Mask to get 2D array
        masked_data = apply_mask(cleaned_img, mask_img)
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
    Fit encoding models for each layer separately.

    Parameters
    ----------
    layer_activations : dict
        Dictionary mapping layer names to activation arrays
    bold_data : np.ndarray
        BOLD data (n_samples, n_voxels)
    mask_img : nibabel.Nifti1Image
        Brain mask (for creating result images)
    train_indices : np.ndarray
        Indices for training
    test_indices : np.ndarray
        Indices for testing
    alphas : array-like, optional
        Alpha values for ridge regression
    valid_mask : np.ndarray, optional
        Boolean mask indicating valid (non-NaN) samples (n_samples,)
        If provided, only valid samples will be used for training/testing

    Returns
    -------
    dict
        Results for each layer with keys:
        - 'model': Fitted RidgeEncodingModel
        - 'r2_train': R² on training set
        - 'r2_test': R² on test set
        - 'r2_map': R² map as Nifti1Image
        - 'best_alpha': Selected alpha value
    """
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
            # Get valid samples in train/test sets
            train_valid = valid_mask[train_indices]
            test_valid = valid_mask[test_indices]

            # Check if activations contain NaNs
            has_nan = np.isnan(X_train).any()

            if has_nan or not train_valid.all():
                # Filter to valid samples only
                X_train = X_train[train_valid]
                y_train = y_train[train_valid]
                X_test = X_test[test_valid]
                y_test = y_test[test_valid]

                print(f"  Using {train_valid.sum()}/{len(train_valid)} train samples")
                print(f"  Using {test_valid.sum()}/{len(test_valid)} test samples")

        # Check if we have any valid samples
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  ⚠ No valid samples for {layer_name}, skipping...")
            continue

        # Create and fit model
        model = RidgeEncodingModel(alphas=alphas, cv=3, standardize=True)
        model.fit(X_train, y_train)

        # Evaluate
        r2_train = model.score(X_train, y_train)
        r2_test = model.score(X_test, y_test)

        # Create R² map
        r2_map = unmask(r2_test, mask_img)

        results[layer_name] = {
            'model': model,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'r2_map': r2_map,
            'best_alpha': model.get_best_alpha(),
            'mean_r2_train': r2_train.mean(),
            'mean_r2_test': r2_test.mean()
        }

        print(f"  Best alpha: {model.get_best_alpha():.1f}")
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
