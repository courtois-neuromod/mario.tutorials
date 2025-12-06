"""
RL agent utilities for Mario fMRI tutorial.
Adapted from mario_generalization approach.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from nilearn.glm.first_level import compute_regressor


class SimpleCNN(nn.Module):
    """
    Simplified CNN for Mario gameplay (PPO-style architecture).
    4-layer CNN + linear layer for actor-critic.
    """

    def __init__(self, n_actions=12, input_channels=4):
        super(SimpleCNN, self).__init__()

        # Convolutional layers (32 filters each, 3x3 kernel, stride 2)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        # Calculate size after convolutions
        # Input: 84x84 -> 42x42 -> 21x21 -> 11x11 -> 6x6
        self.feature_size = 32 * 6 * 6

        # Linear layer
        self.fc = nn.Linear(self.feature_size, 512)

        # Actor and critic heads
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

        self.relu = nn.ReLU()

    def forward(self, x, return_activations=False):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input frames (B, C, H, W)
        return_activations : bool, default=False
            If True, return activations from all layers

        Returns
        -------
        tuple or dict
            If return_activations=False: (action_logits, value)
            If return_activations=True: dict with all layer activations
        """
        # Normalize input (assuming 0-255 uint8 input)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        activations = {}

        # Conv layers
        x1 = self.relu(self.conv1(x))
        if return_activations:
            activations['conv1'] = x1

        x2 = self.relu(self.conv2(x1))
        if return_activations:
            activations['conv2'] = x2

        x3 = self.relu(self.conv3(x2))
        if return_activations:
            activations['conv3'] = x3

        x4 = self.relu(self.conv4(x3))
        if return_activations:
            activations['conv4'] = x4

        # Flatten
        x_flat = x4.view(x4.size(0), -1)

        # Linear layer
        x_fc = self.relu(self.fc(x_flat))
        if return_activations:
            activations['linear'] = x_fc

        # Actor-critic heads
        action_logits = self.actor(x_fc)
        value = self.critic(x_fc)

        if return_activations:
            activations['action_logits'] = action_logits
            activations['value'] = value
            return activations
        else:
            return action_logits, value


def preprocess_frame(frame):
    """
    Preprocess a single game frame.

    Parameters
    ----------
    frame : np.ndarray
        RGB frame (H, W, 3)

    Returns
    -------
    np.ndarray
        Grayscale, resized frame (84, 84)
    """
    import cv2

    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame

    # Resize to 84x84
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    return resized


def create_frame_stack(frames):
    """
    Create stacked frames for input to CNN.

    Parameters
    ----------
    frames : list or np.ndarray
        List of 4 preprocessed frames (84, 84)

    Returns
    -------
    np.ndarray
        Stacked frames (4, 84, 84)
    """
    if isinstance(frames, list):
        frames = np.stack(frames, axis=0)

    # Ensure shape is (4, 84, 84)
    if frames.shape != (4, 84, 84):
        raise ValueError(f"Expected frames shape (4, 84, 84), got {frames.shape}")

    return frames


def extract_activations_from_frames(model, frames, layer_names, device='cpu', batch_size=32):
    """
    Extract CNN activations from game frames.

    Parameters
    ----------
    model : SimpleCNN
        Trained CNN model
    frames : np.ndarray
        Preprocessed, stacked frames (N, 4, 84, 84)
    layer_names : list of str
        Layer names to extract ('conv1', 'conv2', 'conv3', 'conv4', 'linear')
    device : str, default='cpu'
        Device for computation
    batch_size : int, default=32
        Batch size for processing

    Returns
    -------
    dict
        Dictionary mapping layer names to activation arrays (N, n_features)
    """
    model = model.to(device)
    model.eval()

    n_frames = frames.shape[0]
    activations = {layer: [] for layer in layer_names}

    with torch.no_grad():
        for i in range(0, n_frames, batch_size):
            batch = frames[i:i+batch_size]

            # Convert to tensor
            batch_tensor = torch.from_numpy(batch).float().to(device)

            # Forward pass
            layer_acts = model(batch_tensor, return_activations=True)

            # Collect activations
            for layer in layer_names:
                if layer in layer_acts:
                    # Flatten spatial dimensions for conv layers
                    act = layer_acts[layer].cpu().numpy()
                    if len(act.shape) == 4:  # Conv layer (B, C, H, W)
                        act = act.reshape(act.shape[0], -1)  # (B, C*H*W)
                    activations[layer].append(act)

    # Concatenate batches
    for layer in layer_names:
        if len(activations[layer]) > 0:
            activations[layer] = np.concatenate(activations[layer], axis=0)

    return activations


def downsample_activations_to_tr(activations, frame_times, tr, run_duration):
    """
    Downsample frame-rate activations to TR resolution.

    Parameters
    ----------
    activations : np.ndarray
        Activation array (n_frames, n_features)
    frame_times : np.ndarray
        Time of each frame in seconds (n_frames,)
    tr : float
        Repetition time in seconds
    run_duration : float
        Total run duration in seconds

    Returns
    -------
    np.ndarray
        Downsampled activations (n_trs, n_features)
    """
    n_trs = int(np.ceil(run_duration / tr))
    n_features = activations.shape[1]

    downsampled = np.zeros((n_trs, n_features))

    for tr_idx in range(n_trs):
        # Define TR window
        tr_start = tr_idx * tr
        tr_end = (tr_idx + 1) * tr

        # Find frames in this TR
        in_window = (frame_times >= tr_start) & (frame_times < tr_end)

        if np.any(in_window):
            # Average activations within window
            downsampled[tr_idx] = activations[in_window].mean(axis=0)
        # else: leave as zeros (no frames in this TR)

    return downsampled


def convolve_with_hrf(activations, tr, hrf_model='spm'):
    """
    Convolve activations with HRF.

    Parameters
    ----------
    activations : np.ndarray
        Activation array (n_trs, n_features)
    tr : float
        Repetition time in seconds
    hrf_model : str, default='spm'
        HRF model ('spm', 'glover', 'spm + derivative', etc.)

    Returns
    -------
    np.ndarray
        Convolved activations (n_trs, n_features)
    """
    from scipy.signal import convolve

    n_trs, n_features = activations.shape

    # Create HRF using nilearn
    frame_times = np.arange(n_trs) * tr
    hrf, _ = compute_regressor(
        exp_condition=np.ones((n_trs, 3)),  # Dummy condition (onset, duration, amplitude)
        hrf_model=hrf_model,
        frame_times=frame_times,
        oversampling=1
    )

    # Get HRF kernel (first column if multiple outputs)
    if hrf.ndim > 1:
        hrf_kernel = hrf[:, 0]
    else:
        hrf_kernel = hrf

    # Normalize HRF
    hrf_kernel = hrf_kernel / hrf_kernel.sum()

    # Convolve each feature
    convolved = np.zeros_like(activations)
    for i in range(n_features):
        convolved[:, i] = convolve(activations[:, i], hrf_kernel, mode='same')

    return convolved


def apply_pca(activations, n_components=50, variance_threshold=0.9):
    """
    Apply PCA to reduce dimensionality of activations.

    Parameters
    ----------
    activations : np.ndarray
        Activation array (n_samples, n_features)
    n_components : int, default=50
        Number of components to keep
    variance_threshold : float, default=0.9
        Keep enough components to explain this much variance

    Returns
    -------
    tuple
        (reduced_activations, pca_model, variance_explained)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Standardize first
    scaler = StandardScaler()
    activations_scaled = scaler.fit_transform(activations)

    # Fit PCA
    pca = PCA(n_components=min(n_components, activations.shape[1]))
    activations_pca = pca.fit_transform(activations_scaled)

    # Check variance explained
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    n_needed = np.searchsorted(cumsum_var, variance_threshold) + 1

    print(f"PCA: {n_needed} components explain {variance_threshold*100:.1f}% variance")
    print(f"Using {pca.n_components_} components (explains {cumsum_var[-1]*100:.1f}% variance)")

    return activations_pca, pca, pca.explained_variance_ratio_


def create_simple_proxy_features(events_df, n_trs, tr):
    """
    Create simplified proxy features from behavioral events (for quick demo).

    Instead of full RL training, use button presses and game events as features.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events dataframe
    n_trs : int
        Number of TRs
    tr : float
        Repetition time

    Returns
    -------
    dict
        Dictionary with 'button_features' and 'event_features' arrays
    """
    # Button press features (one-hot encoding of button types)
    buttons = ['A', 'B', 'LEFT', 'RIGHT', 'UP', 'DOWN']
    button_features = np.zeros((n_trs, len(buttons)))

    for idx, button in enumerate(buttons):
        button_events = events_df[events_df['trial_type'] == button]
        for _, event in button_events.iterrows():
            tr_idx = int(event['onset'] / tr)
            if tr_idx < n_trs:
                button_features[tr_idx, idx] = 1

    # Game event features
    game_events = ['Kill/stomp', 'Kill/kick', 'Hit/life_lost',
                   'Powerup_collected', 'Coin_collected']
    event_features = np.zeros((n_trs, len(game_events)))

    for idx, event_type in enumerate(game_events):
        type_events = events_df[events_df['trial_type'] == event_type]
        for _, event in type_events.iterrows():
            tr_idx = int(event['onset'] / tr)
            if tr_idx < n_trs:
                event_features[tr_idx, idx] = 1

    return {
        'button_features': button_features,
        'event_features': event_features,
        'combined_features': np.concatenate([button_features, event_features], axis=1)
    }


def load_pretrained_model(model_path, device='cpu'):
    """
    Load a pre-trained CNN model.

    Parameters
    ----------
    model_path : str or Path
        Path to model checkpoint
    device : str, default='cpu'
        Device for model

    Returns
    -------
    SimpleCNN
        Loaded model
    """
    model = SimpleCNN()

    # Load state dict
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model
