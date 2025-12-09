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
    from nilearn.glm.first_level import glover_hrf, spm_hrf

    n_trs, n_features = activations.shape

    # Create HRF kernel at TR resolution
    hrf_length = 32  # seconds
    oversampling = 16  # Higher oversampling for smoother HRF

    if hrf_model == 'spm':
        hrf_kernel = spm_hrf(tr, oversampling=oversampling, time_length=hrf_length)
    elif hrf_model == 'glover':
        hrf_kernel = glover_hrf(tr, oversampling=oversampling, time_length=hrf_length)
    else:
        # Default to SPM
        hrf_kernel = spm_hrf(tr, oversampling=oversampling, time_length=hrf_length)

    # Downsample HRF to TR resolution
    # The HRF is generated at (tr / oversampling) resolution
    # We need to take every oversampling-th point
    hrf_at_tr = hrf_kernel[::oversampling]

    # Normalize HRF
    if hrf_at_tr.sum() != 0:
        hrf_at_tr = hrf_at_tr / hrf_at_tr.sum()

    # Convolve each feature
    convolved = np.zeros_like(activations)
    for i in range(n_features):
        convolved[:, i] = convolve(activations[:, i], hrf_at_tr, mode='same')

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


def apply_pca_with_nan_handling(activations_dict, mask, n_components=50, variance_threshold=0.9):
    """
    Apply PCA to layer activations with proper NaN handling.

    Parameters
    ----------
    activations_dict : dict
        Dictionary mapping layer names to activation arrays (n_trs, n_features)
        Arrays may contain NaN values for non-gameplay periods
    mask : np.ndarray
        Boolean mask indicating valid (non-NaN) TRs (n_trs,)
    n_components : int, default=50
        Number of PCA components to keep
    variance_threshold : float, default=0.9
        Variance threshold for determining minimum components

    Returns
    -------
    dict
        Dictionary with keys:
        - 'reduced_activations': dict mapping layer names to PCA-reduced arrays (n_trs, n_components)
        - 'pca_models': dict mapping layer names to fitted PCA objects
        - 'variance_explained': dict mapping layer names to variance ratios
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print("\nApplying PCA to layer activations...")

    reduced_activations = {}
    pca_models = {}
    variance_explained = {}

    for layer_name, acts in activations_dict.items():
        print(f"\n  {layer_name}:")

        # Extract valid (non-NaN) samples for PCA fitting
        valid_acts = acts[mask]

        print(f"    Original features: {valid_acts.shape[1]}")

        # Standardize
        scaler = StandardScaler()
        valid_acts_scaled = scaler.fit_transform(valid_acts)

        # Fit PCA
        n_comp = min(n_components, valid_acts.shape[1], valid_acts.shape[0])
        pca = PCA(n_components=n_comp)
        valid_acts_pca = pca.fit_transform(valid_acts_scaled)

        # Check variance explained
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        n_needed = np.searchsorted(cumsum_var, variance_threshold) + 1

        print(f"    PCA: {n_needed} components explain {variance_threshold*100:.1f}% variance")
        print(f"    Using {pca.n_components_} components (explains {cumsum_var[-1]*100:.1f}% variance)")

        # Create output array with NaN for invalid TRs
        reduced = np.full((acts.shape[0], pca.n_components_), np.nan)
        reduced[mask] = valid_acts_pca

        reduced_activations[layer_name] = reduced
        pca_models[layer_name] = {'pca': pca, 'scaler': scaler}
        variance_explained[layer_name] = pca.explained_variance_ratio_

    print("\n✓ PCA complete")

    return {
        'reduced_activations': reduced_activations,
        'pca_models': pca_models,
        'variance_explained': variance_explained
    }


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
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

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


def action_to_buttons(action):
    """
    Convert action index to NES button array.

    COMPLEX_MOVEMENT action space (12 actions):
    0: NOOP
    1: right
    2: right + A (jump)
    3: right + B (run)
    4: right + A + B (run + jump)
    5: A (jump)
    6: left
    7: left + A
    8: left + B
    9: left + A + B
    10: down
    11: up

    Parameters
    ----------
    action : int
        Action index (0-11)

    Returns
    -------
    list
        Array of 9 buttons: [B, null, select, start, up, down, left, right, A]
    """
    buttons = [0] * 9  # 9 buttons for NES

    if action == 1:  # right
        buttons[7] = 1
    elif action == 2:  # right + A
        buttons[7] = 1
        buttons[8] = 1
    elif action == 3:  # right + B
        buttons[7] = 1
        buttons[0] = 1
    elif action == 4:  # right + A + B
        buttons[7] = 1
        buttons[8] = 1
        buttons[0] = 1
    elif action == 5:  # A
        buttons[8] = 1
    elif action == 6:  # left
        buttons[6] = 1
    elif action == 7:  # left + A
        buttons[6] = 1
        buttons[8] = 1
    elif action == 8:  # left + B
        buttons[6] = 1
        buttons[0] = 1
    elif action == 9:  # left + A + B
        buttons[6] = 1
        buttons[8] = 1
        buttons[0] = 1
    elif action == 10:  # down
        buttons[5] = 1
    elif action == 11:  # up
        buttons[4] = 1
    # action == 0 is NOOP (all zeros)

    return buttons


def play_agent_episode(model, level, sourcedata_path, max_steps=5000, device='cpu', fps=60):
    """
    Play one episode with trained agent.

    Parameters
    ----------
    model : SimpleCNN
        Trained model
    level : str
        Level to play (e.g., 'Level1-1')
    sourcedata_path : Path
        Path to sourcedata (for ROM)
    max_steps : int, default=5000
        Maximum steps per episode
    device : str, default='cpu'
        Device for model
    fps : int, default=60
        Target frames per second for display

    Returns
    -------
    dict
        Episode statistics (steps, reward, completed)
    """
    import retro
    import cv2
    import time

    # Import ROM from custom path
    rom_path = sourcedata_path / 'mario' / 'stimuli'
    if rom_path.exists():
        retro.data.Integrations.add_custom_path(str(rom_path))

    # Create environment WITHOUT render_mode (we'll handle display ourselves)
    env = retro.make(
        game='SuperMarioBros-Nes',
        state=level,
        inttype=retro.data.Integrations.ALL,
        render_mode=None
    )

    # Window name for OpenCV
    window_name = 'Mario Agent'
    window_open = True
    user_quit = False
    
    # Create window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 512, 480)

    def check_window_closed():
        """Check if OpenCV window was closed."""
        try:
            # getWindowProperty returns -1 if window is closed
            return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    try:
        # Play one episode
        obs, info = env.reset()
        frame_stack = [preprocess_frame(obs)] * 4

        episode_reward = 0
        done = False
        step = 0
        terminated = False
        frame_time = 1.0 / fps

        model = model.to(device)
        model.eval()

        while not done and step < max_steps and not user_quit:
            start_time = time.time()
            
            # Check if window was closed
            if check_window_closed():
                print("\nWindow closed by user.")
                user_quit = True
                break

            # Stack frames
            state = torch.from_numpy(np.stack(frame_stack)).unsqueeze(0).float().to(device)

            # Get action from model
            with torch.no_grad():
                action_logits, value = model(state)
                action_probs = torch.softmax(action_logits, dim=-1)
                action = torch.argmax(action_probs, dim=-1).item()

            # Convert to button array
            buttons = action_to_buttons(action)

            # Step environment
            try:
                next_obs, reward, terminated, truncated, info = env.step(buttons)
            except Exception as e:
                print(f"\nEnvironment error: {e}")
                user_quit = True
                break

            done = terminated or truncated

            # Display frame using OpenCV
            # Convert RGB to BGR for OpenCV
            display_frame = cv2.cvtColor(next_obs, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, display_frame)

            # Update state
            frame_stack = frame_stack[1:] + [preprocess_frame(next_obs)]
            episode_reward += reward
            step += 1

            # Check for key press (q or ESC to quit)
            # waitKey also pumps the event loop which is necessary for window updates
            elapsed = time.time() - start_time
            wait_ms = max(1, int((frame_time - elapsed) * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\nUser pressed quit key.")
                user_quit = True
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        user_quit = True

    finally:
        # Clean up
        try:
            env.close()
        except Exception:
            pass
        
        # Destroy OpenCV window
        try:
            cv2.destroyWindow(window_name)
            # Multiple waitKey calls help ensure window is destroyed
            for _ in range(5):
                cv2.waitKey(1)
        except Exception:
            pass

    return {
        'steps': step,
        'reward': episode_reward,
        'completed': terminated and not user_quit
    }

def extract_layer_activations(model, level, sourcedata_path, max_steps=1000, device='cpu'):
    """
    Extract CNN layer activations from agent gameplay.

    Parameters
    ----------
    model : SimpleCNN
        Trained model
    level : str
        Level to play (e.g., 'Level1-1')
    sourcedata_path : Path
        Path to sourcedata (for ROM)
    max_steps : int, default=1000
        Maximum steps per episode
    device : str, default='cpu'
        Device for model

    Returns
    -------
    dict
        Dictionary mapping layer names to activation arrays (steps, features)
    """
    import retro

    # Import ROM from custom path
    rom_path = sourcedata_path / 'mario' / 'stimuli'
    if rom_path.exists():
        retro.data.Integrations.add_custom_path(str(rom_path))

    # Create environment
    env = retro.make(
        game='SuperMarioBros-Nes',
        state=level,
        inttype=retro.data.Integrations.ALL,
        render_mode=None
    )

    model.eval()
    model.to(device)

    # Storage for activations
    layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'linear']
    activations_dict = {name: [] for name in layer_names}

    # Reset environment
    obs, _ = env.reset()
    frame_stack = [preprocess_frame(obs)] * 4

    for step in range(max_steps):
        # Create input
        state = create_frame_stack(frame_stack)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)

        # Get activations
        with torch.no_grad():
            acts = model(state_tensor, return_activations=True)

        # Store activations (flatten spatial dimensions)
        for layer_name in layer_names:
            act = acts[layer_name].cpu().numpy()
            if len(act.shape) == 4:  # Conv layers (B, C, H, W)
                act = act.reshape(act.shape[0], -1)  # Flatten to (B, C*H*W)
            activations_dict[layer_name].append(act[0])  # Remove batch dim

        # Get action
        action_logits = acts['action_logits']
        action = torch.argmax(action_logits, dim=1).item()
        buttons = action_to_buttons(action)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(buttons)
        done = terminated or truncated

        # Update frame stack
        frame_stack = frame_stack[1:] + [preprocess_frame(obs)]

        if done:
            break

    env.close()

    # Convert lists to arrays
    for layer_name in layer_names:
        activations_dict[layer_name] = np.array(activations_dict[layer_name])

    return activations_dict


def load_replay_and_extract_frames(replay_path, sourcedata_path, level_name):
    """
    Load a replay file (.bk2) and extract all frames.

    Parameters
    ----------
    replay_path : str or Path
        Path to .bk2 replay file
    sourcedata_path : Path
        Path to sourcedata directory (for ROM)
    level_name : str
        Level name (e.g., 'w1l1' -> 'Level1-1')

    Returns
    -------
    list of np.ndarray
        List of RGB frames (H, W, 3)
    """
    import retro
    import os

    # Resolve symlinks (git-annex uses symlinks)
    replay_path = Path(replay_path)
    if replay_path.is_symlink():
        replay_path = replay_path.resolve()

    if not replay_path.exists():
        raise FileNotFoundError(f"Replay file not found: {replay_path}")

    # Convert level name from annotation format to retro format
    # w1l1 -> Level1-1, w4l2 -> Level4-2, etc.
    if level_name.startswith('w') and 'l' in level_name:
        world = level_name[1:level_name.index('l')]
        level = level_name[level_name.index('l')+1:]
        state_name = f'Level{world}-{level}'
    else:
        state_name = level_name

    print(f"    Level format: {level_name} -> {state_name}")

    # Import ROM from custom path
    rom_path = sourcedata_path / 'mario' / 'stimuli'
    if rom_path.exists():
        retro.data.Integrations.add_custom_path(str(rom_path))
    else:
        raise FileNotFoundError(f"ROM path not found: {rom_path}")

    # Load the replay using retro.Movie
    movie = retro.Movie(str(replay_path))

    # Create environment with the replay
    env = retro.make(
        game='SuperMarioBros-Nes',
        state=state_name,
        inttype=retro.data.Integrations.ALL,
        render_mode=None,
        players=movie.players
    )

    # Configure movie with environment
    movie.configure(state_name, env.em)

    frames = []
    obs, _ = env.reset()
    frames.append(obs.copy())

    step_count = 0
    max_steps = 50000  # Safety limit

    while movie.step():
        if step_count >= max_steps:
            print(f"    Warning: Reached max steps ({max_steps}), truncating replay")
            break

        # Get keys from movie
        keys = []
        for p in range(movie.players):
            for i in range(env.num_buttons):
                keys.append(movie.get_key(i, p))

        # Step environment with replay actions
        obs, reward, terminated, truncated, info = env.step(keys)
        frames.append(obs.copy())
        step_count += 1

        if terminated or truncated:
            break

    env.close()
    movie.close()

    return frames


def extract_activations_from_replay(model, replay_path, sourcedata_path, level_name,
                                     frame_start=None, frame_stop=None, device='cpu'):
    """
    Extract RL activations from a replay file.

    Parameters
    ----------
    model : SimpleCNN
        Trained RL model
    replay_path : str or Path
        Path to .bk2 replay file
    sourcedata_path : Path
        Path to sourcedata directory
    level_name : str
        Level name (e.g., 'w1l1')
    frame_start : int, optional
        Start frame index (for sub-segment extraction)
    frame_stop : int, optional
        Stop frame index (for sub-segment extraction)
    device : str, default='cpu'
        Device for computation

    Returns
    -------
    dict
        Dictionary mapping layer names to activation arrays (n_frames, n_features)
    """
    # Load replay and extract frames
    print(f"  Loading replay: {Path(replay_path).name}")
    frames = load_replay_and_extract_frames(replay_path, sourcedata_path, level_name)

    # Extract sub-segment if specified
    if frame_start is not None or frame_stop is not None:
        frame_start = frame_start or 0
        frame_stop = frame_stop or len(frames)
        frames = frames[int(frame_start):int(frame_stop)]

    print(f"  Processing {len(frames)} frames...")

    # Preprocess all frames
    preprocessed = [preprocess_frame(f) for f in frames]

    # Create frame stacks (4 consecutive frames per timepoint)
    frame_stacks = []
    for i in range(len(preprocessed)):
        if i < 3:
            # For first 3 frames, repeat first frame to fill stack
            stack = [preprocessed[0]] * (3 - i) + preprocessed[:i+1]
        else:
            stack = preprocessed[i-3:i+1]
        frame_stacks.append(np.stack(stack, axis=0))

    frame_stacks = np.array(frame_stacks)  # (n_frames, 4, 84, 84)

    # Extract activations
    layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'linear']
    activations = extract_activations_from_frames(
        model, frame_stacks, layer_names, device=device, batch_size=32
    )

    return activations


def align_activations_to_bold(model, subject, session, runs, sourcedata_path,
                               tr=1.49, device='cpu', apply_hrf=True, bold_imgs=None):
    """
    Align RL activations to BOLD data using replay files and annotations.

    This function:
    1. Loads annotation files to find gym-retro_game segments
    2. For each segment, loads the replay and extracts RL activations
    3. Downsamples activations from 60Hz to TR
    4. Masks inter-game periods with NaN
    5. Applies HRF convolution (optional)

    Parameters
    ----------
    model : SimpleCNN
        Trained RL model
    subject : str
        Subject ID (e.g., 'sub-01')
    session : str
        Session ID (e.g., 'ses-010')
    runs : list of str
        List of run IDs (e.g., ['run-1', 'run-2', ...])
    sourcedata_path : Path
        Path to sourcedata directory
    tr : float, default=1.49
        Repetition time in seconds
    device : str, default='cpu'
        Device for computation
    apply_hrf : bool, default=True
        Whether to apply HRF convolution
    bold_imgs : list of nibabel.Nifti1Image, optional
        List of BOLD images (one per run) to get actual TR count.
        If None, TR count will be estimated from event timing.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'activations': dict mapping layer names to arrays (n_trs_total, n_features)
        - 'mask': boolean array indicating valid (gameplay) TRs (n_trs_total,)
        - 'run_info': list of dicts with info per run
    """
    import pandas as pd
    from pathlib import Path

    print(f"\n{'='*70}")
    print(f"Aligning RL activations to BOLD for {subject} {session}")
    print(f"{'='*70}\n")

    model.eval()
    model.to(device)

    layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'linear']

    # Storage for all runs
    all_run_activations = {layer: [] for layer in layer_names}
    all_run_masks = []
    run_info = []

    for run_idx, run in enumerate(runs):
        print(f"\nProcessing {run}:")
        print("-" * 50)

        # Normalize run format (run-1 -> run-01, or keep as is)
        run_num = run.split('-')[1]
        run_normalized = f"run-{int(run_num):02d}"

        # Load events from mario/ (not mario.annotations)
        events_path = (sourcedata_path / 'mario' / subject / session / 'func' /
                      f"{subject}_{session}_task-mario_{run_normalized}_events.tsv")

        if not events_path.exists():
            print(f"  ⚠ Events file not found: {events_path}")
            continue

        events_df = pd.read_csv(events_path, sep='\t')

        # Find gym-retro_game trials
        game_trials = events_df[events_df['trial_type'] == 'gym-retro_game'].copy()

        if len(game_trials) == 0:
            print(f"  ⚠ No gym-retro_game trials found in {run}")
            continue

        print(f"  Found {len(game_trials)} game trial(s)")

        # Calculate durations from onset differences (or use end of run for last trial)
        # Get run duration from BOLD image or use last event
        # For now, estimate from next event onset
        game_trials = game_trials.reset_index(drop=True)
        durations = []
        for idx in range(len(game_trials)):
            if idx < len(game_trials) - 1:
                # Duration = next trial onset - current onset
                duration = game_trials.loc[idx + 1, 'onset'] - game_trials.loc[idx, 'onset']
            else:
                # Last trial: estimate from subsequent events or use default
                all_subsequent = events_df[events_df['onset'] > game_trials.loc[idx, 'onset']]
                if len(all_subsequent) > 0:
                    duration = all_subsequent['onset'].min() - game_trials.loc[idx, 'onset']
                else:
                    # Default to 60 seconds if no subsequent events
                    duration = 60.0
            durations.append(duration)

        game_trials['duration'] = durations

        # Get actual number of TRs from BOLD if available, otherwise estimate
        if bold_imgs is not None and run_idx < len(bold_imgs):
            n_trs = bold_imgs[run_idx].shape[3]  # 4th dimension is time
            print(f"  Using actual BOLD length: {n_trs} TRs")
        else:
            # Estimate from events (fallback)
            run_duration = events_df['onset'].max() + 10.0  # Add buffer
            n_trs = int(np.ceil(run_duration / tr))
            print(f"  Estimated from events: {n_trs} TRs (duration: {run_duration:.2f}s)")
            print(f"  ⚠ Warning: Using estimated TR count. Pass bold_imgs for exact alignment.")

        # Initialize activations for this run (will be NaN for non-game periods)
        # Determine feature dimensions from first segment
        run_activations = None
        run_mask = np.zeros(n_trs, dtype=bool)

        # Process each game trial
        for bk2_idx, trial in game_trials.iterrows():
            level = trial['level']
            onset = trial['onset']
            duration = trial['duration']
            replay_file = trial['stim_file']

            print(f"\n  Repetition {bk2_idx}: {level}")
            print(f"    Onset: {onset:.2f}s, Duration: {duration:.2f}s")

            # Construct replay path
            replay_path = sourcedata_path / 'mario' / replay_file

            if not replay_path.exists():
                print(f"    ⚠ Replay file not found: {replay_path}")
                continue

            # Extract activations from this trial (full replay)
            try:
                segment_acts = extract_activations_from_replay(
                    model, replay_path, sourcedata_path, level,
                    frame_start=None, frame_stop=None, device=device
                )
            except Exception as e:
                print(f"    ⚠ Error extracting activations: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Downsample to TR
            # Frames are at 60Hz, so frame times relative to segment onset
            n_frames = list(segment_acts.values())[0].shape[0]
            frame_times = np.arange(n_frames) / 60.0  # 60 Hz

            print(f"    Extracted {n_frames} frames → downsampling to TR...")

            # Downsample each layer
            segment_acts_downsampled = {}
            for layer in layer_names:
                downsampled = downsample_activations_to_tr(
                    segment_acts[layer], frame_times, tr, duration
                )
                segment_acts_downsampled[layer] = downsampled

            # Determine which TRs this segment occupies
            tr_start_idx = int(np.floor(onset / tr))
            tr_end_idx = tr_start_idx + downsampled.shape[0]

            # Clip to run length
            tr_end_idx = min(tr_end_idx, n_trs)
            n_trs_segment = tr_end_idx - tr_start_idx

            print(f"    → {n_trs_segment} TRs (indices {tr_start_idx}-{tr_end_idx})")

            # Initialize run_activations if first segment
            if run_activations is None:
                n_features = {layer: segment_acts_downsampled[layer].shape[1]
                             for layer in layer_names}
                run_activations = {layer: np.full((n_trs, n_features[layer]), np.nan)
                                  for layer in layer_names}

            # Place downsampled activations in run array
            for layer in layer_names:
                # Make sure we don't exceed array bounds
                acts_to_place = segment_acts_downsampled[layer][:n_trs_segment]
                run_activations[layer][tr_start_idx:tr_end_idx] = acts_to_place

            # Update mask
            run_mask[tr_start_idx:tr_end_idx] = True

        # Store run results
        if run_activations is not None:
            for layer in layer_names:
                all_run_activations[layer].append(run_activations[layer])
            all_run_masks.append(run_mask)

            run_info.append({
                'run': run,
                'n_trs': n_trs,
                'n_valid_trs': run_mask.sum(),
                'n_segments': len(game_trials)
            })

            print(f"\n  ✓ {run}: {run_mask.sum()}/{n_trs} TRs with gameplay")

    # Check if any runs were processed successfully
    if len(all_run_masks) == 0:
        print(f"\n{'='*70}")
        print("❌ ERROR: No runs were successfully processed!")
        print(f"{'='*70}\n")
        print("Possible issues:")
        print("  1. Annotation files not found")
        print("  2. No gym-retro_game segments in annotation files")
        print("  3. Replay files (.bk2) not found")
        print("  4. Error during activation extraction")
        print("\nPlease check the error messages above.")
        raise ValueError("No valid runs to process. Check annotation and replay files.")

    # Concatenate all runs
    print(f"\n{'='*70}")
    print("Concatenating runs...")

    concatenated_activations = {}
    for layer in layer_names:
        concatenated_activations[layer] = np.concatenate(all_run_activations[layer], axis=0)

    concatenated_mask = np.concatenate(all_run_masks, axis=0)

    total_trs = concatenated_mask.shape[0]
    valid_trs = concatenated_mask.sum()

    print(f"  Total TRs: {total_trs}")
    print(f"  Valid TRs (gameplay): {valid_trs} ({valid_trs/total_trs*100:.1f}%)")

    # Apply HRF convolution if requested
    if apply_hrf:
        print("\nApplying HRF convolution...")
        for layer in layer_names:
            # Only convolve valid (non-NaN) activations
            # We'll convolve the entire timeseries but NaNs will remain NaN
            acts = concatenated_activations[layer].copy()

            # Replace NaN with 0 for convolution
            acts_filled = np.nan_to_num(acts, nan=0.0)

            # Convolve
            acts_convolved = convolve_with_hrf(acts_filled, tr, hrf_model='spm')

            # Restore NaN mask
            acts_convolved[~concatenated_mask] = np.nan

            concatenated_activations[layer] = acts_convolved

        print("  ✓ HRF convolution complete")

    print(f"\n{'='*70}")
    print("✓ Alignment complete!")
    print(f"{'='*70}\n")

    return {
        'activations': concatenated_activations,
        'mask': concatenated_mask,
        'run_info': run_info
    }
