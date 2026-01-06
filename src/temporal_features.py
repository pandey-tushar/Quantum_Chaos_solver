"""
Temporal Reservoir Computing - with memory window.

Key insight: Reservoirs work better with temporal context, not single states!
We need to process SEQUENCES, not isolated points.
"""

import numpy as np
from typing import Tuple


def create_temporal_features(
    reservoir_features: np.ndarray,
    window_size: int = 3
) -> np.ndarray:
    """
    Create temporal features by concatenating time windows.
    
    Instead of predicting y[t] from reservoir[t] alone,
    we predict y[t] from [reservoir[t-window+1], ..., reservoir[t]].
    
    This gives the readout temporal context.
    
    Args:
        reservoir_features: Shape (n_timesteps, feature_dim)
        window_size: Number of past states to include
    
    Returns:
        Temporal features: Shape (n_timesteps - window_size + 1, feature_dim * window_size)
    """
    n_timesteps, feature_dim = reservoir_features.shape
    
    if window_size > n_timesteps:
        raise ValueError(f"Window size {window_size} > {n_timesteps} timesteps")
    
    # Number of valid samples (need window_size past states)
    n_samples = n_timesteps - window_size + 1
    
    # Output dimension: feature_dim * window_size
    temporal_features = np.zeros((n_samples, feature_dim * window_size))
    
    for i in range(n_samples):
        # Concatenate features from [i, i+1, ..., i+window_size-1]
        window_features = reservoir_features[i:i+window_size].flatten()
        temporal_features[i] = window_features
    
    return temporal_features


def align_targets_with_temporal_features(
    targets: np.ndarray,
    window_size: int
) -> np.ndarray:
    """
    Align targets with temporal features.
    
    If we use window_size=3, the first prediction is at t=2
    (requires states at t=0,1,2 to predict t=2).
    
    Args:
        targets: Shape (n_timesteps, output_dim)
        window_size: Window size used for temporal features
    
    Returns:
        Aligned targets: Shape (n_timesteps - window_size + 1, output_dim)
    """
    # The first window_size-1 states are used for context only
    return targets[window_size-1:]

