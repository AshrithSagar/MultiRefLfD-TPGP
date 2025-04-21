"""
lfd/utils/utils.py \n
General utilities
"""

import random
from typing import Union

import numpy as np
import pyro
import torch
from shapely import LineString

from .frames import DemonstrationSet, append_progress_values
from .lasa import load_data


def set_seed(seed: int = 42):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dset(key: Union[str, int], show_plot: bool = False) -> DemonstrationSet:
    """Load a dataset and convert it to a DemonstrationSet."""
    data, _, _ = load_data(key, show_plot)
    dset = append_progress_values([LineString(traj) for traj in data])
    return dset


def prepare_data(dset: DemonstrationSet, include_phi: bool = True):
    """
    Turn a DemonstrationSet of shape (N demos, L points, 3 coords)
    into X and Y matrices for GP training.

    :return X: [xi, phi] at time t
    :return Y: Delta xi = xi(t+1) − xi(t), Delta phi = phi(t+1) − phi(t)
    """
    arr = np.stack([np.array(d.coords) for d in dset])  # Shape (N, L, 3)
    X = arr[:, :-1, : 3 if include_phi else 2]  # Shape (N, L−1, 2 or 3)
    Y = arr[:, 1:, : 3 if include_phi else 2] - X  # Shape (N, L−1, 2 or 3)

    # Flatten demos & timesteps
    X_flat = X.reshape(-1, 3)  # Shape (N*(L−1), 2 or 3)
    Y_flat = Y.reshape(-1, 3)  # Shape (N*(L−1), 2 or 3)
    return X_flat, Y_flat
