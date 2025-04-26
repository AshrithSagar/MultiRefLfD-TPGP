"""
lfd/utils/utils.py \n
General utilities
"""

import random
from typing import Union

import numpy as np
import pyro
import torch
from numpy.typing import NDArray
from shapely import LineString

from .frames import DemonstrationSet, append_progress_values
from .lasa import load_data, load_data3, plot_trajectories


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


def load_fdset(key: str) -> NDArray:
    """Load a dataset including frames."""
    Data, time = load_data3(key)
    As, Bs = plot_trajectories(Data)

    n_frames = 2
    n_traj, n_length, n_dim = Data.shape
    X = [Data]
    for m in range(n_frames):
        Dm = []
        for n in range(n_traj):
            _0_dn = Data[n]
            Hm = As[m, n]
            tm = Bs[m, n]
            _m_dn = _0_dn @ Hm + tm
            Dm.append(_m_dn)
        Dm = np.array(Dm)
        X.append(Dm)
    X = np.array(X)  # (n_frames+1, n_traj, n_length, n_dim)
    return X


def load_fdset2(key: str) -> NDArray:
    """Load a dataset including frames."""
    Data, time = load_data3(key)
    As, Bs = plot_trajectories(Data)

    n_frames = 2
    X = [Data]
    for m in range(n_frames):
        Hm = As[m]  # (n_traj, n_dim, n_dim)
        tm = Bs[m]  # (n_traj, n_dim)
        Dm = np.einsum("nld,ndk->nlk", Data, Hm) + tm[:, None, :]
        X.append(Dm)
    X = np.stack(X, axis=0)  # (n_frames+1, n_traj, n_length, n_dim)
    return X
