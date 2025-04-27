"""
lfd/utils/utils.py \n
General utilities
"""

import random
from typing import List

import numpy as np
import pyro
import torch
from numpy.typing import NDArray

from .lasa import getA, load_data3


def set_seed(seed: int = 42):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_fdset(key: str) -> NDArray:
    """
    Load and transform dataset.

    :param key: Key to load the dataset.
    :return: Transformed demonstration set (n_frames+1, n_traj, n_length, n_dim)
    """
    Data, time = load_data3(key)
    As, Bs = get_frames(Data)
    fdset = transform_data(Data, As, Bs)
    return fdset


def get_frames(Data):
    """
    Computes orientation (As) and origin (Bs) of start and end frames for each trajectory.

    :param Data: ndarray of shape (n_traj, n_length, 3) with time, x, y.
    :return As: shape (2, n_traj, 3, 3) - Transformation matrices
    :return Bs: shape (2, n_traj, 3) - Origins of the frames / Translation vectors
    """
    As = []
    Bs = []
    for d in Data:
        a1 = (d[100] - d[0])[1:]
        A1 = getA(a1)
        b1 = d[0].copy()
        b1[0] = 0

        a2 = (d[-100] - d[-1])[1:]
        A2 = getA(a2)
        b2 = d[-1].copy()
        b2[0] = 0

        As.append([A1, A2])
        Bs.append([b1, b2])

    As = np.array(As).transpose(1, 0, 2, 3)
    Bs = np.array(Bs).transpose(1, 0, 2)
    return As, Bs


def transform_data(D0: NDArray, As: NDArray, Bs: NDArray) -> NDArray:
    """
    Transform data using transformation matrices and translation vectors.

    :param D0: Demonstration set in global frame to be transformed (n_traj, n_length, n_dim)
    :param As: Transformation matrices (n_frames, n_traj, n_dim, n_dim)
    :param Bs: Translation vectors (n_frames, n_traj, n_dim)
    :return: Transformed demonstration set (n_frames+1, n_traj, n_length, n_dim)
    """
    X = [D0]
    n_frames = As.shape[0]
    n_traj, n_length, n_dim = D0.shape

    for m in range(n_frames):
        Dm: List[NDArray] = []
        for n in range(n_traj):
            _0_dn: NDArray = D0[n]  # (n_length, n_dim)
            Hm: NDArray = As[m, n]  # (n_dim, n_dim)
            tm: NDArray = Bs[m, n]  # (n_dim,)
            _m_dn = _0_dn @ Hm + tm  # (n_length, n_dim)
            Dm.append(_m_dn)
        Dm = np.array(Dm)  # (n_traj, n_length, n_dim)
        X.append(Dm)

    X = np.array(X)  # (n_frames+1, n_traj, n_length, n_dim)
    return X


def transform_data2(D0: NDArray, As: NDArray, Bs: NDArray) -> NDArray:
    """
    Transform data using transformation matrices and translation vectors.

    :param D0: Demonstration set in global frame to be transformed (n_traj, n_length, n_dim)
    :param As: Transformation matrices (n_frames, n_traj, n_dim, n_dim)
    :param Bs: Translation vectors (n_frames, n_traj, n_dim)
    :return: Transformed demonstration set (n_frames+1, n_traj, n_length, n_dim)
    """
    X = [D0]
    n_frames = As.shape[0]

    for m in range(n_frames):
        Hm = As[m]  # (n_traj, n_dim, n_dim)
        tm = Bs[m]  # (n_traj, n_dim)
        Dm = np.einsum("nld,ndk->nlk", D0, Hm) + tm[:, None, :]
        X.append(Dm)

    X = np.stack(X, axis=0)  # (n_frames+1, n_traj, n_length, n_dim)
    return X
