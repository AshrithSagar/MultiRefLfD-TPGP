"""
lfd/utils/utils.py \n
General utilities
"""

import random
from typing import List, Optional

import numpy as np
import pyro
import torch
from deprecated import deprecated
from numpy.typing import NDArray
from torch import Tensor

from .lasa import getA, load_data_with_phi


def set_seed(seed: int = 42):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def load_fdset(key: str) -> NDArray:
    """
    Load and transform dataset.

    :param key: Key to load the dataset.
    :return: Transformed demonstration set (n_frames+1, n_traj, n_length, n_dim)
    """
    data, time = load_data_with_phi(key)
    As, Bs = get_frames(data)
    fdset = transform_data(data, As, Bs)
    return fdset


def get_frames(data):
    """
    Computes orientation (As) and origin (Bs) of start and end frames for each trajectory.
    The x-axis of the frames is in the direction of starting and ending of trajectory.

    :param data: (n_traj, n_length, 3) with time, x, y.
    :return As: Transformation matrices (2, n_traj, 3, 3)
    :return Bs: Origins of the frames / Translation vectors (2, n_traj, 3)
    """
    As = []
    Bs = []
    for d in data:
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


def transform_data(
    D0: NDArray, As: Optional[NDArray] = None, Bs: Optional[NDArray] = None
) -> NDArray:
    """
    Transform data using transformation matrices and translation vectors.

    :param D0: Demonstration set in global frame to be transformed (n_traj, n_length, n_dim)
    :param As: Transformation matrices (n_frames, n_traj, n_dim, n_dim)
    :param Bs: Translation vectors (n_frames, n_traj, n_dim)
    :return: Transformed demonstration set (n_frames+1, n_traj, n_length, n_dim)
    """
    X = [D0]
    _As, _Bs = get_frames(D0)
    if As is None:
        As = _As
    if Bs is None:
        Bs = _Bs
    n_frames = As.shape[0]

    for m in range(n_frames):
        Hm = As[m]  # (n_traj, n_dim, n_dim)
        tm = Bs[m]  # (n_traj, n_dim)
        Dm = np.einsum("nld,ndk->nlk", D0, Hm) + tm[:, None, :]
        X.append(Dm)

    X = np.stack(X, axis=0)  # (n_frames+1, n_traj, n_length, n_dim)
    return X


# Alternate
@deprecated
def transform_data_2(
    D0: NDArray, As: Optional[NDArray] = None, Bs: Optional[NDArray] = None
) -> NDArray:
    X = [D0]
    _As, _Bs = get_frames(D0)
    if As is None:
        As = _As
    if Bs is None:
        Bs = _Bs
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


def vectorized_derivative(x: NDArray):
    dt = 1 / (x.shape[1] - 1)
    xd_diff = np.diff(x, axis=1) / dt
    padding = np.zeros((x.shape[0], 1, x.shape[2] if x.ndim > 2 else 1))
    xd = np.concatenate((xd_diff, padding), axis=1)
    v_factor = np.cos(np.linspace(0, np.pi / 2, x.shape[1])) ** 2
    xd = xd * (v_factor[None, ..., None])
    return xd


def toTensor(x, dtype=torch.float32, device: torch.device = "cpu") -> Tensor:
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=dtype)
    return x.to(device)
