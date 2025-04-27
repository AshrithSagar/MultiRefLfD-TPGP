"""
lfd/utils/alignment.py \n
Alignment of demonstations
"""

from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from deprecated import deprecated
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist


def phi2index(dn: NDArray, phi: float) -> int:
    """
    Get the index of point in the demonstration with progress value phi.

    :param dn: Demonstration (n_length, n_dim)
    :param phi: Progress value
    :return: Index
    """
    return np.argmin(np.abs(dn[:, 0] - phi))


def resample_keypoint(dn, keyp_idx: int, align_idx: int) -> NDArray:
    """
    Resample a demonstration aligning the keypoint at index
    keyp_idx to shift to the one at index align_idx.

    :param dn: Demonstration (n_length, n_dim)
    :param keyp_idx: Index of the keypoint in the demonstration
    :param align_idx: Index to align the keypoint to
    :return: Resampled demonstration (n_length, n_dim)
    """
    phi, xi, n = dn[:, 0], dn[:, 1:], len(dn)

    s1 = align_idx / keyp_idx if keyp_idx else 1.0
    s2 = (n - align_idx - 1) / (n - keyp_idx - 1) if keyp_idx != n - 1 else 1.0

    eps = 1e-6  # Avoid division by zero
    new_phi = np.where(
        np.arange(n) <= keyp_idx,
        phi * s1,
        phi[keyp_idx] * s1 + (phi - phi[keyp_idx]) * s2,
    )
    new_phi = np.maximum.accumulate(new_phi + eps * np.arange(n))

    new_xi = interp1d(new_phi, xi, axis=0, kind="linear", fill_value="extrapolate")(phi)
    return np.column_stack((phi, new_xi))


def computeP(fdset: NDArray) -> NDArray:
    """
    Compute P(m, i), which is the keypoint progress value for
    trajectory i in frame m, computed as the median of the
    progress values from B over all other trajectories.
    A(m, i, j) is the index of the closest point in trajectory i
    to all other points in trajectory j, w.r.t. frame m.
    B(m, i, j) is the corresponding progress value, at index A(m, i, j),
    i.e., normalised time found by diving by the number of points in the trajectory

    :param fdset: Transformed demonstration set (n_frames, n_traj, n_length, n_dim)
    :return P: P matrix (n_frames, n_traj)
    """
    n_frames, n_traj, n_length, n_dim = fdset.shape
    n_frames -= 1

    A = np.zeros((n_frames, n_traj, n_traj), dtype=int)
    B = np.zeros((n_frames, n_traj, n_traj), dtype=float)

    for m in range(1, n_frames + 1):
        Dm = fdset[m]  # (n_traj, n_length, n_dim)
        for i in range(n_traj):
            _m_xi = Dm[i]  # (n_length, n_dim)

            # Precompute distances between _m_xi and all _m_xj
            # (n_traj, n_length, n_length)
            dists_all = np.array([cdist(_m_xi, Dm[j]) for j in range(n_traj)])
            min_dist_per_point = np.min(dists_all, axis=2)  # (n_traj, n_length)

            h = np.argmin(min_dist_per_point, axis=1)  # (n_traj,)
            A[m - 1, i, :] = h
            B[m - 1, i, :] = Dm[np.arange(n_traj), h, 0]  # phi

    P = np.median(B, axis=2)  # (n_frames, n_traj)
    return P


# Alternate
@deprecated
def computeP_2(fdset: NDArray) -> NDArray:
    n_frames, n_traj, n_length, n_dim = fdset.shape
    n_frames -= 1

    A = np.zeros((n_frames, n_traj, n_traj), dtype=int)
    B = np.zeros((n_frames, n_traj, n_traj), dtype=float)

    for m in range(1, n_frames + 1):
        Dm = fdset[m]  # (n_traj, n_length, n_dim)
        for i in range(n_traj):
            _m_xi = Dm[i]  # (n_length, n_dim)
            for j in range(n_traj):
                _m_xj = Dm[j]  # (n_length, n_dim)
                dists: NDArray  # (n_length, n_length)
                dists = np.linalg.norm(_m_xi[:, None, :] - _m_xj[None, :, :], axis=-1)
                min_dist_per_point = np.min(dists, axis=1)  # (n_length,)
                h = np.argmin(min_dist_per_point)
                A[m - 1, i, j] = h
                B[m - 1, i, j] = Dm[i, h, 0]  # phi

    P = np.median(B, axis=2)  # (n_frames, n_traj)
    return P


def align_demonstrations(fdset: NDArray, P: Optional[NDArray] = None) -> NDArray:
    """
    Resample each demonstration in the demonstration set over
    different frames, aligning each keypoint found in a frame
    to the same progress value.

    :param fdset: Transformed demonstration set (n_frames, n_traj, n_length, n_dim)
    :param P: P matrix (n_frames, n_traj)
    :return D0_star: Aligned demonstration set (n_traj, n_length, n_dim)
    """
    n_frames, n_traj, n_length, n_dim = fdset.shape
    n_frames -= 1

    P = computeP(fdset) if P is None else P  # (n_frames, n_traj)
    D0_star: NDArray = fdset[0].copy()  # (n_traj, n_length, n_dim)
    aligned_phi: NDArray = np.median(P, axis=1)  # (n_frames,)

    for m in range(n_frames - 1):
        for n in range(n_traj):
            dn: NDArray = D0_star[n]  # (n_length, n_dim)
            align_idx = phi2index(dn, aligned_phi[m])
            keyp_idx = phi2index(dn, P[m, n])
            D0_star[n][:-1] = resample_keypoint(dn[:-1], keyp_idx, align_idx)

    return D0_star


def plot_keypoints(fdset: NDArray, P: Optional[NDArray] = None, alpha: float = 0.5):
    """
    Plot the keypoints for each demonstration and for each frame.

    :param fdset: Transformed demonstration set (n_frames, n_traj, n_length, n_dim)
    :param P: P matrix (n_frames, n_traj)
    :param alpha: Transparency of the demonstrations
    """
    n_frames, n_traj, n_length, n_dim = fdset.shape
    n_frames -= 1

    P = computeP(fdset) if P is None else P  # (n_frames, n_traj)

    cmap = mpl.colormaps["Set1"]
    colors = cmap(np.linspace(0, 1, n_frames))
    fig, ax = plt.subplots(1, n_dim - 1, figsize=(10, 5))
    ax: List[plt.Axes]

    for d in range(n_dim - 1):
        for n in range(n_traj):
            # Plot original demonstration
            dn = fdset[0, n]  # (n_length, n_dim)
            ax[d].plot(dn[:-1, 0], dn[:-1, d + 1], alpha=alpha, zorder=1)

            # Mark endpoints
            ax[d].scatter(dn[0, 0], dn[0, d + 1], c="k", zorder=2)
            ax[d].scatter(dn[-2, 0], dn[-2, d + 1], c="b", zorder=2)

            # Mark keypoints
            for m in range(n_frames):
                keyp = dn[phi2index(dn, P[m, n])]  # (n_dim,)
                ax[d].scatter(keyp[0], keyp[d + 1], c=[colors[m]], marker="x", zorder=3)

        ax[d].set_xlabel(r"Progress value $\varphi$")
        ax[d].set_ylabel(r"Cartesian dimension $\xi_{%d}$" % (d + 1))
        ax[d].grid(True)

    plt.tight_layout()


def plot_alignments(
    fdset: NDArray,
    D0_star: NDArray,
    P: Optional[NDArray] = None,
    alpha: float = 0.5,
    show_original: bool = False,
):
    """
    Plot the aligned demonstrations that are aligned to
    a common progress value.

    :param fdset: Transformed demonstration set (n_frames, n_traj, n_length, n_dim)
    :param D0_star: Aligned demonstration set (n_traj, n_length, n_dim)
    :param P: P matrix (n_frames, n_traj)
    :param alpha: Transparency of the demonstrations
    :param show_original: Show original demonstrations overlayed
    """
    n_frames, n_traj, n_length, n_dim = fdset.shape
    n_frames -= 1

    P = computeP(fdset) if P is None else P  # (n_frames, n_traj)
    aligned_phi: NDArray = np.median(P, axis=1)  # (n_frames,)

    cmap = mpl.colormaps["Set1"]
    colors = cmap(np.linspace(0, 1, n_frames))
    fig, ax = plt.subplots(1, n_dim - 1, figsize=(10, 5))
    ax: List[plt.Axes]

    for d in range(n_dim - 1):
        for n in range(n_traj):
            # Plot aligned demonstration
            dn_star = D0_star[n]  # (n_length, n_dim)
            ax[d].plot(dn_star[:-1, 0], dn_star[:-1, d + 1], alpha=alpha, zorder=1)

            # Mark endpoints
            ax[d].scatter(dn_star[0, 0], dn_star[0, d + 1], c="k", zorder=2)
            ax[d].scatter(dn_star[-2, 0], dn_star[-2, d + 1], c="b", zorder=2)

            # Plot original demonstration
            if show_original:
                dn = fdset[0, n]  # (n_length, n_dim)
                ax[d].plot(dn[:-1, 0], dn[:-1, d + 1], alpha=alpha, zorder=1)

            for m in range(n_frames):
                # Mark aligned keypoints
                keyp = dn_star[phi2index(dn_star, aligned_phi[m])]  # (n_dim,)
                ax[d].scatter(keyp[0], keyp[d + 1], c=[colors[m]], marker="x", zorder=3)

                # Mark keypoints
                if show_original:
                    keyp = dn[phi2index(dn, P[m, n])]  # (n_dim,)
                    ax[d].scatter(
                        keyp[0], keyp[d + 1], c=[colors[m]], marker="x", zorder=3
                    )

        ymin, ymax = ax[d].get_ylim()
        ax[d].vlines(aligned_phi, ymin, ymax, colors=colors, linestyles="--", zorder=4)
        ax[d].set_xlabel(r"Progress value $\varphi$")
        ax[d].set_ylabel(r"Cartesian dimension $\xi_{%d}$" % (d + 1))
        ax[d].grid(True)

    plt.tight_layout()
