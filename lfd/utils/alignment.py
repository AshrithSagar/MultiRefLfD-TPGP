"""
lfd/utils/alignment.py \n
Alignment of demonstations
"""

import math
from typing import List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from deprecated import deprecated
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from shapely import LineString, Point, get_point, shortest_line

from .frames import Demonstration, DemonstrationSet, Frame


@deprecated
def compute_h(dset: DemonstrationSet, i: int, j: int) -> Tuple[int, int]:
    """Given two trajectory indices i and j, compute A(i, j), A(j, i)"""

    def closest_vertex_index(d: Demonstration, p: Point) -> int:
        """Index of the vertex in demonstration nearest to point"""
        best_idx, best_dist = None, float("inf")
        for idx, (x, y, _) in enumerate(d.coords):
            dist = math.hypot(x - p.x, y - p.y)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    connector: LineString = shortest_line(dset[i], dset[j])
    p1, p2 = connector.coords
    idx1 = closest_vertex_index(dset[i], Point(p1))
    idx2 = closest_vertex_index(dset[j], Point(p2))
    return idx1, idx2


@deprecated
def compute_A(dset: DemonstrationSet) -> NDArray:
    """
    Compute A(i, j) for all pairs of trajectories. \n
    A(i, j) is the index of the closest point
    in trajectory i to all other points in trajectory j.
    """
    N = len(dset)
    A = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            idx1, idx2 = compute_h(dset, i, j)
            A[i, j], A[j, i] = idx1, idx2
    return A


@deprecated
def compute_B(dset: DemonstrationSet, A: Optional[NDArray] = None) -> NDArray:
    """
    Compute B(i, j) from A(i, j). \n
    B is the corresponding progress value, i.e., normalised time
    found by diving by the number of points in the trajectory
    """
    N = len(dset)
    B = np.zeros((N, N), dtype=float)
    A = compute_A(dset) if A is None else A
    for i in range(N):
        for j in range(N):
            B[i, j] = get_point(dset[i], A[i, j]).z
    return B


@deprecated
def compute_P(
    dset: DemonstrationSet, B: Optional[NDArray] = None, A: Optional[NDArray] = None
) -> NDArray:
    """
    Compute P(i, j) from B(i, j). \n
    P(i) is the keypoint progress value for trajectory i,
    computed as the median of the progress values from B
    over all other trajectories.
    """
    N = len(dset)
    P = np.zeros((N,), dtype=float)
    B = compute_B(dset, A) if B is None else B
    for i in range(N):
        P[i] = np.median(B[i, :])
    return P


# TODO: Verify
@deprecated
def resample(dset: DemonstrationSet, frames: List[Frame]) -> DemonstrationSet:
    """
    Resample each demonstration in the demonstration set over
    different frames, aligning each keypoint found in a frame
    to the same progress value.
    """
    N, M = len(dset), len(frames)
    P = np.zeros((M, N), dtype=float)
    for i, frame in enumerate(frames):
        transformed_dset = frame.transform(dset)
        P[i] = compute_P(transformed_dset)
    P = np.median(P, axis=0)

    aligned_dset: DemonstrationSet = []
    for i, demo in enumerate(dset):
        points = list(demo.coords)
        progress = [pt[2] for pt in points]
        xy = [pt[:2] for pt in points]
        x, y = zip(*xy)
        x_interp = np.interp(np.linspace(0, 1, len(progress)), progress, x)
        y_interp = np.interp(np.linspace(0, 1, len(progress)), progress, y)
        new_points = [
            (x_, y_, t)
            for x_, y_, t in zip(x_interp, y_interp, np.linspace(0, 1, len(progress)))
        ]
        aligned_dset.append(LineString(new_points))

    return aligned_dset


def resample_keyp(d0, keyp_idx: int, align_idx: int):
    phi, xi, n = d0[:, 0], d0[:, 1:], len(d0)
    s1 = align_idx / keyp_idx if keyp_idx else 1.0
    s2 = (n - align_idx - 1) / (n - keyp_idx - 1) if keyp_idx != n - 1 else 1.0
    new_phi = np.where(
        np.arange(n) <= keyp_idx,
        phi * s1,
        phi[keyp_idx] * s1 + (phi - phi[keyp_idx]) * s2,
    )
    eps = 1e-6
    new_phi = np.maximum.accumulate(new_phi + eps * np.arange(n))
    new_xi = interp1d(new_phi, xi, axis=0, kind="linear", fill_value="extrapolate")(phi)
    return np.column_stack((phi, new_xi))


def phi2index(dn, phi):
    return np.argmin(np.abs(dn[:, 0] - phi))


def computeP(fdset: NDArray) -> NDArray:
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
                dists = np.linalg.norm(
                    _m_xi[:, None, :] - _m_xj[None, :, :], axis=-1
                )  # (n_length, n_length)
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
            D0_star[n][:-1] = resample_keyp(dn[:-1], keyp_idx, align_idx)
    return D0_star


def plot_Keypoints(fdset: NDArray, P: Optional[NDArray] = None):
    n_frames, n_traj, n_length, n_dim = fdset.shape
    n_frames -= 1
    P = computeP(fdset) if P is None else P  # (n_frames, n_traj)
    cmap = mpl.colormaps["Set1"]
    colors = cmap(np.linspace(0, 1, n_frames))
    fig, ax = plt.subplots(1, n_dim - 1, figsize=(10, 5))
    ax: List[plt.Axes]
    for d in range(n_dim - 1):
        for n in range(n_traj):
            dn = fdset[0, n]  # (n_length, n_dim)
            ax[d].plot(dn[:-1, 0], dn[:-1, d + 1], alpha=0.4, zorder=1)
            ax[d].scatter(dn[0, 0], dn[0, d + 1], c="k", zorder=2)
            ax[d].scatter(dn[-2, 0], dn[-2, d + 1], c="b", zorder=2)
            for m in range(n_frames):
                keyp = dn[phi2index(dn, P[m, n])]  # (n_dim,)
                ax[d].scatter(keyp[0], keyp[d + 1], c=[colors[m]], marker="x", zorder=3)
        ax[d].set_xlabel(r"Progress value $\varphi$")
        ax[d].set_ylabel(r"Cartesian dimension $\xi_{%d}$" % (d + 1))
        ax[d].grid(True)
    plt.tight_layout()


def plot_alignments(fdset: NDArray, D0_star: NDArray, P: Optional[NDArray] = None):
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
            dn = D0_star[n]  # (n_length, n_dim)
            ax[d].plot(dn[:-1, 0], dn[:-1, d + 1], alpha=0.4, zorder=1)
            ax[d].scatter(dn[0, 0], dn[0, d + 1], c="k", zorder=2)
            ax[d].scatter(dn[-2, 0], dn[-2, d + 1], c="b", zorder=2)
            # dn = X[0, n]  # (n_length, n_dim)
            # ax[d].plot(dn[:-1, 0], dn[:-1, d + 1], alpha=0.4, zorder=1)
            for m in range(n_frames):
                keyp = dn[phi2index(dn, aligned_phi[m])]  # (n_dim,)
                ax[d].scatter(keyp[0], keyp[d + 1], c=[colors[m]], marker="x", zorder=3)
                # keyp = dn[phi2index(dn, P[m, n])]  # (n_dim,)
                # ax[d].scatter(keyp[0], keyp[d + 1], c=[colors[m]], marker="x", zorder=3)
        ax[d].vlines(aligned_phi, -10, 50, colors=colors, linestyles="--", zorder=4)
        ax[d].set_xlabel(r"Progress value $\varphi$")
        ax[d].set_ylabel(r"Cartesian dimension $\xi_{%d}$" % (d + 1))
        ax[d].grid(True)
    plt.tight_layout()
