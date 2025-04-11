"""
utils/alignment.py \n
Alignment of demonstations
"""

import math
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from shapely import LineString, Point, shortest_line


def compute_h(ls: List[LineString], i: int, j: int) -> Tuple[int, int]:
    """Given two trajectory indices i and j, compute A(i, j), A(j, i)"""

    def closest_vertex_index(
        ls: LineString, p: Union[Point, Tuple[float, float]]
    ) -> int:
        """Index of the vertex in ls.coords nearest to point"""
        try:
            px, py = p.x, p.y
        except AttributeError:
            px, py = p
        best_idx, best_dist = None, float("inf")
        for idx, (x, y) in enumerate(ls.coords):
            d = math.hypot(x - px, y - py)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx

    connector: LineString = shortest_line(ls[i], ls[j])
    p1, p2 = connector.coords
    idx1 = closest_vertex_index(ls[i], Point(p1))
    idx2 = closest_vertex_index(ls[j], Point(p2))
    return idx1, idx2


def compute_A(ls: List[LineString]) -> NDArray:
    """
    Compute A(i, j) for all pairs of trajectories. \n
    A(i, j) is the index of the closest point
    in trajectory i to all other points in trajectory j.
    """
    N = len(ls)
    A = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            idx1, idx2 = compute_h(ls, i, j)
            A[i, j], A[j, i] = idx1, idx2
    return A


def compute_B(ls: List[LineString], A: Optional[NDArray] = None) -> NDArray:
    """
    Compute B(i, j) from A(i, j). \n
    B is the corresponding progress value, i.e., normalised time
    found by diving by the number of points in the trajectory
    """
    N = len(ls)
    B = np.zeros((N, N), dtype=float)
    A = compute_A(ls) if A is None else A
    for i in range(N):
        for j in range(N):
            B[i, j] = A[i, j] / len(ls[i].coords)
    return B


def compute_P(ls: List[LineString], B: Optional[NDArray] = None) -> NDArray:
    """
    Compute P(i, j) from B(i, j). \n
    P(i) is the keypoint progress value for trajectory i,
    computed as the median of the progress values from B
    over all other trajectories.
    """
    N = len(ls)
    P = np.zeros((N,), dtype=float)
    B = compute_B(ls) if B is None else B
    for i in range(N):
        P[i] = np.median(B[i, :])
    return P


def _get_indices(
    ls: List[LineString],
    indices: Union[int, List[int]] = None,
) -> List[int]:
    """Get the indices of the trajectories to be plotted."""
    indices = [indices] if isinstance(indices, int) else indices
    indices: List[int] = indices or range(len(ls))
    return indices


def plot_trajectories(
    ls: List[LineString],
    indices: Union[int, List[int]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Plot the trajectories for given indices."""
    indices = _get_indices(ls, indices)
    if ax is None:
        _, ax = plt.subplots()
    for i in indices:
        ax.plot(*ls[i].xy, label=str(i), **kwargs)
    return ax


def plot_index_points(
    ls: List[LineString],
    A: Optional[NDArray] = None,
    indices: Union[int, List[int]] = None,
    only_between: bool = False,
) -> plt.Axes:
    """
    Plot the closest points using A(i, j)
    for a given indices of trajectories.
    """
    A = compute_A(ls) if A is None else A
    indices = _get_indices(ls, indices)
    other: List[int] = (
        indices if (only_between and len(indices) > 1) else range(len(ls))
    )
    ax = plot_trajectories(ls, indices)
    for i in indices:
        for j in other:
            if i != j:
                ax.plot(*ls[i].coords[int(A[i, j])], "ro")
    return ax


def plot_keypoints(
    ls: List[LineString],
    P: Optional[NDArray] = None,
    indices: Union[int, List[int]] = None,
) -> plt.Axes:
    """Plot the keypoints using P(i) for a given indices of trajectories."""
    P = compute_P(ls) if P is None else P
    indices = _get_indices(ls, indices)
    ax = plot_trajectories(ls, indices)
    for i in indices:
        ax.plot(*ls[i].coords[int(P[i] * len(ls[i].coords))], "ro")
    return ax
