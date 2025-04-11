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
    A(i, j) is the index of the closest point in trajectory i to all other points in trajectory j.
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
    B is the corresponding progress value, i.e., normalised time found by diving by the number of points in the trajectory
    """
    N = len(ls)
    B = np.zeros((N, N), dtype=float)
    A = compute_A(ls) if A is None else A
    for i in range(N):
        for j in range(N):
            B[i, j] = A[i, j] / len(ls[i].coords)
    return B


def plot_index_points(
    ls: List[LineString], A: Optional[NDArray] = None, indices: List[int] = None
) -> None:
    """Plot the closest points using A(i, j) for a given indices of trajectories."""
    A = compute_A(ls) if A is None else A
    indices = indices or range(len(ls))
    _, ax = plt.subplots()
    for i in indices:
        ax.plot(*ls[i].xy, label=str(i))
        for j in indices:
            if i != j:
                ax.plot(*ls[i].coords[int(A[i, j])], "ro")
    plt.tight_layout()
    plt.show()
