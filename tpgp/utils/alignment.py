"""
utils/alignment.py \n
Alignment of demonstations
"""

import math
from typing import Tuple, Union

import numpy as np
from shapely import LineString, Point, shortest_line


def compute_h(ls, i, j) -> Tuple[int, int]:
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


def compute_A(ls):
    """Compute A(i, j) for all pairs of trajectories"""
    N = len(ls)
    A = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            idx1, idx2 = compute_h(ls, i, j)
            A[i, j], A[j, i] = idx1, idx2
    return A
