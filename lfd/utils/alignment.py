"""
lfd/utils/alignment.py \n
Alignment of demonstations
"""

import math
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from shapely import LineString, Point, get_point, shortest_line

from .frames import Demonstration, DemonstrationSet, Frame


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
