"""
lfd/utils/plots.py \n
Plotting utilities
"""

from typing import List, Optional, Union

import matplotlib.pyplot as plt
from numpy.typing import NDArray
from shapely import LineString

from .alignment import compute_A, compute_P


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
