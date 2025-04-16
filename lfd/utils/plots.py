"""
lfd/utils/plots.py \n
Plotting utilities
"""

from typing import List, Optional, Union

import matplotlib.pyplot as plt
from numpy.typing import NDArray
from shapely import get_coordinates

from .alignment import compute_A, compute_P
from .frames import DemonstrationSet


def _get_indices(
    dset: DemonstrationSet,
    indices: Union[int, List[int]] = None,
) -> List[int]:
    """Get the indices of the trajectories to be plotted."""
    indices = [indices] if isinstance(indices, int) else indices
    indices: List[int] = indices or range(len(dset))
    return indices


def plot_trajectories(
    dset: DemonstrationSet,
    indices: Union[int, List[int]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Plot the trajectories for given indices."""
    indices = _get_indices(dset, indices)
    if ax is None:
        _, ax = plt.subplots()
    for i in indices:
        ax.plot(*dset[i].xy, label=str(i), **kwargs)
    return ax


def plot_index_points(
    dset: DemonstrationSet,
    A: Optional[NDArray] = None,
    indices: Union[int, List[int]] = None,
    only_between: bool = False,
) -> plt.Axes:
    """
    Plot the closest points using A(i, j)
    for a given indices of trajectories.
    """
    A = compute_A(dset) if A is None else A
    indices = _get_indices(dset, indices)
    other: List[int] = (
        indices if (only_between and len(indices) > 1) else range(len(dset))
    )
    ax = plot_trajectories(dset, indices)
    for i in indices:
        for j in other:
            if i != j:
                ax.plot(*get_coordinates(dset[i])[int(A[i, j])], "ro")
    return ax


def plot_keypoints(
    dset: DemonstrationSet,
    P: Optional[NDArray] = None,
    indices: Union[int, List[int]] = None,
) -> plt.Axes:
    """Plot the keypoints using P(i) for a given indices of trajectories."""
    P = compute_P(dset) if P is None else P
    indices = _get_indices(dset, indices)
    ax = plot_trajectories(dset, indices)
    for i in indices:
        ax.plot(*get_coordinates(dset[i])[int(P[i] * (len(dset[i].coords) - 1))], "ro")
    return ax
