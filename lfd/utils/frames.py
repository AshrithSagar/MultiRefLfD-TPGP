"""
lfd/utils/frames.py \n
Frames and transformations
"""

from typing import List, NewType, Union, overload

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from shapely import LineString

Demonstration = NewType("Demonstration", LineString)
"""
A demonstration is a sequence of Cartesian points and a progress value.
The progress value is the normalised time for a trajectory; a value between 0 and 1.
"""

DemonstrationSet = NewType("DemonstrationSet", List[Demonstration])
"""A set of demonstrations"""


class Frame:
    """
    A frame has a rotation matrix and a translation vector with respect to a reference frame.
    """

    def __init__(
        self, index: int, rotation: Rotation = None, translation: NDArray = None
    ):
        self.index = index
        self.rotation = rotation if rotation is not None else Rotation.identity(2)
        self.translation = (
            np.array(translation, dtype=float)
            if translation is not None
            else np.zeros(2)
        )

    def __repr__(self):
        return f"Frame(index={self.index})"


GlobalFrame = Frame(0)


@overload
def append_progress_values(xi: LineString) -> Demonstration: ...
@overload
def append_progress_values(xi: List[LineString]) -> DemonstrationSet: ...
def append_progress_values(
    xi: Union[LineString, List[LineString]],
) -> Union[Demonstration, DemonstrationSet]:
    """
    Parameters:
        xi: A LineString object representing a trajectory (2D).
    Returns:
        A LineString object with progress values added to the coordinates (3D).
    """

    def append_phi(xi_: LineString) -> Demonstration:
        l = len(xi_.coords)
        x = LineString([(*pos, i / (l - 1)) for i, pos in enumerate(xi_.coords)])
        return x

    if isinstance(xi, LineString):
        return append_phi(xi)
    else:
        return [append_phi(xi_) for xi_ in xi]


def transform_demonstration_set(
    dset: DemonstrationSet, frame: Frame
) -> DemonstrationSet:
    """
    Transform a demonstration set from global frame to a new frame, without affecting the progress values.
    Parameters:
        dset: A list of demonstrations (LineString objects with progress values).
        frame: The frame to transform to.
    Returns:
        A list of transformed demonstrations.
    """
    assert dset[0].has_z, "Progress values missing in demonstration set"
    transformed_dset: DemonstrationSet = []
    R, t = frame.rotation.as_matrix()[:2, :2], frame.translation
    for d in dset:
        transformed_xi = LineString(np.array(d.coords)[:, :2] @ R.T + t)
        transformed_dset.append(append_progress_values(transformed_xi))
    return transformed_dset
