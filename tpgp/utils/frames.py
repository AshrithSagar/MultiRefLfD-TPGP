"""
utils/frames.py \n
Frames and transformations
"""

from typing import List

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from shapely import LineString


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


# A demonstration is a sequence of Cartesian points and a progress value.
# The progress value is the normalised time for a trajectory; a value between 0 and 1.


def transform_demonstration_set(ls: List[LineString], frame: Frame) -> List[LineString]:
    """
    Transform a demonstration set from global frame to a new frame, without affecting the progress values.
    """
    transformed_ls: List[LineString] = []
    R, t = frame.rotation.as_matrix()[:2, :2], frame.translation
    for demo in ls:
        points = np.array(demo.coords)
        transformed_points = LineString(points @ R.T + t)
        transformed_ls.append(transformed_points)
    return transformed_ls
