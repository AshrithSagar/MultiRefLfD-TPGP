"""
utils/frames.py \n
Frames and transformations
"""

import numpy as np
from numpy.typing import NDArray


class Frame:
    """
    A frame has a rotation matrix and a translation vector with respect to a reference frame.
    """

    def __init__(
        self, index: int, rotation: NDArray = None, translation: NDArray = None
    ):
        self.index = index
        self.rotation = rotation if rotation is not None else np.eye(2)
        self.translation = translation if translation is not None else np.zeros(2)

    def __repr__(self):
        return f"Frame(index={self.index})"


GlobalFrame = Frame(0)


class Demonstration:
    """
    A demonstration is a sequence of Cartesian points and a progress value.
    The progress value is the normalised time for a trajectory; a value between 0 and 1.
    """

    def __init__(self, points: NDArray, progress: float):
        self.points = points
        self.progress = progress

    def __repr__(self):
        return f"Demonstration(points={self.points}, progress={self.progress})"


def transform_demonstration_set(D: NDArray, frame: Frame) -> NDArray:
    """
    Transform a demonstration set from global frame to a new frame, without affecting the progress values.
    """
    R, t = frame.rotation, frame.translation
    H = np.eye(R.shape[0] + 1, dtype=R.dtype)
    H[:-1, :-1] = R
    T = np.zeros(t.shape[0] + 1, dtype=t.dtype)
    T[:-1] = t
    D_m = T + H @ D
    return D_m
