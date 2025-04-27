"""
lfd/utils/frames.py \n
Frames and transformations
"""

from functools import partial
from typing import Tuple, Union

from shapely import Point
from shapely.affinity import rotate, translate

from .demonstrations import DemonstrationSet


class Frame:
    """Frame class representing a coordinate frame in 2D space."""

    def __init__(
        self,
        index: int,
        rotation: float = 0,
        translation: Tuple[float, float] = (0, 0),
        origin: Union[str, Point, Tuple[float, float]] = (0, 0),
    ):
        """
        A frame has a rotation matrix and a translation vector
        with respect to a reference (global) frame.
        Rotations are with respect to the origin (0, 0).
        Parameters:
            index: Frame identifier
            rotation: Rotation angle (in degrees)
            translation: Translation vector (2D)
        """
        self.index = index
        self.rotation = rotation
        self.translation = translation
        assert origin in ["center", "centroid"] if isinstance(origin, str) else True, (
            "Origin must be 'center', 'centroid', or a point"
        )
        self.origin = origin

        self._R = partial(rotate, angle=rotation, origin=origin)
        self._t = partial(translate, xoff=translation[0], yoff=translation[1], zoff=0)

    def __repr__(self):
        return f"Frame(index={self.index}, rotation={self.rotation}, translation={self.translation})"

    def transform(self, dset: DemonstrationSet) -> DemonstrationSet:
        """
        Transform a point from the global frame to this frame,
        without affecting the progress values.
        Parameters:
            dset: A list of demonstrations (LineString objects with progress values).
            frame: The frame to transform to.
        Returns:
            A list of transformed demonstrations.
        """
        assert dset[0].has_z, "Progress values missing in demonstration set"
        return [self._t(self._R(d)) for d in dset]


GlobalFrame = Frame(0)
"""Global frame (reference) with index 0."""
GlobalFrame = Frame(0)
"""Global frame (reference) with index 0."""
