"""
lfd/utils/frames.py \n
Frames and transformations
"""

from functools import partial
from typing import List, NewType, Tuple, Union, overload

from shapely import LineString, Point, get_num_points
from shapely.affinity import rotate, translate

Demonstration = NewType("Demonstration", LineString)
"""A demonstration is a sequence of Cartesian points and a progress value."""

DemonstrationSet = NewType("DemonstrationSet", List[Demonstration])
"""A set of demonstrations."""


@overload
def append_progress_values(xi: LineString) -> Demonstration: ...
@overload
def append_progress_values(xi: List[LineString]) -> DemonstrationSet: ...
def append_progress_values(
    xi: Union[LineString, List[LineString]],
) -> Union[Demonstration, DemonstrationSet]:
    """
    Append progress values to a LineString or a list of LineStrings.
    The progress value is the normalised time for a trajectory (lies in [0, 1]),
    found by dividing the index of the point by the number of points in the trajectory.

    Here, the progress value is added as a third coordinate to the LineString.
    Shapely does not consider z-coordinates for geometric computations,
    so the LineString is still treated as a 2D object effectively.

    :param xi: A LineString or a list of LineStrings representing trajectories (2D).
    :return: A LineString or a list of LineStrings with progress values added to the coordinates (3D).
    """

    def append_phi(xi_: LineString) -> Demonstration:
        l = get_num_points(xi_)
        x = LineString([(*pos, i / (l - 1)) for i, pos in enumerate(xi_.coords)])
        return x

    if isinstance(xi, LineString):
        return append_phi(xi)
    elif isinstance(xi, list) and all(isinstance(x, LineString) for x in xi):
        return [append_phi(xi_) for xi_ in xi]
    else:
        raise TypeError(
            "xi must be a LineString or a list of LineStrings, "
            f"not {type(xi).__name__}"
        )


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
        assert (
            origin in ["center", "centroid"] if isinstance(origin, str) else True
        ), "Origin must be 'center', 'centroid', or a point"
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
