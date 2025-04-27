"""
lfd/utils/demonstrations.py \n
Demonstrations
"""

from typing import List, NewType, Union, overload

from shapely import LineString, get_num_points

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
        n_length = get_num_points(xi_)
        x = LineString([(*pos, i / (n_length - 1)) for i, pos in enumerate(xi_.coords)])
        return x

    if isinstance(xi, LineString):
        return append_phi(xi)
    elif isinstance(xi, list) and all(isinstance(x, LineString) for x in xi):
        return [append_phi(xi_) for xi_ in xi]
    else:
        raise TypeError(
            f"xi must be a LineString or a list of LineStrings, not {type(xi).__name__}"
        )
