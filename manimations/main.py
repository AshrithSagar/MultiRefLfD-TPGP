"""
manimations/main.py
Generate ManimCE animations
"""

import colorsys
from typing import Tuple

import manim as mn
import numpy as np
from numpy.typing import NDArray

from lfd import Demonstration, load_dset

dset = load_dset("s")


# Utility functions
def demo_to_points(d: Demonstration, include_z: bool = False) -> NDArray:
    if include_z:
        coords = [(x, y, z) for x, y, z in d.coords]
        arr = np.array([np.array([x, y, z]) for x, y, z in coords])
    else:
        coords = [(x, y) for x, y, _ in d.coords]
        arr = np.array([np.array([x, y, 0]) for x, y in coords])
    return arr


def find_scale_and_mean(
    d: Demonstration, scale_target: float = 6
) -> Tuple[float, NDArray]:
    arr = demo_to_points(d)
    mean: NDArray = arr.mean(axis=0)
    # Fit to screen
    scale: float = scale_target / max(arr.max(axis=0) - arr.min(axis=0))
    return scale, mean


def normalize_points(points, scale: float, mean: NDArray):
    return [(p - mean) * scale for p in points]


def hsv_to_manim_color(h, s=1.0, v=1.0):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return mn.utils.color.rgb_to_color((r, g, b))


class DemonstrationScene(mn.Scene):
    def setup(self):
        self.camera.background_color = mn.BLACK

    def construct(self):
        scale, mean = find_scale_and_mean(dset[-1])

        # Loop and draw each trajectory
        num_traj = len(dset)
        for i, traj in enumerate(dset[:num_traj]):
            arr = demo_to_points(traj, include_z=False)
            norm_points = normalize_points(arr, scale, mean)

            color = hsv_to_manim_color(i / len(dset))
            path = mn.VMobject()
            path.set_points_as_corners(norm_points)
            path.set_stroke(color=color, width=3)

            self.play(mn.Create(path), run_time=2)
            self.wait(1e-2)


class DemonstrationScene3D(mn.ThreeDScene):
    def setup(self):
        self.camera.background_color = mn.BLACK

    def construct(self):
        # Set an isometric camera view
        self.set_camera_orientation(phi=60 * mn.DEGREES, theta=-45 * mn.DEGREES)

        scale, mean = find_scale_and_mean(dset[-1])

        # Loop and draw each trajectory
        num_traj = len(dset)
        for i, traj in enumerate(dset[:num_traj]):
            arr = demo_to_points(traj, include_z=True)
            norm_points = normalize_points(arr, scale, mean)

            color = hsv_to_manim_color(i / len(dset))
            path = mn.VMobject()
            path.set_points_as_corners(norm_points)
            path.set_stroke(color=color, width=3)

            self.begin_ambient_camera_rotation(rate=0.015)
            self.play(mn.Create(path), run_time=2)
            self.stop_ambient_camera_rotation()
