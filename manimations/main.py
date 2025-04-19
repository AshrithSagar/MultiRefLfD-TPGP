"""
manimations/main.py
Generate ManimCE animations
"""

import colorsys

import manim as mn
import numpy as np
from shapely import LineString

from lfd.utils import append_progress_values
from lfd.utils.lasa import load_data

data, _, _ = load_data("s")
dset = append_progress_values([LineString(traj) for traj in data])


def normalize_points(points):
    arr = np.array(points)
    mean = arr.mean(axis=0)
    scale = 6 / max(arr.max(axis=0) - arr.min(axis=0))  # Fit to screen
    return [(p - mean) * scale for p in points]


def hsv_to_manim_color(h, s=1.0, v=1.0):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return mn.utils.color.rgb_to_color((r, g, b))


class DemonstrationSetup(mn.Scene):
    def construct(self):
        # Loop and draw each trajectory
        for i, traj in enumerate(dset):
            coords_2d = [(x, y) for x, y, _ in traj.coords]
            points = [np.array([x, y, 0]) for x, y in coords_2d]
            norm_points = normalize_points(points)

            color = hsv_to_manim_color(i / len(dset))
            path = mn.VMobject()
            path.set_points_as_corners(norm_points)
            path.set_stroke(color=color, width=3)

            self.play(mn.Create(path), run_time=0.5)


class DemonstrationSetup3D(mn.ThreeDScene):
    def construct(self):
        # Set an isometric camera view
        self.set_camera_orientation(phi=60 * mn.DEGREES, theta=-45 * mn.DEGREES)

        # Loop and show each trajectory
        for i, traj in enumerate(dset):
            coords_3d = [(x, y, z) for x, y, z in traj.coords]
            points = [np.array([x, y, z]) for x, y, z in coords_3d]
            norm_points = normalize_points(points)

            color = hsv_to_manim_color(i / len(dset))
            path = mn.VMobject()
            path.set_points_as_corners(norm_points)
            path.set_stroke(color=color, width=3)

            self.add(path)
            self.wait(0.2)
