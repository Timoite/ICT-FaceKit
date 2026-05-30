from __future__ import annotations

import unittest

import numpy as np

from tongue_scripts.tongue_animation.generate_tongue_animation import sagittal_profile_points


class TongueMatplotlibContextTests(unittest.TestCase):
    def test_sagittal_profile_points_selects_x_near_cut_plane_and_returns_zy(self) -> None:
        vertices = np.asarray(
            [
                [0.0, 1.0, 2.0],
                [0.2, 3.0, 4.0],
                [1.2, 5.0, 6.0],
            ],
            dtype=np.float32,
        )

        points = sagittal_profile_points(vertices, x_abs_threshold=0.25)

        np.testing.assert_allclose(points, [[2.0, 1.0], [4.0, 3.0]])


if __name__ == "__main__":
    unittest.main()
