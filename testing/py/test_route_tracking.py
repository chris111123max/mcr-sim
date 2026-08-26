"""Regression tests for continuous selected-route tracking."""

from __future__ import annotations

import unittest

import numpy as np

from mcr_sim.route_tracking import normalized_route_progress, project_to_route


def cumulative(points: np.ndarray) -> np.ndarray:
    return np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )


class RouteTrackingTest(unittest.TestCase):
    def setUp(self) -> None:
        # Two spatially adjacent arms with opposite route directions emulate a
        # tight 180-degree bend. At x=4 mm the return arm is only 1 mm away,
        # but about 13 mm farther along the selected route.
        self.u_turn = np.asarray(
            [
                [0.000, 0.000, 0.0],
                [0.010, 0.000, 0.0],
                [0.010, 0.001, 0.0],
                [0.000, 0.001, 0.0],
            ],
            dtype=np.float64,
        )
        self.arc = cumulative(self.u_turn)

    def test_initial_localization_may_use_complete_route(self) -> None:
        projection = project_to_route(
            self.u_turn,
            self.arc,
            np.asarray([0.004, 0.001, 0.0]),
        )
        self.assertGreater(projection.progress, 0.010)
        self.assertFalse(projection.jump_rejected)

    def test_nearby_return_arm_cannot_create_progress_jump(self) -> None:
        previous = 0.004
        for _ in range(20):
            projection = project_to_route(
                self.u_turn,
                self.arc,
                np.asarray([0.004, 0.001, 0.0]),
                previous_progress=previous,
                max_progress_step=0.002,
            )
            self.assertAlmostEqual(projection.progress, 0.004, places=6)
            self.assertTrue(projection.jump_rejected)
            previous = projection.progress

    def test_exact_self_crossing_prefers_topological_continuity(self) -> None:
        crossing = np.asarray(
            [
                [-0.010, 0.000, 0.0],
                [0.010, 0.000, 0.0],
                [0.010, 0.010, 0.0],
                [0.000, 0.010, 0.0],
                [0.000, -0.010, 0.0],
            ],
            dtype=np.float64,
        )
        projection = project_to_route(
            crossing,
            cumulative(crossing),
            np.zeros(3, dtype=np.float64),
            previous_progress=0.010,
            max_progress_step=0.002,
        )
        self.assertAlmostEqual(projection.progress, 0.010, places=6)
        self.assertEqual(projection.segment_index, 0)
        self.assertFalse(projection.jump_rejected)

    def test_normal_forward_progress_is_not_clipped(self) -> None:
        projection = project_to_route(
            self.u_turn,
            self.arc,
            np.asarray([0.005, 0.000, 0.0]),
            previous_progress=0.004,
            max_progress_step=0.002,
        )
        self.assertAlmostEqual(projection.progress, 0.005, places=6)
        self.assertFalse(projection.jump_rejected)

    def test_normal_retraction_remains_available(self) -> None:
        projection = project_to_route(
            self.u_turn,
            self.arc,
            np.asarray([0.003, 0.000, 0.0]),
            previous_progress=0.004,
            max_progress_step=0.002,
        )
        self.assertAlmostEqual(projection.progress, 0.003, places=6)
        self.assertFalse(projection.jump_rejected)

    def test_continuous_progress_is_bounded(self) -> None:
        self.assertEqual(normalized_route_progress(-1.0, 0.0, 1.0), 0.0)
        self.assertEqual(normalized_route_progress(0.25, 0.0, 1.0), 0.25)
        self.assertEqual(normalized_route_progress(2.0, 0.0, 1.0), 1.0)

    def test_potential_difference_telescopes_after_retraction(self) -> None:
        progress = [0.0, 0.3, 0.2, 0.3, 0.7]
        potential = [normalized_route_progress(value, 0.0, 1.0) for value in progress]
        deltas = [b - a for a, b in zip(potential, potential[1:])]
        self.assertAlmostEqual(sum(deltas), 0.7)
        self.assertAlmostEqual(deltas[1] + deltas[2], 0.0)


if __name__ == "__main__":
    unittest.main()
