from __future__ import annotations

from pathlib import Path
import inspect
import unittest

from tongue_scripts.pipelines.vocaset_tongue_color_experiment import (
    DARK_RED_TONGUE_COLOR,
    LIGHT_RED_TONGUE_COLOR,
    MID_RED_TONGUE_COLOR,
    ColorExperimentParams,
    EXPERIMENT_NAME,
    ColorExperimentPaths,
    color_experiment_paths,
    experiment_name_for,
    tongue_color_for,
)
from tongue_scripts.rendering.render_dual_tongue_comparison import (
    render_video_with_dynamic_tongue,
)


class VocasetTongueColorExperimentTests(unittest.TestCase):
    def test_dynamic_renderer_exposes_default_tongue_color_parameter(self) -> None:
        signature = inspect.signature(render_video_with_dynamic_tongue)

        self.assertIn("tongue_color", signature.parameters)
        self.assertEqual(
            signature.parameters["tongue_color"].default,
            (1.0, 0.6, 0.6, 1.0),
        )

    def test_color_experiment_paths_are_organized_under_tests_subfolder(self) -> None:
        paths = color_experiment_paths(
            output_root=Path("/large/color"),
            link_root=Path("/repo/tests/vocaset_outputs/color_search"),
            color_name="dark_red",
            params=ColorExperimentParams(),
            speaker="FaceTalk_170725_00137_TA",
            sentence="sentence01",
        )

        self.assertIsInstance(paths, ColorExperimentPaths)
        self.assertEqual(paths.out_dir, Path("/large/color/darker_red_std0p27_rot5/FaceTalk_170725_00137_TA/sentence01"))
        self.assertEqual(paths.link_dir, Path("/repo/tests/vocaset_outputs/color_search") / EXPERIMENT_NAME)
        self.assertTrue(paths.audio_video.name.endswith("_darkred_std0p27_rot5_active_tongue_with_audio.mp4"))
        self.assertTrue(paths.report_csv.name.endswith("_darkred_std0p27_rot5_vsr_comparison.csv"))

    def test_light_red_paths_are_separate_from_dark_red_paths(self) -> None:
        paths = color_experiment_paths(
            output_root=Path("/large/color"),
            link_root=Path("/repo/tests/vocaset_outputs/color_search"),
            color_name="light_red",
            params=ColorExperimentParams(thickness=1.4),
            speaker="FaceTalk_170725_00137_TA",
            sentence="sentence01",
        )

        self.assertEqual(paths.out_dir, Path("/large/color/light_red_std0p27_rot5_th1p40/FaceTalk_170725_00137_TA/sentence01"))
        self.assertEqual(paths.link_dir.name, "light_red_std0p27_rot5_th1p40")
        self.assertTrue(paths.audio_video.name.endswith("_lightred_std0p27_rot5_th1p40_active_tongue_with_audio.mp4"))
        self.assertTrue(paths.report_csv.name.endswith("_lightred_std0p27_rot5_th1p40_vsr_comparison.csv"))

    def test_dark_red_tongue_color_is_redder_and_darker_than_default(self) -> None:
        default = (1.0, 0.6, 0.6, 1.0)

        self.assertEqual(DARK_RED_TONGUE_COLOR, (0.72, 0.08, 0.08, 1.0))
        self.assertLess(DARK_RED_TONGUE_COLOR[0], default[0])
        self.assertLess(DARK_RED_TONGUE_COLOR[1], default[1])
        self.assertLess(DARK_RED_TONGUE_COLOR[2], default[2])

    def test_color_lookup_supports_light_mid_and_dark_red(self) -> None:
        self.assertEqual(LIGHT_RED_TONGUE_COLOR, (1.0, 0.6, 0.6, 1.0))
        self.assertEqual(MID_RED_TONGUE_COLOR, (0.86, 0.34, 0.34, 1.0))
        self.assertEqual(tongue_color_for("light_red"), LIGHT_RED_TONGUE_COLOR)
        self.assertEqual(tongue_color_for("mid_red"), MID_RED_TONGUE_COLOR)
        self.assertEqual(tongue_color_for("dark_red"), DARK_RED_TONGUE_COLOR)
        self.assertEqual(experiment_name_for("light_red"), "light_red_std0p27_rot5")
        self.assertEqual(
            experiment_name_for("light_red", ColorExperimentParams(thickness=1.4)),
            "light_red_std0p27_rot5_th1p40",
        )

    def test_mid_red_thickness_paths_are_separate(self) -> None:
        paths = color_experiment_paths(
            output_root=Path("/large/color"),
            link_root=Path("/repo/tests/vocaset_outputs/color_search"),
            color_name="mid_red",
            params=ColorExperimentParams(thickness=1.4),
            speaker="FaceTalk_170725_00137_TA",
            sentence="sentence01",
        )

        self.assertEqual(paths.link_dir.name, "mid_red_std0p27_rot5_th1p40")
        self.assertTrue(paths.audio_video.name.endswith("_midred_std0p27_rot5_th1p40_active_tongue_with_audio.mp4"))


if __name__ == "__main__":
    unittest.main()
