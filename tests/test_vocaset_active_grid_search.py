from __future__ import annotations

import unittest

from tongue_scripts.pipelines.summarize_vocaset_grid_search import varied_parameter_names
from tongue_scripts.pipelines.visualize_vocaset_grid_descent import (
    best_so_far,
    infer_search_stage,
)
from tongue_scripts.pipelines.grid_search_vocaset_active_tongue import (
    DEFAULT_VOCASET_LINK_DIR,
    TongueGridParams,
    build_grid,
    build_default_grid,
    coerce_metric_row,
    param_slug,
)
from tongue_scripts.pipelines.grid_search.run_vocaset_std_scalar_th14_rot0 import (
    build_std_scalar_values,
    build_runner_argv,
)
from tongue_scripts.pipelines.grid_search.run_vocaset_std_scalar_fine_th14_rot0 import (
    build_fine_std_scalar_values,
    build_fine_runner_argv,
    split_values,
)


class VocasetActiveGridSearchTests(unittest.TestCase):
    def test_default_grid_is_small_and_centered_on_baseline(self) -> None:
        grid = build_default_grid()

        self.assertEqual(len(grid), 9)
        self.assertIn(TongueGridParams(std_scalar=0.20, shift_z=0.0), grid)
        self.assertEqual(sorted({p.std_scalar for p in grid}), [0.15, 0.20, 0.25])
        self.assertEqual(sorted({p.shift_z for p in grid}), [-0.5, 0.0, 0.5])

    def test_param_slug_is_filesystem_stable(self) -> None:
        params = TongueGridParams(std_scalar=0.15, shift_z=-0.5)

        self.assertEqual(param_slug(params), "std0p15_zm0p50")

    def test_custom_grid_uses_supplied_values(self) -> None:
        grid = build_grid(
            std_scalars=(0.23, 0.27),
            shift_z_values=(-0.1, 0.1),
            rotation_deg_values=(0.0,),
            thickness_values=(1.0,),
            shift_y_values=(0.2,),
        )

        self.assertEqual(
            grid,
            [
                TongueGridParams(std_scalar=0.23, shift_z=-0.1, rotation_deg=0.0, thickness=1.0, shift_y=0.2),
                TongueGridParams(std_scalar=0.23, shift_z=0.1, rotation_deg=0.0, thickness=1.0, shift_y=0.2),
                TongueGridParams(std_scalar=0.27, shift_z=-0.1, rotation_deg=0.0, thickness=1.0, shift_y=0.2),
                TongueGridParams(std_scalar=0.27, shift_z=0.1, rotation_deg=0.0, thickness=1.0, shift_y=0.2),
            ],
        )

    def test_param_slug_includes_nondefault_geometry(self) -> None:
        params = TongueGridParams(
            std_scalar=0.27,
            shift_z=0.0,
            rotation_deg=0.0,
            thickness=1.0,
            shift_y=0.2,
        )

        self.assertEqual(param_slug(params), "std0p27_z0p00_rot0p00_th1p00_y0p20")

    def test_summary_detects_varied_parameters(self) -> None:
        rows = [
            {"std_scalar": "0.27", "shift_z": "0.0", "rotation_deg": "5.0", "thickness": "1.2", "shift_y": "0.0"},
            {"std_scalar": "0.27", "shift_z": "0.0", "rotation_deg": "10.0", "thickness": "1.2", "shift_y": "0.0"},
            {"std_scalar": "", "shift_z": "", "rotation_deg": "", "thickness": "", "shift_y": ""},
        ]

        self.assertEqual(varied_parameter_names(rows), ["rotation_deg"])

    def test_grid_search_links_default_to_tests_subfolder(self) -> None:
        self.assertEqual(DEFAULT_VOCASET_LINK_DIR.name, "videos")
        self.assertEqual(DEFAULT_VOCASET_LINK_DIR.parent.name, "grid_search")
        self.assertEqual(DEFAULT_VOCASET_LINK_DIR.parent.parent.name, "vocaset_outputs")

    def test_descent_visualizer_tracks_best_so_far(self) -> None:
        self.assertEqual(best_so_far([0.9, 0.7, 0.8, 0.6]), [0.9, 0.7, 0.7, 0.6])

    def test_descent_visualizer_infers_search_stage_from_video_path(self) -> None:
        self.assertEqual(infer_search_stage("/tmp/refined_std_z/std0p27/video.mp4"), "std/shift refine")
        self.assertEqual(infer_search_stage("/tmp/thickness_refine/std0p27/video.mp4"), "thickness refine")
        self.assertEqual(infer_search_stage("/tmp/std0p25_z0p00/video.mp4"), "coarse grid")

    def test_std_scalar_th14_rot0_grid_values_are_0p05_granularity(self) -> None:
        self.assertEqual(
            build_std_scalar_values(),
            [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        )

    def test_std_scalar_th14_rot0_runner_uses_fixed_geometry(self) -> None:
        argv = build_runner_argv()

        self.assertIn("--thickness-values", argv)
        self.assertEqual(argv[argv.index("--thickness-values") + 1], "1.4")
        self.assertIn("--rotation-deg-values", argv)
        self.assertEqual(argv[argv.index("--rotation-deg-values") + 1], "0.0")
        self.assertIn("--shift-z-values", argv)
        self.assertEqual(argv[argv.index("--shift-z-values") + 1], "0.0")

    def test_metric_rows_loaded_from_backfilled_json_are_numeric(self) -> None:
        row = coerce_metric_row({"ver": "0.6176", "wer_norm": "1.0", "wer_raw": "1.0", "composite_index": "0.8088"})

        self.assertEqual(row["ver"], 0.6176)
        self.assertEqual(row["wer_norm"], 1.0)
        self.assertEqual(row["wer_raw"], 1.0)
        self.assertEqual(row["composite_index"], 0.8088)

    def test_fine_std_scalar_values_are_0p01_granularity(self) -> None:
        values = build_fine_std_scalar_values()

        self.assertEqual(values[0], 0.20)
        self.assertEqual(values[-1], 0.40)
        self.assertEqual(len(values), 21)
        self.assertIn(0.27, values)

    def test_fine_runner_can_render_only(self) -> None:
        argv = build_fine_runner_argv(std_scalars=[0.2, 0.21], render_only=True)

        self.assertIn("--render-only", argv)
        self.assertIn("--std-scalars", argv)
        idx = argv.index("--std-scalars")
        self.assertEqual(argv[idx + 1: idx + 3], ["0.2", "0.21"])

    def test_split_values_preserves_order_and_covers_all_values(self) -> None:
        chunks = split_values([0.2, 0.21, 0.22, 0.23, 0.24], workers=2)

        self.assertEqual(chunks, [[0.2, 0.22, 0.24], [0.21, 0.23]])


if __name__ == "__main__":
    unittest.main()
