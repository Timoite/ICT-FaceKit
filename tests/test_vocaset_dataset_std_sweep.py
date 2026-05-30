from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tongue_scripts.pipelines.grid_search.vocaset_dataset_std_sweep_th14_rot0 import (
    DEFAULT_LINK_ROOT,
    StdSweepMetric,
    StdSweepParams,
    active_output_paths,
    best_vote_summary,
    build_std_values,
    collect_std_sweep_jobs,
    comparison_rows_by_std,
    std_slug,
)


class VocasetDatasetStdSweepTests(unittest.TestCase):
    def test_std_values_are_0p025_granularity_from_0p10_to_0p30(self) -> None:
        self.assertEqual(
            build_std_values(),
            [0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.275, 0.300],
        )

    def test_std_values_can_start_at_zero_for_followup_sweep(self) -> None:
        self.assertEqual(
            build_std_values(start=0.0, stop=0.3, step=0.025),
            [0.000, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.275, 0.300],
        )

    def test_std_slug_preserves_three_decimal_precision(self) -> None:
        self.assertEqual(std_slug(0.1), "std0p100")
        self.assertEqual(std_slug(0.125), "std0p125")
        self.assertEqual(std_slug(0.3), "std0p300")

    def test_default_link_root_is_organized_under_tests_vocaset_outputs(self) -> None:
        self.assertEqual(DEFAULT_LINK_ROOT.name, "std_0p100_0p300_step0p025_th1p400_rot0")
        self.assertEqual(DEFAULT_LINK_ROOT.parent.name, "grid_search")
        self.assertEqual(DEFAULT_LINK_ROOT.parent.parent.name, "vocaset_outputs")

    def test_active_output_paths_include_fixed_default_geometry(self) -> None:
        paths = active_output_paths(
            output_root=Path("/large/sweep"),
            link_root=Path("/repo/tests/vocaset_outputs/grid_search/sweep"),
            speaker="SpeakerA",
            sentence="sentence01",
            params=StdSweepParams(std_scalar=0.125),
        )

        self.assertEqual(paths.out_dir, Path("/large/sweep/std0p125/SpeakerA/sentence01"))
        self.assertEqual(paths.motion_path, Path("/large/sweep/_motion/SpeakerA/sentence01/tongue_motion.npy"))
        self.assertTrue(paths.audio_video.name.endswith("_std0p125_th1p400_rot0p000_active_tongue_with_audio.mp4"))
        self.assertEqual(paths.link.parent, Path("/repo/tests/vocaset_outputs/grid_search/sweep/videos/std0p125"))

    def test_active_output_paths_include_configured_thickness(self) -> None:
        paths = active_output_paths(
            output_root=Path("/large/sweep_th1"),
            link_root=Path("/repo/tests/vocaset_outputs/grid_search/sweep_th1"),
            speaker="SpeakerA",
            sentence="sentence01",
            params=StdSweepParams(std_scalar=0.025, thickness=1.0),
        )

        self.assertTrue(paths.audio_video.name.endswith("_std0p025_th1p000_rot0p000_active_tongue_with_audio.mp4"))
        self.assertEqual(paths.link.parent, Path("/repo/tests/vocaset_outputs/grid_search/sweep_th1/videos/std0p025"))


    def test_collect_std_sweep_jobs_crosses_renderable_clips_with_each_std(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_root = root / "blendshape_json"
            wav_root = root / "wav"
            text_root = root / "sentencestext"
            (json_root / "SpeakerA").mkdir(parents=True)
            wav_root.mkdir()
            text_root.mkdir()
            (json_root / "SpeakerA" / "sentence01.json").write_text("{}", encoding="utf-8")
            (json_root / "SpeakerA" / "sentence02.json").write_text("{}", encoding="utf-8")
            (wav_root / "SpeakerA_sentence01.wav").write_bytes(b"wav")
            (text_root / "SpeakerA.txt").write_text("first sentence\n", encoding="utf-8")

            jobs = collect_std_sweep_jobs(
                json_root=json_root,
                wav_root=wav_root,
                transcript_root=text_root,
                std_values=[0.100, 0.125],
                thickness=1.0,
            )

        self.assertEqual([job.clip_id for job in jobs], ["SpeakerA_sentence01", "SpeakerA_sentence01"])
        self.assertEqual([job.params.std_scalar for job in jobs], [0.100, 0.125])
        self.assertEqual([job.params.thickness for job in jobs], [1.0, 1.0])

    def test_best_vote_summary_counts_lowest_ver_per_clip(self) -> None:
        rows = [
            StdSweepMetric("clip1", "S", "sentence01", 0.100, 0.6, 1.0, 0.8, "", "", ""),
            StdSweepMetric("clip1", "S", "sentence01", 0.125, 0.5, 1.0, 0.75, "", "", ""),
            StdSweepMetric("clip2", "S", "sentence02", 0.100, 0.4, 1.0, 0.7, "", "", ""),
            StdSweepMetric("clip2", "S", "sentence02", 0.125, 0.4, 0.8, 0.6, "", "", ""),
        ]

        summary = best_vote_summary(rows)

        self.assertEqual(summary[0]["std_scalar"], 0.125)
        self.assertEqual(summary[0]["best_vote_count"], 2)
        self.assertEqual(summary[0]["clip_count"], 2)
        self.assertEqual(summary[1]["std_scalar"], 0.100)
        self.assertEqual(summary[1]["best_vote_count"], 0)

    def test_comparison_rows_by_std_reports_mean_and_median_ver(self) -> None:
        rows = [
            StdSweepMetric("clip1", "S", "sentence01", 0.100, 0.6, 1.0, 0.8, "", "", ""),
            StdSweepMetric("clip2", "S", "sentence02", 0.100, 0.4, 1.0, 0.7, "", "", ""),
            StdSweepMetric("clip1", "S", "sentence01", 0.125, 0.5, 1.0, 0.75, "", "", ""),
        ]

        summary = comparison_rows_by_std(rows)

        self.assertEqual(summary[0]["std_scalar"], 0.100)
        self.assertEqual(summary[0]["clip_count"], 2)
        self.assertAlmostEqual(summary[0]["mean_ver"], 0.5)
        self.assertAlmostEqual(summary[0]["median_ver"], 0.5)


if __name__ == "__main__":
    unittest.main()
