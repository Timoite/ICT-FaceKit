from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tongue_scripts.pipelines.evaluate_vocaset_active_passive_batch import (
    BatchEvaluationPaths,
    build_eval_jobs,
    sentence_ground_truth,
    should_write_incremental_summary,
)
from tongue_scripts.pipelines.render_vocaset_active_best_worker import (
    BEST_PARAMS,
    DEFAULT_ACTIVE_BEST_LINK_DIR,
    active_output_paths,
    collect_render_jobs,
    split_jobs,
)


class VocasetActiveBestBatchTests(unittest.TestCase):
    def test_active_best_output_paths_include_fixed_params_and_tests_subfolder(self) -> None:
        paths = active_output_paths(
            output_root=Path("/large/out"),
            link_dir=Path("/repo/tests/vocaset_outputs/active_best"),
            speaker="FaceTalk_170725_00137_TA",
            sentence="sentence01",
            params=BEST_PARAMS,
        )

        self.assertEqual(paths.clip_id, "FaceTalk_170725_00137_TA_sentence01")
        self.assertEqual(paths.out_dir, Path("/large/out/FaceTalk_170725_00137_TA/sentence01"))
        self.assertEqual(paths.motion_path, paths.out_dir / "tongue_motion" / "tongue_motion.npy")
        self.assertTrue(paths.audio_video.name.endswith("_std0p27_z0p00_rot5p00_active_tongue_with_audio.mp4"))
        self.assertEqual(paths.link.parent, Path("/repo/tests/vocaset_outputs/active_best"))

    def test_default_active_best_link_dir_is_under_vocaset_outputs(self) -> None:
        self.assertEqual(DEFAULT_ACTIVE_BEST_LINK_DIR.name, "active_best_std0p27_z0p00_rot5")
        self.assertEqual(DEFAULT_ACTIVE_BEST_LINK_DIR.parent.name, "vocaset_outputs")

    def test_collect_render_jobs_requires_json_wav_and_transcript(self) -> None:
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

            jobs = collect_render_jobs(json_root=json_root, wav_root=wav_root, transcript_root=text_root)

        self.assertEqual([job.clip_id for job in jobs], ["SpeakerA_sentence01"])

    def test_sentence_ground_truth_uses_one_based_sentence_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "SpeakerA.txt"
            transcript.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

            self.assertEqual(sentence_ground_truth(transcript, "sentence02"), "beta")

    def test_split_jobs_preserves_all_jobs(self) -> None:
        jobs = ["a", "b", "c", "d", "e"]

        self.assertEqual(split_jobs(jobs, workers=2), [["a", "c", "e"], ["b", "d"]])

    def test_build_eval_jobs_pairs_active_passive_and_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = BatchEvaluationPaths(
                active_root=root / "active",
                passive_root=root / "passive",
                transcript_root=root / "sentencestext",
            )
            active_dir = paths.active_root / "SpeakerA" / "sentence01"
            passive_dir = paths.passive_root / "SpeakerA" / "sentence01"
            active_dir.mkdir(parents=True)
            passive_dir.mkdir(parents=True)
            active_video = active_dir / "SpeakerA_sentence01_std0p27_z0p00_rot5p00_active_tongue_with_audio.mp4"
            passive_video = passive_dir / "SpeakerA_sentence01_passive_tongue_with_audio.mp4"
            active_video.write_bytes(b"mp4")
            passive_video.write_bytes(b"mp4")
            paths.transcript_root.mkdir()
            (paths.transcript_root / "SpeakerA.txt").write_text("first sentence\n", encoding="utf-8")

            jobs = build_eval_jobs(paths)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].clip_id, "SpeakerA_sentence01")
        self.assertEqual(jobs[0].active_video, active_video)
        self.assertEqual(jobs[0].passive_video, passive_video)
        self.assertEqual(jobs[0].ground_truth, "first sentence")

    def test_parallel_eval_workers_can_disable_incremental_summary_writes(self) -> None:
        class Args:
            metrics_only = True
            summarize_only = False

        self.assertFalse(should_write_incremental_summary(Args()))


if __name__ == "__main__":
    unittest.main()
