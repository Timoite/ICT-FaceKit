from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tongue_scripts.pipelines import run_render_dual_for_dataset as dataset_pipeline
from tongue_scripts.pipelines import run_render_dual_for_speaker as speaker_pipeline


class RenderPipelineGpuTests(unittest.TestCase):
    def test_configure_render_backend_enables_egl_gpu_backend(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            dataset_pipeline.configure_render_backend(use_gpu=True)

            self.assertEqual(os.environ["PYOPENGL_PLATFORM"], "egl")
            self.assertEqual(os.environ["__GLX_VENDOR_LIBRARY_NAME"], "nvidia")

    def test_configure_render_backend_leaves_default_backend_when_disabled(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            dataset_pipeline.configure_render_backend(use_gpu=False)

            self.assertNotIn("PYOPENGL_PLATFORM", os.environ)
            self.assertNotIn("__GLX_VENDOR_LIBRARY_NAME", os.environ)

    def test_speaker_pipeline_forwards_use_gpu_to_dataset_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            beat_root = root / "beat"
            speaker_root = beat_root / "9"
            speaker_root.mkdir(parents=True)
            (speaker_root / "9_test_0_1_1.json").write_text("{}", encoding="utf-8")

            motion_dir = root / "motion"
            motion_dir.mkdir()
            (motion_dir / "9_test_0_1_1.npy").write_bytes(b"placeholder")

            output_dir = root / "videos"
            commands: list[list[str]] = []

            def capture_run(cmd: list[str], cwd: Path | None = None) -> None:
                commands.append(cmd)

            argv = [
                "run_render_dual_for_speaker.py",
                "--speaker-id",
                "9",
                "--beat-root",
                str(beat_root),
                "--motion-dir",
                str(motion_dir),
                "--output-dir",
                str(output_dir),
                "--use-gpu",
                "--limit",
                "1",
            ]

            with mock.patch.object(sys, "argv", argv), mock.patch.object(speaker_pipeline, "run", capture_run):
                speaker_pipeline.main()

        self.assertEqual(len(commands), 1)
        self.assertIn("--use-gpu", commands[0])


if __name__ == "__main__":
    unittest.main()
