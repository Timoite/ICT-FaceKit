from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tongue_scripts.real_video.arkit_to_ict import (
    SAID_ARKIT_NAMES,
    arkit_name_to_ict_names,
    convert_arkit_to_ict,
)
from tongue_scripts.real_video.smirk_flame_to_arkit import (
    ARKitBasis,
    fit_coefficients,
    fit_smirk_vertices_file,
)
from tongue_scripts.pipelines.run_fadg0_real_video_pipeline import (
    apply_render_shift,
    discover_videos,
    resample_sequence_to_frames,
    shift_tongue_motion_file,
)
from tongue_scripts.tongue_animation.generate_tongue_animation import load_blendshape_json_sequence


ICT_NAMES = [
    "browDown_L",
    "browDown_R",
    "browInnerUp_L",
    "browInnerUp_R",
    "browOuterUp_L",
    "browOuterUp_R",
    "cheekPuff_L",
    "cheekPuff_R",
    "cheekSquint_L",
    "cheekSquint_R",
    "eyeBlink_L",
    "eyeBlink_R",
    "eyeLookDown_L",
    "eyeLookDown_R",
    "eyeLookIn_L",
    "eyeLookIn_R",
    "eyeLookOut_L",
    "eyeLookOut_R",
    "eyeLookUp_L",
    "eyeLookUp_R",
    "eyeSquint_L",
    "eyeSquint_R",
    "eyeWide_L",
    "eyeWide_R",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimple_L",
    "mouthDimple_R",
    "mouthFrown_L",
    "mouthFrown_R",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDown_L",
    "mouthLowerDown_R",
    "mouthPress_L",
    "mouthPress_R",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmile_L",
    "mouthSmile_R",
    "mouthStretch_L",
    "mouthStretch_R",
    "mouthUpperUp_L",
    "mouthUpperUp_R",
    "noseSneer_L",
    "noseSneer_R",
]


class FakeFaceModel:
    expression_names = ICT_NAMES


class RealVideoPipelineTests(unittest.TestCase):
    def test_all_said_arkit_names_map_to_known_ict_or_intended_zero_fill(self) -> None:
        allowed_zero_fill = {
            "eyeBlink_L",
            "eyeBlink_R",
            "eyeLookDown_L",
            "eyeLookDown_R",
            "eyeLookIn_L",
            "eyeLookIn_R",
            "eyeLookOut_L",
            "eyeLookOut_R",
            "eyeLookUp_L",
            "eyeLookUp_R",
            "eyeSquint_L",
            "eyeSquint_R",
            "eyeWide_L",
            "eyeWide_R",
            "browDown_L",
            "browDown_R",
            "browInnerUp_L",
            "browInnerUp_R",
            "browOuterUp_L",
            "browOuterUp_R",
        }
        coeffs = np.ones((2, len(SAID_ARKIT_NAMES)), dtype=np.float32) * 0.5
        ict_coeffs, report = convert_arkit_to_ict(coeffs, SAID_ARKIT_NAMES, ICT_NAMES)

        self.assertEqual(ict_coeffs.shape, (2, len(ICT_NAMES)))
        self.assertEqual(report.missing_ict_channels, {})
        self.assertTrue(set(report.zero_filled_ict_channels).issubset(allowed_zero_fill))
        self.assertEqual(arkit_name_to_ict_names("mouthSmileLeft"), ["mouthSmile_L"])
        self.assertEqual(arkit_name_to_ict_names("mouthLowerDownRight"), ["mouthLowerDown_R"])
        self.assertEqual(arkit_name_to_ict_names("cheekPuff"), ["cheekPuff_L", "cheekPuff_R"])

    def test_qp_fitter_returns_bounded_coefficients_and_preserves_frame_count(self) -> None:
        neutral = np.zeros((4, 3), dtype=np.float32)
        deltas = np.zeros((2, 4, 3), dtype=np.float32)
        deltas[0, 0, 0] = 1.0
        deltas[1, 1, 1] = 2.0
        basis = ARKitBasis(names=["a", "b"], neutral_vertices=neutral, deltas=deltas)
        expected = np.asarray([[0.0, 0.0], [0.4, 0.2], [0.7, 0.8]], dtype=np.float32)
        vertices = neutral[None, :, :] + np.einsum("fc,cvd->fvd", expected, deltas)

        coeffs, diagnostics = fit_coefficients(vertices, basis, temporal_delta=1.0, prefer_qp=False)

        self.assertEqual(coeffs.shape, expected.shape)
        self.assertTrue(np.all(coeffs >= 0.0))
        self.assertTrue(np.all(coeffs <= 1.0))
        np.testing.assert_allclose(coeffs, expected, atol=1e-4)
        self.assertEqual(len(diagnostics.full_head_rmse), len(expected))

    def test_qp_fitter_handles_neutral_all_zero_input(self) -> None:
        neutral = np.zeros((3, 3), dtype=np.float32)
        deltas = np.ones((2, 3, 3), dtype=np.float32)
        basis = ARKitBasis(names=["a", "b"], neutral_vertices=neutral, deltas=deltas)
        vertices = np.repeat(neutral[None, :, :], 4, axis=0)

        coeffs, diagnostics = fit_coefficients(vertices, basis, temporal_delta=0.1, prefer_qp=False)

        np.testing.assert_allclose(coeffs, 0.0, atol=1e-6)
        self.assertFalse(np.any(diagnostics.failed_frame_mask))

    def test_json_loader_accepts_beat_and_arkit_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            beat_json = tmp_path / "beat.json"
            beat_json.write_text(
                json.dumps(
                    {
                        "names": ["mouthSmileLeft", "cheekPuff", "jawOpen"],
                        "frames": [
                            {"time": 0.0, "weights": [0.2, 0.3, 0.1]},
                            {"time": 0.5, "weights": [0.4, 0.5, 0.2]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            arkit_json = tmp_path / "arkit.json"
            arkit_json.write_text(
                json.dumps(
                    {
                        "source": "smirk_said_arkit",
                        "fps": 2,
                        "names": ["mouthLowerDownRight", "mouthFunnel"],
                        "frames": [
                            {"time": 0.0, "weights": [0.6, 0.7]},
                            {"time": 0.5, "weights": [0.8, 0.9]},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            beat_seq = load_blendshape_json_sequence(beat_json, FakeFaceModel(), source_fps=2, target_fps=2)
            arkit_seq = load_blendshape_json_sequence(arkit_json, FakeFaceModel(), target_fps=2)

        self.assertEqual(beat_seq.shape, (2, len(ICT_NAMES)))
        self.assertAlmostEqual(beat_seq[0, ICT_NAMES.index("mouthSmile_L")], 0.2)
        self.assertAlmostEqual(beat_seq[0, ICT_NAMES.index("cheekPuff_L")], 0.3)
        self.assertAlmostEqual(beat_seq[0, ICT_NAMES.index("cheekPuff_R")], 0.3)
        self.assertAlmostEqual(beat_seq[0, ICT_NAMES.index("jawOpen")], 0.1)
        self.assertEqual(arkit_seq.shape, (2, len(ICT_NAMES)))
        self.assertAlmostEqual(arkit_seq[0, ICT_NAMES.index("mouthLowerDown_R")], 0.6)
        self.assertAlmostEqual(arkit_seq[0, ICT_NAMES.index("mouthFunnel")], 0.7)

    def test_said_residual_loader_fits_head_crop_from_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            said_dir = root / "said_data"
            said_dir.mkdir()
            (said_dir / "ARKit_blendshapes.txt").write_text("a\nb\n", encoding="utf-8")
            (said_dir / "FLAME_head_idx.txt").write_text("0\n2\n", encoding="utf-8")
            deltas = {
                "speaker": {
                    "a": np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
                    "b": np.asarray([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32),
                }
            }
            with (said_dir / "blendshape_residuals.pickle").open("wb") as f:
                pickle.dump(deltas, f)

            neutral = np.zeros((4, 3), dtype=np.float32)
            vertices = np.repeat(neutral[None, :, :], 2, axis=0)
            vertices[1, 0, 0] = 0.25
            vertices[1, 2, 1] = 1.0
            vertices_npz = root / "smirk_flame_vertices.npz"
            np.savez_compressed(vertices_npz, vertices=vertices, neutral_vertices=neutral, fps=np.float32(25))
            coeffs_csv = root / "arkit_coeffs.csv"
            diagnostics_json = root / "diagnostics.json"

            coeffs, diagnostics, basis = fit_smirk_vertices_file(
                vertices_npz,
                said_dir,
                coeffs_csv,
                diagnostics_json,
                said_person_id="speaker",
                temporal_delta=1.0,
            )

            self.assertEqual(basis.vertex_indices.tolist(), [0, 2])
            self.assertTrue(coeffs_csv.is_file())
            self.assertTrue(diagnostics_json.is_file())
            np.testing.assert_allclose(coeffs[1], [0.25, 0.5], atol=1e-4)
            self.assertFalse(np.any(diagnostics.failed_frame_mask))

    def test_discover_videos_defaults_to_first_sorted_mp4_for_smoke_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.mp4").write_bytes(b"")
            (root / "a.mp4").write_bytes(b"")
            (root / "ignore.txt").write_text("nope", encoding="utf-8")

            videos = discover_videos(video=None, video_dir=root, smoke=True)

        self.assertEqual([path.name for path in videos], ["a.mp4"])

    def test_discover_videos_can_return_all_sorted_mp4s(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.mp4").write_bytes(b"")
            (root / "a.mp4").write_bytes(b"")

            videos = discover_videos(video=None, video_dir=root, smoke=False)

        self.assertEqual([path.name for path in videos], ["a.mp4", "b.mp4"])

    def test_resample_sequence_to_frames_preserves_target_frame_count(self) -> None:
        seq = np.arange(8, dtype=np.float32).reshape(4, 2)

        out = resample_sequence_to_frames(seq, target_frames=2, source_fps=50.0, target_fps=25.0)

        self.assertEqual(out.shape, (2, 2))
        np.testing.assert_allclose(out[0], seq[0], atol=1e-6)

    def test_apply_render_shift_uses_renderer_sign_convention(self) -> None:
        seq = np.arange(5, dtype=np.float32).reshape(5, 1)

        delayed = apply_render_shift(seq, shift_seconds=0.08, fps=25.0)
        advanced = apply_render_shift(seq, shift_seconds=-0.08, fps=25.0)

        np.testing.assert_array_equal(delayed[:, 0], [0, 0, 0, 1, 2])
        np.testing.assert_array_equal(advanced[:, 0], [2, 3, 4, 4, 4])

    def test_shift_tongue_motion_file_applies_subframe_render_shift_at_50fps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "tongue.npy"
            dst = root / "tongue_shifted.npy"
            motion = np.arange(5, dtype=np.float32).reshape(5, 1)
            np.save(src, motion)

            shift_tongue_motion_file(src, dst, shift_seconds=-0.02, tongue_fps=50.0)

            shifted = np.load(dst)

        np.testing.assert_array_equal(shifted[:, 0], [1, 2, 3, 4, 4])


if __name__ == "__main__":
    unittest.main()
