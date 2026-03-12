from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tongue_scripts.phoneme_lbfgsb_optimizer import (
    MouthBox,
    OptimizationConfig,
    OptimizationWeights,
    PhonemeSpan,
    Plane,
    RegionRect3D,
    RigFrameResult,
    SpanOptimizationContext,
    TIP_ANCHOR_IDX,
    build_contact_window_mask,
    build_knots_from_traj,
    compute_contact_loss,
    distance_point_to_region,
    export_debug_report,
    interpolate_knots,
    map_phoneme_to_class,
    make_bounds,
    normalize_phone,
    optimize_span,
    optimize_utterance,
    parse_textgrid,
    point_inside_mouth_box,
    signed_distance_to_plane,
)


class DummyRigBackend:
    def __init__(self, expose_keypoints: bool = True) -> None:
        self.expose_keypoints = expose_keypoints

    def forward_frame(self, anchor_positions, jaw_controls_t, face_controls_t):
        anchors = np.asarray(anchor_positions, dtype=np.float64)
        keypoints = {}
        if self.expose_keypoints:
            keypoints = {
                "back": anchors[0].copy(),
                "dorsum": anchors[1].copy(),
                "blade": anchors[2].copy(),
                "tip": anchors[3].copy(),
            }
        return RigFrameResult(tongue_mesh=anchors.copy(), tongue_keypoints=keypoints, face_mesh=None)


def _make_region(point):
    return RegionRect3D(
        point=np.asarray(point, dtype=np.float64),
        normal=np.array([0.0, 1.0, 0.0]),
        tangent_u=np.array([1.0, 0.0, 0.0]),
        tangent_v=np.array([0.0, 0.0, 1.0]),
        extent_u=0.5,
        extent_v=0.5,
    )


def test_normalize_phone_and_class_mapping():
    assert normalize_phone(" ah0 ") == "AH"
    assert normalize_phone("th") == "TH"
    assert map_phoneme_to_class("T") == "alveolar"
    assert map_phoneme_to_class("DH") == "interdental"
    assert map_phoneme_to_class("AA") == "other"


def test_parse_textgrid_skips_blank_and_strips_stress(tmp_path: Path):
    textgrid = tmp_path / "sample.TextGrid"
    textgrid.write_text(
        '\n'.join(
            [
                'File type = "ooTextFile"',
                'Object class = "TextGrid"',
                "",
                "item []:",
                "    item [1]:",
                '        class = "IntervalTier"',
                '        name = "phones"',
                "        intervals: size = 3",
                "        intervals [1]:",
                "            xmin = 0",
                "            xmax = 0.1",
                '            text = ""',
                "        intervals [2]:",
                "            xmin = 0.1",
                "            xmax = 0.2",
                '            text = "AY1"',
                "        intervals [3]:",
                "            xmin = 0.2",
                "            xmax = 0.3",
                '            text = "TH"',
            ]
        ),
        encoding="utf-8",
    )

    spans = parse_textgrid(textgrid, fps=50.0)

    assert [span.label for span in spans] == ["AY1", "TH"]
    assert spans[0].label_norm == "AY"
    assert spans[1].phoneme_class == "interdental"
    assert spans[0].start_frame == 5
    assert spans[1].end_frame == 14


def test_build_knots_and_interpolate_short_span():
    traj = np.arange(5 * 4 * 3, dtype=np.float64).reshape(5, 4, 3)
    knots = build_knots_from_traj(traj, [1, 2, 3])
    interp = interpolate_knots(knots, 1)

    assert knots.shape == (4, 3, 3)
    assert interp.shape == (1, 4, 3)
    np.testing.assert_allclose(knots[:, 0], traj[1])
    np.testing.assert_allclose(knots[:, 1], traj[2])
    np.testing.assert_allclose(knots[:, 2], traj[3])


def test_geometry_helpers():
    region = _make_region([0.0, 0.0, 0.0])
    assert distance_point_to_region(np.array([0.0, 0.0, 0.0]), region) == 0.0
    assert np.isclose(distance_point_to_region(np.array([0.0, 2.0, 0.0]), region), 2.0)

    mouth_box = MouthBox(min_corner=np.array([-1.0, -1.0, -1.0]), max_corner=np.array([1.0, 1.0, 1.0]))
    assert point_inside_mouth_box(np.array([0.0, 0.0, 0.0]), mouth_box)
    assert not point_inside_mouth_box(np.array([2.0, 0.0, 0.0]), mouth_box)

    plane = Plane(point=np.zeros(3), normal=np.array([0.0, 1.0, 0.0]))
    assert signed_distance_to_plane(np.array([0.0, 2.0, 0.0]), plane) == 2.0
    assert signed_distance_to_plane(np.array([0.0, -2.0, 0.0]), plane) == -2.0


def test_contact_window_and_tip_fallback_uses_anchor_three():
    region = _make_region([0.0, 0.0, 3.0])
    anchor_traj = np.zeros((5, 4, 3), dtype=np.float64)
    anchor_traj[:, TIP_ANCHOR_IDX, 2] = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    rig_results = [RigFrameResult(tongue_mesh=None, tongue_keypoints={}, face_mesh=None) for _ in range(5)]

    loss, min_dist = compute_contact_loss(
        anchor_traj=anchor_traj,
        rig_results=rig_results,
        phoneme_class="alveolar",
        contact_window_mask=build_contact_window_mask(5, 0.4),
        alveolar_region=region,
        interdental_region=None,
        tau_alveolar=0.0,
        tau_interdental=0.0,
    )

    assert loss > 0.0
    assert np.isclose(min_dist, 1.0)


def test_make_bounds_supports_delta_and_absolute_modes():
    reference_knots = np.zeros((4, 3, 3), dtype=np.float64)
    delta_bounds = make_bounds(
        reference_knots,
        OptimizationConfig(mode="delta", delta_bounds_mm=8.0, tip_delta_bounds_mm=10.0),
    )
    absolute_bounds = make_bounds(
        reference_knots,
        OptimizationConfig(mode="absolute", delta_bounds_mm=8.0, tip_delta_bounds_mm=10.0),
    )

    assert delta_bounds.lb.shape == (36,)
    assert absolute_bounds.ub.shape == (36,)
    assert np.isclose(delta_bounds.lb[0], -8.0)
    assert np.isclose(delta_bounds.ub[-1], 10.0)
    assert np.isclose(absolute_bounds.lb[0], -8.0)
    assert np.isclose(absolute_bounds.ub[-1], 10.0)


def test_optimize_span_reduces_contact_distance():
    n_frames = 7
    wavlm = np.zeros((n_frames, 4, 3), dtype=np.float64)
    span = PhonemeSpan(
        label="T",
        label_norm="T",
        phoneme_class="alveolar",
        start_time=0.0,
        end_time=0.14,
        start_frame=0,
        end_frame=n_frames - 1,
    )
    context = SpanOptimizationContext(
        span=span,
        frame_positions=np.arange(n_frames),
        wavlm_traj=wavlm,
        jaw_traj=[0.0] * n_frames,
        face_traj=[{} for _ in range(n_frames)],
        rig_forward=DummyRigBackend(expose_keypoints=False),
        config=OptimizationConfig(maxiter=80, contact_window_fraction=0.4),
        weights=OptimizationWeights(
            lambda_data=0.1,
            lambda_contact_alveolar=10.0,
            lambda_contact_interdental=3.0,
            lambda_contact_other=0.0,
            lambda_smooth=0.05,
            lambda_prior=0.01,
            lambda_compat=0.0,
        ),
        alveolar_region=_make_region([0.0, 0.0, 4.0]),
    )

    result = optimize_span(context)

    assert result.final_losses["total"] <= result.initial_losses["total"]
    assert result.final_tip_distance is not None
    assert result.initial_tip_distance is not None
    assert result.final_tip_distance < result.initial_tip_distance
    assert result.optimized_traj.shape == (n_frames, 4, 3)


def test_optimize_utterance_and_export_debug_report(tmp_path: Path):
    n_frames = 12
    frames = np.arange(n_frames)
    wavlm = np.zeros((n_frames, 4, 3), dtype=np.float64)
    spans = [
        PhonemeSpan("T", "T", "alveolar", 0.0, 0.12, 0, 5),
        PhonemeSpan("AA1", "AA", "other", 0.12, 0.24, 6, 11),
    ]
    optimized, reports = optimize_utterance(
        frames=frames,
        wavlm_anchor_pred=wavlm,
        jaw_controls=[0.0] * n_frames,
        face_controls=[{} for _ in range(n_frames)],
        phoneme_spans=spans,
        rig_forward=DummyRigBackend(),
        config=OptimizationConfig(maxiter=60, seam_blend_frames=2),
        weights=OptimizationWeights(
            lambda_data=0.1,
            lambda_contact_alveolar=10.0,
            lambda_contact_interdental=10.0,
            lambda_contact_other=0.0,
            lambda_smooth=0.05,
            lambda_prior=0.01,
            lambda_compat=0.0,
        ),
        alveolar_region=_make_region([0.0, 0.0, 3.0]),
    )

    assert optimized.shape == (n_frames, 4, 3)
    assert len(reports) == 1
    assert reports[0].final_tip_distance < reports[0].initial_tip_distance
    np.testing.assert_allclose(optimized[6:], wavlm[6:])

    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "summary.csv"
    tip_path = tmp_path / "tips.json"
    export_debug_report(
        span_results=reports,
        json_output_path=json_path,
        summary_csv_path=csv_path,
        tip_output_path=tip_path,
        baseline_anchor_traj=wavlm,
        optimized_anchor_traj=optimized,
    )

    assert json_path.exists()
    assert csv_path.exists()
    assert tip_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(payload["spans"]) == 1
    assert payload["spans"][0]["phoneme_label"] == "T"
