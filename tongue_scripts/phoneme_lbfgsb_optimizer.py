from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Mapping, Optional, Protocol, Sequence

import numpy as np
from scipy.optimize import Bounds, minimize

logger = logging.getLogger(__name__)

BACK_ANCHOR_IDX = 0
DORSUM_ANCHOR_IDX = 1
BLADE_ANCHOR_IDX = 2
TIP_ANCHOR_IDX = 3

ALVEOLAR_PHONEMES = frozenset({"T", "D", "N", "L"})
INTERDENTAL_PHONEMES = frozenset({"TH", "DH"})
DEFAULT_TARGET_PHONEME_CLASSES = frozenset({"alveolar", "interdental"})
KEYPOINT_NAMES = ("back", "dorsum", "blade", "tip")


@dataclass(frozen=True)
class PhonemeSpan:
    label: str
    label_norm: str
    phoneme_class: str
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class OptimizationWeights:
    lambda_data: float = 1.0
    lambda_contact_alveolar: float = 3.0
    lambda_contact_interdental: float = 3.0
    lambda_contact_other: float = 0.0
    lambda_smooth: float = 0.5
    lambda_prior: float = 0.2
    lambda_compat: float = 1.0

    def contact_lambda(self, phoneme_class: str) -> float:
        if phoneme_class == "alveolar":
            return self.lambda_contact_alveolar
        if phoneme_class == "interdental":
            return self.lambda_contact_interdental
        return self.lambda_contact_other


@dataclass(frozen=True)
class RegionRect3D:
    point: np.ndarray
    normal: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray
    extent_u: float
    extent_v: float

    def __post_init__(self) -> None:
        point = np.asarray(self.point, dtype=np.float64)
        normal = _normalize(np.asarray(self.normal, dtype=np.float64))
        tangent_u = np.asarray(self.tangent_u, dtype=np.float64)
        tangent_u = tangent_u - np.dot(tangent_u, normal) * normal
        tangent_u = _normalize(tangent_u)
        tangent_v = np.asarray(self.tangent_v, dtype=np.float64)
        tangent_v = tangent_v - np.dot(tangent_v, normal) * normal
        tangent_v = tangent_v - np.dot(tangent_v, tangent_u) * tangent_u
        tangent_v = _normalize(tangent_v)
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "tangent_u", tangent_u)
        object.__setattr__(self, "tangent_v", tangent_v)
        object.__setattr__(self, "extent_u", float(self.extent_u))
        object.__setattr__(self, "extent_v", float(self.extent_v))


@dataclass(frozen=True)
class Plane:
    point: np.ndarray
    normal: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", np.asarray(self.point, dtype=np.float64))
        object.__setattr__(self, "normal", _normalize(np.asarray(self.normal, dtype=np.float64)))


@dataclass(frozen=True)
class MouthBox:
    min_corner: np.ndarray
    max_corner: np.ndarray

    def __post_init__(self) -> None:
        min_corner = np.asarray(self.min_corner, dtype=np.float64)
        max_corner = np.asarray(self.max_corner, dtype=np.float64)
        if min_corner.shape != (3,) or max_corner.shape != (3,):
            raise ValueError("MouthBox corners must be shape (3,)")
        object.__setattr__(self, "min_corner", np.minimum(min_corner, max_corner))
        object.__setattr__(self, "max_corner", np.maximum(min_corner, max_corner))


@dataclass(frozen=True)
class OptimizationConfig:
    mode: str = "delta"
    fps: float = 50.0
    anchor_order: tuple[str, str, str, str] = ("back", "dorsum", "blade", "tip")
    anchor_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.5, 2.0)
    delta_bounds_mm: float = 8.0
    tip_delta_bounds_mm: Optional[float] = None
    tau_alveolar_mm: float = 2.5
    tau_interdental_mm: float = 2.0
    contact_window_fraction: float = 0.4
    maxiter: int = 100
    seam_blend_frames: int = 2
    mm_per_world_unit: float = 1.0

    def __post_init__(self) -> None:
        mode = self.mode.lower().strip()
        if mode not in {"delta", "absolute"}:
            raise ValueError(f"Unsupported optimization mode: {self.mode}")
        if len(self.anchor_order) != 4:
            raise ValueError("anchor_order must contain 4 entries")
        if len(self.anchor_weights) != 4:
            raise ValueError("anchor_weights must contain 4 entries")
        if not (0.0 < float(self.contact_window_fraction) <= 1.0):
            raise ValueError("contact_window_fraction must be in (0, 1]")
        if float(self.mm_per_world_unit) <= 0.0:
            raise ValueError("mm_per_world_unit must be positive")
        object.__setattr__(self, "mode", mode)

    def mm_to_world(self, value_mm: float) -> float:
        return float(value_mm) / float(self.mm_per_world_unit)

    @property
    def tau_alveolar(self) -> float:
        return self.mm_to_world(self.tau_alveolar_mm)

    @property
    def tau_interdental(self) -> float:
        return self.mm_to_world(self.tau_interdental_mm)

    @property
    def delta_bounds_world(self) -> float:
        return self.mm_to_world(self.delta_bounds_mm)

    @property
    def tip_delta_bounds_world(self) -> float:
        tip_mm = self.tip_delta_bounds_mm if self.tip_delta_bounds_mm is not None else self.delta_bounds_mm
        return self.mm_to_world(tip_mm)


@dataclass
class RigFrameResult:
    tongue_mesh: Optional[np.ndarray]
    tongue_keypoints: Mapping[str, np.ndarray]
    face_mesh: Optional[np.ndarray] = None


class RigForwardProtocol(Protocol):
    def forward_frame(
        self,
        anchor_positions: np.ndarray,
        jaw_controls_t: Any,
        face_controls_t: Any,
    ) -> RigFrameResult:
        ...


@dataclass
class SpanOptimizationResult:
    span: PhonemeSpan
    success: bool
    status: int
    n_iters: int
    initial_losses: dict[str, float]
    final_losses: dict[str, float]
    initial_tip_distance: Optional[float]
    final_tip_distance: Optional[float]
    delta_norm: float
    optimized_knots: np.ndarray
    optimized_traj: np.ndarray
    message: str = ""
    anchor_delta_magnitudes: tuple[float, float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0))

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "phoneme_label": self.span.label,
            "phoneme_class": self.span.phoneme_class,
            "start_time": self.span.start_time,
            "end_time": self.span.end_time,
            "start_frame": self.span.start_frame,
            "end_frame": self.span.end_frame,
            "success": self.success,
            "status": self.status,
            "message": self.message,
            "n_iters": self.n_iters,
            "initial_total_loss": self.initial_losses.get("total"),
            "final_total_loss": self.final_losses.get("total"),
            "initial_losses": self.initial_losses,
            "final_losses": self.final_losses,
            "initial_tip_distance": self.initial_tip_distance,
            "final_tip_distance": self.final_tip_distance,
            "delta_norm": self.delta_norm,
            "anchor_delta_magnitudes": list(self.anchor_delta_magnitudes),
            "optimized_knots": self.optimized_knots.tolist(),
        }


@dataclass
class SpanOptimizationContext:
    span: PhonemeSpan
    frame_positions: np.ndarray
    wavlm_traj: np.ndarray
    jaw_traj: Sequence[Any]
    face_traj: Sequence[Any]
    rig_forward: RigForwardProtocol
    config: OptimizationConfig
    weights: OptimizationWeights
    alveolar_region: Optional[RegionRect3D] = None
    interdental_region: Optional[RegionRect3D] = None
    mouth_box: Optional[MouthBox] = None
    forbidden_planes: Sequence[Plane] = field(default_factory=tuple)
    reference_knots: Optional[np.ndarray] = None
    bounds: Optional[Bounds] = None

    def __post_init__(self) -> None:
        frame_positions = np.asarray(self.frame_positions, dtype=np.int32)
        if frame_positions.ndim != 1 or len(frame_positions) == 0:
            raise ValueError("frame_positions must be a non-empty 1D array")
        wavlm_traj = np.asarray(self.wavlm_traj, dtype=np.float64)
        if wavlm_traj.shape != (len(frame_positions), 4, 3):
            raise ValueError("wavlm_traj must be shape (T, 4, 3) for this span")
        object.__setattr__(self, "frame_positions", frame_positions)
        object.__setattr__(self, "wavlm_traj", wavlm_traj)
        if self.reference_knots is None:
            object.__setattr__(self, "reference_knots", build_knots_from_traj(wavlm_traj, np.arange(len(wavlm_traj))))

    @property
    def n_frames(self) -> int:
        return int(len(self.frame_positions))

    @property
    def contact_window_mask(self) -> np.ndarray:
        return build_contact_window_mask(self.n_frames, self.config.contact_window_fraction)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize near-zero vector")
    return vector / norm


def hinge_sq(value: float | np.ndarray) -> float | np.ndarray:
    return np.square(np.maximum(value, 0.0))


def normalize_phone(label: str) -> str:
    return re.sub(r"[0-9]", "", label.strip().upper())


def map_phoneme_to_class(label_norm: str) -> str:
    if label_norm in ALVEOLAR_PHONEMES:
        return "alveolar"
    if label_norm in INTERDENTAL_PHONEMES:
        return "interdental"
    return "other"


def parse_textgrid(
    textgrid_path: str | Path,
    fps: float,
    tier_name: str = "phones",
    total_frames: Optional[int] = None,
) -> list[PhonemeSpan]:
    spans: list[PhonemeSpan] = []
    path = Path(textgrid_path)
    in_tier = False
    current: dict[str, str] = {}

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("item ["):
                in_tier = False
                continue
            if line.startswith('name = "'):
                tier = line.split("=", 1)[1].strip().strip('"')
                in_tier = tier == tier_name
                continue
            if not in_tier:
                continue
            if line.startswith("intervals ["):
                current = {}
                continue
            if line.startswith("xmin ="):
                current["start"] = line.split("=", 1)[1].strip()
                continue
            if line.startswith("xmax ="):
                current["end"] = line.split("=", 1)[1].strip()
                continue
            if line.startswith("text ="):
                label = line.split("=", 1)[1].strip()
                if label.startswith('"') and label.endswith('"'):
                    label = label[1:-1]
                current["label"] = label
                if {"start", "end", "label"} <= current.keys():
                    if not label.strip():
                        continue
                    start_time = float(current["start"])
                    end_time = float(current["end"])
                    start_frame = int(np.floor(start_time * fps))
                    end_frame = max(start_frame, int(np.ceil(end_time * fps)) - 1)
                    if total_frames is not None and total_frames > 0:
                        start_frame = min(max(start_frame, 0), total_frames - 1)
                        end_frame = min(max(end_frame, start_frame), total_frames - 1)
                    label_norm = normalize_phone(label)
                    spans.append(
                        PhonemeSpan(
                            label=label,
                            label_norm=label_norm,
                            phoneme_class=map_phoneme_to_class(label_norm),
                            start_time=start_time,
                            end_time=end_time,
                            start_frame=start_frame,
                            end_frame=end_frame,
                        )
                    )
    return spans


def build_knots_from_traj(anchor_traj: np.ndarray, frame_ids: Sequence[int]) -> np.ndarray:
    frame_ids_arr = np.asarray(frame_ids, dtype=np.int32)
    if frame_ids_arr.ndim != 1 or len(frame_ids_arr) == 0:
        raise ValueError("frame_ids must be a non-empty 1D sequence")
    traj = np.asarray(anchor_traj, dtype=np.float64)
    if traj.ndim != 3 or traj.shape[1:] != (4, 3):
        raise ValueError("anchor_traj must have shape (T, 4, 3)")
    if np.max(frame_ids_arr) >= len(traj) or np.min(frame_ids_arr) < 0:
        raise IndexError("frame_ids out of range for anchor_traj")
    mid_idx = frame_ids_arr[len(frame_ids_arr) // 2]
    sample = traj[[frame_ids_arr[0], mid_idx, frame_ids_arr[-1]]]
    return np.transpose(sample, (1, 0, 2))


def interpolate_knots(knots: np.ndarray, n_frames: int) -> np.ndarray:
    knots_arr = np.asarray(knots, dtype=np.float64)
    if knots_arr.shape != (4, 3, 3):
        raise ValueError("knots must have shape (4, 3, 3)")
    if n_frames < 0:
        raise ValueError("n_frames must be non-negative")
    if n_frames == 0:
        return np.zeros((0, 4, 3), dtype=np.float64)
    src_t = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    dst_t = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)
    out = np.empty((n_frames, 4, 3), dtype=np.float64)
    for anchor_idx in range(4):
        for coord_idx in range(3):
            out[:, anchor_idx, coord_idx] = np.interp(dst_t, src_t, knots_arr[anchor_idx, :, coord_idx])
    return out


def signed_distance_to_plane(point: np.ndarray, plane: Plane) -> float:
    pt = np.asarray(point, dtype=np.float64)
    return float(np.dot(pt - plane.point, plane.normal))


def distance_point_to_region(point: np.ndarray, region: RegionRect3D) -> float:
    pt = np.asarray(point, dtype=np.float64)
    rel = pt - region.point
    u = float(np.dot(rel, region.tangent_u))
    v = float(np.dot(rel, region.tangent_v))
    u_clamped = np.clip(u, -region.extent_u, region.extent_u)
    v_clamped = np.clip(v, -region.extent_v, region.extent_v)
    closest = region.point + u_clamped * region.tangent_u + v_clamped * region.tangent_v
    return float(np.linalg.norm(pt - closest))


def point_inside_mouth_box(point: np.ndarray, mouth_box: MouthBox) -> bool:
    pt = np.asarray(point, dtype=np.float64)
    return bool(np.all(pt >= mouth_box.min_corner) and np.all(pt <= mouth_box.max_corner))


def build_contact_window_mask(n_frames: int, fraction: float) -> np.ndarray:
    if n_frames <= 0:
        return np.zeros((0,), dtype=bool)
    half_margin = max(0.0, (1.0 - fraction) * 0.5)
    positions = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)
    return (positions >= half_margin - 1e-9) & (positions <= 1.0 - half_margin + 1e-9)


def rig_forward_batch(
    anchor_traj: np.ndarray,
    jaw_traj: Sequence[Any],
    face_traj: Sequence[Any],
    rig_forward: RigForwardProtocol,
) -> list[RigFrameResult]:
    traj = np.asarray(anchor_traj, dtype=np.float64)
    if traj.ndim != 3 or traj.shape[1:] != (4, 3):
        raise ValueError("anchor_traj must have shape (T, 4, 3)")
    if len(jaw_traj) != len(traj) or len(face_traj) != len(traj):
        raise ValueError("jaw_traj and face_traj must align with anchor_traj length")
    return [
        rig_forward.forward_frame(traj[i], jaw_traj[i], face_traj[i])
        for i in range(len(traj))
    ]


def _get_keypoint(
    rig_result: RigFrameResult,
    name: str,
    anchor_positions: np.ndarray,
) -> np.ndarray:
    if name in rig_result.tongue_keypoints:
        return np.asarray(rig_result.tongue_keypoints[name], dtype=np.float64)
    fallback_idx = {
        "back": BACK_ANCHOR_IDX,
        "dorsum": DORSUM_ANCHOR_IDX,
        "blade": BLADE_ANCHOR_IDX,
        "tip": TIP_ANCHOR_IDX,
    }[name]
    return np.asarray(anchor_positions[fallback_idx], dtype=np.float64)


def compute_data_loss(
    anchor_traj: np.ndarray,
    wavlm_anchor_pred: np.ndarray,
    anchor_weights: Sequence[float],
) -> float:
    traj = np.asarray(anchor_traj, dtype=np.float64)
    target = np.asarray(wavlm_anchor_pred, dtype=np.float64)
    if traj.shape != target.shape:
        raise ValueError("anchor_traj and wavlm_anchor_pred must have identical shape")
    weights = np.asarray(anchor_weights, dtype=np.float64)
    if weights.shape != (4,):
        raise ValueError("anchor_weights must have shape (4,)")
    sq = np.sum(np.square(traj - target), axis=2)
    return float(np.sum(sq * weights[None, :]))


def compute_contact_loss(
    anchor_traj: np.ndarray,
    rig_results: Sequence[RigFrameResult],
    phoneme_class: str,
    contact_window_mask: np.ndarray,
    alveolar_region: Optional[RegionRect3D],
    interdental_region: Optional[RegionRect3D],
    tau_alveolar: float,
    tau_interdental: float,
) -> tuple[float, Optional[float]]:
    if phoneme_class == "alveolar":
        region = alveolar_region
        tau = tau_alveolar
    elif phoneme_class == "interdental":
        region = interdental_region
        tau = tau_interdental
    else:
        return 0.0, None
    if region is None:
        return 0.0, None

    loss = 0.0
    distances: list[float] = []
    mask = np.asarray(contact_window_mask, dtype=bool)
    for i in range(len(anchor_traj)):
        if i >= len(mask) or not mask[i]:
            continue
        keypoint = _get_keypoint(rig_results[i], "tip", anchor_traj[i])
        # Minimal target mode: contact is defined as tip-to-target-point distance.
        dist = float(np.linalg.norm(keypoint - region.point))
        distances.append(dist)
        loss += float(hinge_sq(dist - tau))
    min_distance = float(np.min(distances)) if distances else None
    return float(loss), min_distance


def compute_smooth_loss(knots: np.ndarray) -> float:
    knots_arr = np.asarray(knots, dtype=np.float64)
    if knots_arr.shape != (4, 3, 3):
        raise ValueError("knots must have shape (4, 3, 3)")
    first_diff = knots_arr[:, 1:, :] - knots_arr[:, :-1, :]
    second_diff = knots_arr[:, 2:, :] - 2.0 * knots_arr[:, 1:2, :] + knots_arr[:, :1, :]
    return float(np.sum(np.square(first_diff)) + np.sum(np.square(second_diff)))


def compute_prior_loss(knots: np.ndarray, reference_knots: np.ndarray) -> float:
    knots_arr = np.asarray(knots, dtype=np.float64)
    ref = np.asarray(reference_knots, dtype=np.float64)
    if knots_arr.shape != (4, 3, 3) or ref.shape != (4, 3, 3):
        raise ValueError("knots and reference_knots must both have shape (4, 3, 3)")
    return float(np.sum(np.square(knots_arr - ref)))


def compute_compat_loss(
    anchor_traj: np.ndarray,
    rig_results: Sequence[RigFrameResult],
    mouth_box: Optional[MouthBox],
    forbidden_planes: Sequence[Plane],
) -> float:
    loss = 0.0
    for i in range(len(anchor_traj)):
        for key in KEYPOINT_NAMES:
            point = _get_keypoint(rig_results[i], key, anchor_traj[i])
            if mouth_box is not None:
                below = np.maximum(mouth_box.min_corner - point, 0.0)
                above = np.maximum(point - mouth_box.max_corner, 0.0)
                loss += float(np.sum(np.square(below + above)))
            for plane in forbidden_planes:
                loss += float(hinge_sq(signed_distance_to_plane(point, plane)))
    return float(loss)


def _vector_to_knots(x: np.ndarray, context: SpanOptimizationContext) -> np.ndarray:
    vec = np.asarray(x, dtype=np.float64).reshape(4, 3, 3)
    if context.config.mode == "delta":
        return context.reference_knots + vec
    return vec


def _knots_to_vector(knots: np.ndarray, context: SpanOptimizationContext) -> np.ndarray:
    knots_arr = np.asarray(knots, dtype=np.float64)
    if context.config.mode == "delta":
        return (knots_arr - context.reference_knots).reshape(-1)
    return knots_arr.reshape(-1)


def make_bounds(reference_knots: np.ndarray, config: OptimizationConfig) -> Bounds:
    reference = np.asarray(reference_knots, dtype=np.float64)
    if reference.shape != (4, 3, 3):
        raise ValueError("reference_knots must have shape (4, 3, 3)")
    if config.mode == "delta":
        low = np.full((4, 3, 3), -config.delta_bounds_world, dtype=np.float64)
        high = np.full((4, 3, 3), config.delta_bounds_world, dtype=np.float64)
        low[TIP_ANCHOR_IDX, :, :] = -config.tip_delta_bounds_world
        high[TIP_ANCHOR_IDX, :, :] = config.tip_delta_bounds_world
        return Bounds(low.reshape(-1), high.reshape(-1))
    radius = np.full((4, 3, 3), config.delta_bounds_world, dtype=np.float64)
    radius[TIP_ANCHOR_IDX, :, :] = config.tip_delta_bounds_world
    return Bounds((reference - radius).reshape(-1), (reference + radius).reshape(-1))


def evaluate_objective(x: np.ndarray, context: SpanOptimizationContext) -> dict[str, Any]:
    knots = _vector_to_knots(x, context)
    anchor_traj = interpolate_knots(knots, context.n_frames)
    rig_results = rig_forward_batch(anchor_traj, context.jaw_traj, context.face_traj, context.rig_forward)

    data_term = compute_data_loss(anchor_traj, context.wavlm_traj, context.config.anchor_weights)
    contact_term, min_tip_distance = compute_contact_loss(
        anchor_traj=anchor_traj,
        rig_results=rig_results,
        phoneme_class=context.span.phoneme_class,
        contact_window_mask=context.contact_window_mask,
        alveolar_region=context.alveolar_region,
        interdental_region=context.interdental_region,
        tau_alveolar=context.config.tau_alveolar,
        tau_interdental=context.config.tau_interdental,
    )
    smooth_term = compute_smooth_loss(knots)
    prior_term = compute_prior_loss(knots, context.reference_knots)
    compat_term = compute_compat_loss(anchor_traj, rig_results, context.mouth_box, context.forbidden_planes)

    weighted_losses = {
        "data": context.weights.lambda_data * data_term,
        "contact": context.weights.contact_lambda(context.span.phoneme_class) * contact_term,
        "smooth": context.weights.lambda_smooth * smooth_term,
        "prior": context.weights.lambda_prior * prior_term,
        "compat": context.weights.lambda_compat * compat_term,
    }
    weighted_losses["total"] = float(sum(weighted_losses.values()))
    anchor_delta = knots - context.reference_knots

    return {
        "knots": knots,
        "anchor_traj": anchor_traj,
        "rig_results": rig_results,
        "losses": weighted_losses,
        "raw_losses": {
            "data": data_term,
            "contact": contact_term,
            "smooth": smooth_term,
            "prior": prior_term,
            "compat": compat_term,
        },
        "min_tip_distance": min_tip_distance,
        "delta_norm": float(np.linalg.norm(anchor_delta.reshape(-1))),
        "anchor_delta_magnitudes": tuple(float(np.linalg.norm(anchor_delta[a])) for a in range(4)),
    }


def objective(x: np.ndarray, context: SpanOptimizationContext) -> float:
    return float(evaluate_objective(x, context)["losses"]["total"])


def optimize_span(context: SpanOptimizationContext) -> SpanOptimizationResult:
    x0 = _knots_to_vector(context.reference_knots, context)
    bounds = context.bounds if context.bounds is not None else make_bounds(context.reference_knots, context.config)
    initial_eval = evaluate_objective(x0, context)
    logger.info(
        "Span %s [%d:%d] initial losses: %s",
        context.span.label,
        context.span.start_frame,
        context.span.end_frame,
        initial_eval["losses"],
    )
    result = minimize(
        objective,
        x0,
        args=(context,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": context.config.maxiter},
    )
    final_eval = evaluate_objective(result.x, context)
    logger.info(
        "Span %s [%d:%d] final losses: %s",
        context.span.label,
        context.span.start_frame,
        context.span.end_frame,
        final_eval["losses"],
    )
    return SpanOptimizationResult(
        span=context.span,
        success=bool(result.success),
        status=int(result.status),
        n_iters=int(getattr(result, "nit", 0)),
        initial_losses=initial_eval["losses"],
        final_losses=final_eval["losses"],
        initial_tip_distance=initial_eval["min_tip_distance"],
        final_tip_distance=final_eval["min_tip_distance"],
        delta_norm=final_eval["delta_norm"],
        optimized_knots=final_eval["knots"],
        optimized_traj=final_eval["anchor_traj"],
        message=str(result.message),
        anchor_delta_magnitudes=final_eval["anchor_delta_magnitudes"],
    )


def span_positions_from_frames(span: PhonemeSpan, frames: Sequence[int]) -> np.ndarray:
    frames_arr = np.asarray(frames, dtype=np.int32)
    return np.flatnonzero((frames_arr >= span.start_frame) & (frames_arr <= span.end_frame))


def optimize_utterance(
    frames: Sequence[int],
    wavlm_anchor_pred: np.ndarray,
    jaw_controls: Sequence[Any],
    face_controls: Sequence[Any],
    phoneme_spans: Sequence[PhonemeSpan],
    rig_forward: RigForwardProtocol,
    config: Optional[OptimizationConfig] = None,
    weights: Optional[OptimizationWeights] = None,
    alveolar_region: Optional[RegionRect3D] = None,
    interdental_region: Optional[RegionRect3D] = None,
    mouth_box: Optional[MouthBox] = None,
    forbidden_planes: Sequence[Plane] = (),
    target_phoneme_classes: Optional[Collection[str]] = None,
) -> tuple[np.ndarray, list[SpanOptimizationResult]]:
    cfg = config or OptimizationConfig()
    w = weights or OptimizationWeights()
    target_classes = frozenset(
        DEFAULT_TARGET_PHONEME_CLASSES if target_phoneme_classes is None else target_phoneme_classes
    )
    frames_arr = np.asarray(frames, dtype=np.int32)
    wavlm = np.asarray(wavlm_anchor_pred, dtype=np.float64)
    if wavlm.shape != (len(frames_arr), 4, 3):
        raise ValueError("wavlm_anchor_pred must have shape (T, 4, 3)")
    if len(jaw_controls) != len(frames_arr) or len(face_controls) != len(frames_arr):
        raise ValueError("jaw_controls and face_controls must match frame count")

    results: list[SpanOptimizationResult] = []
    for span in phoneme_spans:
        if span.phoneme_class not in target_classes:
            continue
        positions = span_positions_from_frames(span, frames_arr)
        if len(positions) == 0:
            continue
        context = SpanOptimizationContext(
            span=span,
            frame_positions=positions,
            wavlm_traj=wavlm[positions],
            jaw_traj=[jaw_controls[p] for p in positions],
            face_traj=[face_controls[p] for p in positions],
            rig_forward=rig_forward,
            config=cfg,
            weights=w,
            alveolar_region=alveolar_region,
            interdental_region=interdental_region,
            mouth_box=mouth_box,
            forbidden_planes=tuple(forbidden_planes),
        )
        results.append(optimize_span(context))

    stitched = stitch_span_results(results, wavlm, cfg.seam_blend_frames)
    return stitched, results


def stitch_span_results(
    span_results: Sequence[SpanOptimizationResult],
    baseline_anchor_traj: np.ndarray,
    seam_blend_frames: int = 2,
) -> np.ndarray:
    baseline = np.asarray(baseline_anchor_traj, dtype=np.float64)
    out = baseline.copy()
    if baseline.ndim != 3 or baseline.shape[1:] != (4, 3):
        raise ValueError("baseline_anchor_traj must have shape (T, 4, 3)")
    ordered = sorted(span_results, key=lambda item: item.span.start_frame)
    for result in ordered:
        start = max(0, result.span.start_frame)
        end = min(len(out) - 1, result.span.end_frame)
        expected = end - start + 1
        if expected <= 0:
            continue
        out[start : end + 1] = result.optimized_traj[:expected]

    if seam_blend_frames <= 0:
        return out

    for result in ordered:
        start = max(0, result.span.start_frame)
        end = min(len(out) - 1, result.span.end_frame)
        expected = end - start + 1
        if expected <= 0:
            continue
        edge_count = min(seam_blend_frames, expected)
        for offset in range(edge_count):
            alpha = float(offset + 1) / float(edge_count + 1)

            left_idx = start + offset
            out[left_idx] = (1.0 - alpha) * baseline[left_idx] + alpha * out[left_idx]

            right_idx = end - offset
            out[right_idx] = (1.0 - alpha) * baseline[right_idx] + alpha * out[right_idx]
    return out


def export_debug_report(
    span_results: Sequence[SpanOptimizationResult],
    json_output_path: str | Path,
    summary_csv_path: str | Path | None = None,
    tip_output_path: str | Path | None = None,
    baseline_anchor_traj: Optional[np.ndarray] = None,
    optimized_anchor_traj: Optional[np.ndarray] = None,
) -> None:
    json_path = Path(json_output_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"spans": [item.to_debug_dict() for item in span_results]}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if summary_csv_path is not None:
        csv_path = Path(summary_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "phoneme_label",
                    "phoneme_class",
                    "start_frame",
                    "end_frame",
                    "success",
                    "initial_total_loss",
                    "final_total_loss",
                    "initial_tip_distance",
                    "final_tip_distance",
                    "delta_norm",
                ],
            )
            writer.writeheader()
            for item in span_results:
                if item.span.phoneme_class not in {"alveolar", "interdental"}:
                    continue
                writer.writerow(
                    {
                        "phoneme_label": item.span.label,
                        "phoneme_class": item.span.phoneme_class,
                        "start_frame": item.span.start_frame,
                        "end_frame": item.span.end_frame,
                        "success": int(item.success),
                        "initial_total_loss": item.initial_losses.get("total"),
                        "final_total_loss": item.final_losses.get("total"),
                        "initial_tip_distance": item.initial_tip_distance,
                        "final_tip_distance": item.final_tip_distance,
                        "delta_norm": item.delta_norm,
                    }
                )

    if tip_output_path is not None and baseline_anchor_traj is not None and optimized_anchor_traj is not None:
        tip_path = Path(tip_output_path)
        tip_path.parent.mkdir(parents=True, exist_ok=True)
        baseline = np.asarray(baseline_anchor_traj, dtype=np.float64)
        optimized = np.asarray(optimized_anchor_traj, dtype=np.float64)
        baseline_tip = baseline[:, TIP_ANCHOR_IDX, :]
        optimized_tip = optimized[:, TIP_ANCHOR_IDX, :]
        if tip_path.suffix.lower() == ".json":
            tip_payload = [
                {
                    "frame": i,
                    "baseline_tip": baseline_tip[i].tolist(),
                    "optimized_tip": optimized_tip[i].tolist(),
                }
                for i in range(min(len(baseline_tip), len(optimized_tip)))
            ]
            tip_path.write_text(json.dumps(tip_payload, indent=2), encoding="utf-8")
        else:
            with tip_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "frame",
                        "baseline_tip_x",
                        "baseline_tip_y",
                        "baseline_tip_z",
                        "optimized_tip_x",
                        "optimized_tip_y",
                        "optimized_tip_z",
                    ]
                )
                for i in range(min(len(baseline_tip), len(optimized_tip))):
                    writer.writerow([i, *baseline_tip[i].tolist(), *optimized_tip[i].tolist()])


class FaceKitRigForwardAdapter:
    """
    Thin adapter around the current FaceKit tongue rig.

    The current tongue rig only deforms the tongue mesh from anchors, so jaw/face
    controls are accepted for interface compatibility and only used when an
    optional face mesh reconstruction is requested.
    """

    def __init__(
        self,
        tongue_rig: Any,
        face_model: Any | None = None,
        include_face_mesh: bool = False,
    ) -> None:
        self.tongue_rig = tongue_rig
        self.face_model = face_model
        self.include_face_mesh = include_face_mesh

    def _controls_to_weights(self, jaw_controls_t: Any, face_controls_t: Any) -> dict[str, float]:
        weights: dict[str, float] = {}
        if isinstance(face_controls_t, Mapping):
            weights.update({str(k): float(v) for k, v in face_controls_t.items()})
        elif self.face_model is not None and face_controls_t is not None:
            face_arr = np.asarray(face_controls_t, dtype=np.float64)
            if face_arr.ndim == 1 and len(face_arr) == len(self.face_model.expression_names):
                weights.update(
                    {
                        name: float(val)
                        for name, val in zip(self.face_model.expression_names, face_arr)
                        if float(val) != 0.0
                    }
                )
        if isinstance(jaw_controls_t, Mapping):
            weights.update({str(k): float(v) for k, v in jaw_controls_t.items()})
        elif self.face_model is not None and np.isscalar(jaw_controls_t):
            if "jawOpen" in getattr(self.face_model, "expression_names", []):
                weights["jawOpen"] = float(jaw_controls_t)
        return weights

    def forward_frame(
        self,
        anchor_positions: np.ndarray,
        jaw_controls_t: Any,
        face_controls_t: Any,
    ) -> RigFrameResult:
        anchors = np.asarray(anchor_positions, dtype=np.float64)
        tongue_mesh, _, _ = self.tongue_rig.deform(anchors)
        keypoints = {
            "back": anchors[BACK_ANCHOR_IDX].copy(),
            "dorsum": anchors[DORSUM_ANCHOR_IDX].copy(),
            "blade": anchors[BLADE_ANCHOR_IDX].copy(),
            "tip": anchors[TIP_ANCHOR_IDX].copy(),
        }
        face_mesh = None
        if self.include_face_mesh and self.face_model is not None:
            weights = self._controls_to_weights(jaw_controls_t, face_controls_t)
            face_mesh = self.face_model.deform(weights).copy()
            if hasattr(self.tongue_rig, "global_indices"):
                face_mesh[self.tongue_rig.global_indices] = tongue_mesh
        return RigFrameResult(tongue_mesh=tongue_mesh, tongue_keypoints=keypoints, face_mesh=face_mesh)


__all__ = [
    "ALVEOLAR_PHONEMES",
    "BACK_ANCHOR_IDX",
    "BLADE_ANCHOR_IDX",
    "DEFAULT_TARGET_PHONEME_CLASSES",
    "DORSUM_ANCHOR_IDX",
    "FaceKitRigForwardAdapter",
    "INTERDENTAL_PHONEMES",
    "MouthBox",
    "OptimizationConfig",
    "OptimizationWeights",
    "PhonemeSpan",
    "Plane",
    "RegionRect3D",
    "RigFrameResult",
    "RigForwardProtocol",
    "SpanOptimizationContext",
    "SpanOptimizationResult",
    "TIP_ANCHOR_IDX",
    "build_contact_window_mask",
    "build_knots_from_traj",
    "compute_compat_loss",
    "compute_contact_loss",
    "compute_data_loss",
    "compute_prior_loss",
    "compute_smooth_loss",
    "distance_point_to_region",
    "export_debug_report",
    "hinge_sq",
    "interpolate_knots",
    "make_bounds",
    "map_phoneme_to_class",
    "normalize_phone",
    "objective",
    "optimize_span",
    "optimize_utterance",
    "parse_textgrid",
    "point_inside_mouth_box",
    "rig_forward_batch",
    "signed_distance_to_plane",
    "span_positions_from_frames",
    "stitch_span_results",
]
