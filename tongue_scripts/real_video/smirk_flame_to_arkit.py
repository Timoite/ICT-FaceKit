"""Fit SMIRK/FLAME vertex sequences to SAiD's ARKit blendshape basis."""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import lsq_linear

try:
    from tongue_scripts.real_video.arkit_to_ict import (
        SAID_ARKIT_NAMES,
        write_arkit_coeffs_csv,
    )
except ImportError:  # pragma: no cover - direct script fallback
    from arkit_to_ict import SAID_ARKIT_NAMES, write_arkit_coeffs_csv


@dataclass
class ARKitBasis:
    """ARKit blendshape basis expressed as deltas over a FLAME head crop."""

    names: list[str]
    neutral_vertices: np.ndarray
    deltas: np.ndarray
    vertex_indices: np.ndarray | None = None
    person_id: str | None = None
    warnings: list[str] | None = None

    @property
    def num_channels(self) -> int:
        return len(self.names)

    @property
    def num_vertices(self) -> int:
        return int(self.neutral_vertices.shape[0])


@dataclass
class FitDiagnostics:
    """Diagnostics for a SMIRK-to-ARKit coefficient solve."""

    full_head_rmse: np.ndarray
    mouth_region_rmse: np.ndarray
    lip_aperture_correlation: float | None
    failed_frame_mask: np.ndarray
    solver: str
    warnings: list[str]

    def to_json_dict(self) -> dict:
        valid_full = self.full_head_rmse[~self.failed_frame_mask]
        valid_mouth = self.mouth_region_rmse[~self.failed_frame_mask]
        return {
            "solver": self.solver,
            "frame_count": int(len(self.full_head_rmse)),
            "failed_frame_count": int(np.count_nonzero(self.failed_frame_mask)),
            "failed_frame_mask": self.failed_frame_mask.astype(bool).tolist(),
            "full_head_rmse_mean": float(np.nanmean(valid_full)) if valid_full.size else None,
            "full_head_rmse_max": float(np.nanmax(valid_full)) if valid_full.size else None,
            "mouth_region_rmse_mean": float(np.nanmean(valid_mouth)) if valid_mouth.size else None,
            "mouth_region_rmse_max": float(np.nanmax(valid_mouth)) if valid_mouth.size else None,
            "lip_aperture_correlation": self.lip_aperture_correlation,
            "warnings": self.warnings,
        }


def parse_text_list(path: Path | str, value_type=str) -> list:
    """Parse SAiD text lists with one value per line."""

    values: list = []
    with Path(path).open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            values.append(value_type(line))
    return values


def load_said_arkit_names(said_data_dir: Path | str) -> list[str]:
    path = Path(said_data_dir) / "ARKit_blendshapes.txt"
    if path.is_file():
        return parse_text_list(path, str)
    return SAID_ARKIT_NAMES.copy()


def load_flame_head_indices(said_data_dir: Path | str) -> np.ndarray | None:
    path = Path(said_data_dir) / "FLAME_head_idx.txt"
    if not path.is_file():
        return None
    return np.asarray(parse_text_list(path, int), dtype=np.int64)


def _looks_like_residual_dict(value: object) -> bool:
    return isinstance(value, dict) and all(
        isinstance(k, str) and hasattr(v, "shape") for k, v in value.items()
    )


def _coerce_residual_dict(
    payload: object,
    names: Iterable[str],
    person_id: str | None = None,
) -> tuple[dict[str, np.ndarray], str | None]:
    """Accept SAiD's nested VOCA residual pickle or a simple name->delta dict."""

    names = list(names)
    if _looks_like_residual_dict(payload):
        return {name: np.asarray(payload[name], dtype=np.float32) for name in names}, None

    if not isinstance(payload, dict):
        raise ValueError("blendshape_residuals.pickle must contain a dictionary")

    if person_id is None:
        candidate_ids = sorted(str(key) for key in payload.keys())
        if not candidate_ids:
            raise ValueError("blendshape_residuals.pickle contains no person ids")
        person_id = candidate_ids[0]

    if person_id not in payload:
        available = ", ".join(sorted(str(key) for key in payload.keys())[:8])
        raise KeyError(f"SAiD person id {person_id!r} not found. Available examples: {available}")

    residuals = payload[person_id]
    if not _looks_like_residual_dict(residuals):
        raise ValueError(f"Residual entry for {person_id!r} is not a blendshape dictionary")

    missing = [name for name in names if name not in residuals]
    if missing:
        raise KeyError(f"Missing residual channels for {person_id!r}: {missing}")

    return {name: np.asarray(residuals[name], dtype=np.float32) for name in names}, person_id


def load_blendshape_residuals(
    said_data_dir: Path | str,
    names: Iterable[str],
    person_id: str | None = None,
) -> tuple[np.ndarray, str | None]:
    """Load SAiD blendshape residuals as (channels, vertices, 3)."""

    path = Path(said_data_dir) / "blendshape_residuals.pickle"
    if not path.is_file():
        raise FileNotFoundError(f"Missing SAiD residual pickle: {path}")

    with path.open("rb") as f:
        payload = pickle.load(f)

    residuals, selected_person_id = _coerce_residual_dict(payload, names, person_id)
    deltas = np.stack([np.asarray(residuals[name], dtype=np.float32) for name in names], axis=0)
    if deltas.ndim != 3 or deltas.shape[-1] != 3:
        raise ValueError(f"Expected residuals with shape (channels, vertices, 3), got {deltas.shape}")
    return deltas, selected_person_id


def _select_neutral_for_deltas(
    neutral_vertices: np.ndarray,
    delta_vertex_count: int,
    head_indices: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    warnings_out: list[str] = []
    neutral_vertices = np.asarray(neutral_vertices, dtype=np.float32)

    if len(neutral_vertices) == delta_vertex_count:
        return neutral_vertices, None, warnings_out

    if head_indices is not None and len(head_indices) == delta_vertex_count:
        if int(np.max(head_indices)) >= len(neutral_vertices):
            raise ValueError(
                "FLAME_head_idx.txt references vertices outside the SMIRK neutral vertex array"
            )
        return neutral_vertices[head_indices], head_indices, warnings_out

    if len(neutral_vertices) > delta_vertex_count:
        msg = (
            "SAiD residual vertex count did not match FLAME_head_idx.txt; "
            "falling back to the first residual-sized vertex block."
        )
        warnings.warn(msg, RuntimeWarning)
        warnings_out.append(msg)
        return neutral_vertices[:delta_vertex_count], np.arange(delta_vertex_count), warnings_out

    raise ValueError(
        f"Residual basis has {delta_vertex_count} vertices but neutral has only {len(neutral_vertices)}"
    )


def build_basis_from_said(
    said_data_dir: Path | str,
    neutral_vertices: np.ndarray,
    person_id: str | None = None,
) -> ARKitBasis:
    """Build the ARKit basis by adding SAiD residuals to a SMIRK neutral head crop."""

    names = load_said_arkit_names(said_data_dir)
    deltas, selected_person_id = load_blendshape_residuals(said_data_dir, names, person_id)
    head_indices = load_flame_head_indices(said_data_dir)
    neutral_head, vertex_indices, warnings_out = _select_neutral_for_deltas(
        neutral_vertices, deltas.shape[1], head_indices
    )
    return ARKitBasis(
        names=names,
        neutral_vertices=neutral_head.astype(np.float32),
        deltas=deltas.astype(np.float32),
        vertex_indices=vertex_indices,
        person_id=selected_person_id,
        warnings=warnings_out,
    )


def select_basis_vertices(vertices: np.ndarray, basis: ARKitBasis) -> np.ndarray:
    """Select the same head crop from a full SMIRK/FLAME vertex sequence."""

    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.ndim == 2:
        if basis.vertex_indices is None:
            return vertices
        return vertices[basis.vertex_indices]
    if vertices.ndim != 3:
        raise ValueError(f"Expected vertices with shape (frames, vertices, 3), got {vertices.shape}")
    if basis.vertex_indices is None:
        if vertices.shape[1] != basis.num_vertices:
            return vertices[:, : basis.num_vertices]
        return vertices
    return vertices[:, basis.vertex_indices]


def reconstruct_vertices(coeffs: np.ndarray, basis: ARKitBasis) -> np.ndarray:
    """Reconstruct vertices from coefficients and an ARKit basis."""

    coeffs = np.asarray(coeffs, dtype=np.float32)
    return basis.neutral_vertices[None, :, :] + np.einsum("fc,cvd->fvd", coeffs, basis.deltas)


def _flatten_basis_deltas(basis: ARKitBasis) -> np.ndarray:
    return basis.deltas.reshape(basis.num_channels, -1).T.astype(np.float64)


def _solve_framewise(A: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coeffs = np.zeros((targets.shape[0], A.shape[1]), dtype=np.float32)
    failed = np.zeros(targets.shape[0], dtype=bool)

    if not np.any(np.abs(A) > 1e-12):
        return coeffs, np.ones(targets.shape[0], dtype=bool)

    for frame_idx, target in enumerate(targets):
        b = target.reshape(-1).astype(np.float64)
        if not np.all(np.isfinite(b)):
            failed[frame_idx] = True
            continue
        result = lsq_linear(A, b, bounds=(0.0, 1.0), lsmr_tol="auto", verbose=0)
        if not result.success:
            failed[frame_idx] = True
        coeffs[frame_idx] = np.clip(result.x, 0.0, 1.0)
    return coeffs, failed


def _temporal_delta_projection(coeffs: np.ndarray, delta: float) -> np.ndarray:
    if len(coeffs) <= 1 or delta <= 0:
        return coeffs

    out = coeffs.copy()
    for frame_idx in range(1, len(out)):
        out[frame_idx] = np.clip(out[frame_idx], out[frame_idx - 1] - delta, out[frame_idx - 1] + delta)
    for frame_idx in range(len(out) - 2, -1, -1):
        out[frame_idx] = np.clip(out[frame_idx], out[frame_idx + 1] - delta, out[frame_idx + 1] + delta)
    return np.clip(out, 0.0, 1.0)


def _solve_qp_chunk(
    A: np.ndarray,
    targets: np.ndarray,
    temporal_delta: float,
    init_vals: np.ndarray | None = None,
) -> np.ndarray | None:
    """Use SAiD's qpsolvers formulation when qpsolvers/cvxopt are installed."""

    try:
        from qpsolvers import solve_qp
        from scipy import linalg as la
        from scipy import sparse as sp
    except Exception:
        return None

    seq_len = targets.shape[0]
    channels = A.shape[1]
    btb = A.T @ A
    p = la.block_diag(*[btb for _ in range(seq_len)])
    p += np.eye(p.shape[0]) * 1e-8
    q = np.vstack([A.T @ (-target.reshape(-1, 1)) for target in targets]).reshape(-1)

    g = None
    h = None
    if seq_len > 1:
        eye = sp.identity(channels, dtype=float, format="csc")
        dipole_eye = sp.bmat([[eye], [-eye]], format="csc")
        g_offset = sp.csc_matrix((0, channels), dtype=float)
        diag_g = sp.block_diag([dipole_eye for _ in range(seq_len - 1)], format="csc")
        pos_g = sp.block_diag((diag_g, g_offset), format="csc")
        neg_g = sp.block_diag((g_offset, diag_g), format="csc")
        g = pos_g - neg_g
        h = np.full(g.shape[0], temporal_delta, dtype=float)
    lb = np.zeros(channels * seq_len, dtype=float)
    ub = np.ones(channels * seq_len, dtype=float)
    init = None if init_vals is None else init_vals.reshape(-1)

    try:
        sol = solve_qp(P=p, q=q, G=g, h=h, lb=lb, ub=ub, solver="cvxopt", initvals=init)
    except Exception:
        return None
    if sol is None or not np.all(np.isfinite(sol)):
        return None
    return np.clip(sol.reshape(seq_len, channels), 0.0, 1.0).astype(np.float32)


def _infer_mouth_mask(neutral_vertices: np.ndarray) -> np.ndarray:
    """Heuristic lower-face mask used only for diagnostics."""

    vertices = np.asarray(neutral_vertices, dtype=np.float32)
    if len(vertices) < 16:
        return np.ones(len(vertices), dtype=bool)

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    x_center = np.median(x)
    x_span = np.ptp(x) or 1.0
    y_low, y_high = np.quantile(y, [0.20, 0.62])
    z_low, z_high = np.quantile(z, [0.10, 0.72])
    mask = (
        (np.abs(x - x_center) < 0.34 * x_span)
        & (y >= y_low)
        & (y <= y_high)
        & (z >= z_low)
        & (z <= z_high)
    )
    if np.count_nonzero(mask) < 8:
        return np.ones(len(vertices), dtype=bool)
    return mask


def _lip_aperture(vertices: np.ndarray, mouth_mask: np.ndarray) -> np.ndarray:
    mouth = vertices[:, mouth_mask, :]
    if mouth.shape[1] == 0:
        return np.full(vertices.shape[0], np.nan, dtype=np.float32)
    return np.ptp(mouth[:, :, 1], axis=1).astype(np.float32)


def compute_diagnostics(
    target_vertices: np.ndarray,
    reconstructed_vertices: np.ndarray,
    failed_frame_mask: np.ndarray,
    solver: str,
    warnings_out: list[str] | None = None,
) -> FitDiagnostics:
    diff = reconstructed_vertices - target_vertices
    full_rmse = np.sqrt(np.nanmean(diff * diff, axis=(1, 2))).astype(np.float32)

    mouth_mask = _infer_mouth_mask(target_vertices[0])
    mouth_diff = diff[:, mouth_mask, :]
    mouth_rmse = np.sqrt(np.nanmean(mouth_diff * mouth_diff, axis=(1, 2))).astype(np.float32)

    target_ap = _lip_aperture(target_vertices, mouth_mask)
    recon_ap = _lip_aperture(reconstructed_vertices, mouth_mask)
    valid = (
        np.isfinite(target_ap)
        & np.isfinite(recon_ap)
        & ~failed_frame_mask
        & (np.std(target_ap) > 1e-8)
        & (np.std(recon_ap) > 1e-8)
    )
    corr = None
    if np.count_nonzero(valid) >= 2:
        corr = float(np.corrcoef(target_ap[valid], recon_ap[valid])[0, 1])

    return FitDiagnostics(
        full_head_rmse=full_rmse,
        mouth_region_rmse=mouth_rmse,
        lip_aperture_correlation=corr,
        failed_frame_mask=failed_frame_mask.astype(bool),
        solver=solver,
        warnings=warnings_out or [],
    )


def fit_coefficients(
    vertices: np.ndarray,
    basis: ARKitBasis,
    temporal_delta: float = 0.1,
    chunk_size: int = 120,
    prefer_qp: bool = True,
) -> tuple[np.ndarray, FitDiagnostics]:
    """Fit ARKit coefficients for every frame in a SMIRK/FLAME vertex sequence."""

    target_vertices = select_basis_vertices(vertices, basis)
    if target_vertices.shape[1:] != basis.neutral_vertices.shape:
        raise ValueError(
            f"Target vertices {target_vertices.shape[1:]} do not match basis {basis.neutral_vertices.shape}"
        )

    finite_mask = np.all(np.isfinite(target_vertices), axis=(1, 2))
    centered_targets = target_vertices - basis.neutral_vertices[None, :, :]
    A = _flatten_basis_deltas(basis)

    coeffs = np.zeros((target_vertices.shape[0], basis.num_channels), dtype=np.float32)
    failed = ~finite_mask
    solver_used = "scipy_lsq_linear"
    warnings_out = list(basis.warnings or [])

    if prefer_qp and np.any(finite_mask):
        solver_used = "qpsolvers_cvxopt"
        qp_failed = False
        for start in range(0, len(centered_targets), chunk_size):
            stop = min(start + chunk_size, len(centered_targets))
            chunk = centered_targets[start:stop]
            if not np.all(finite_mask[start:stop]):
                qp_failed = True
                break
            solved = _solve_qp_chunk(A, chunk, temporal_delta=temporal_delta)
            if solved is None:
                qp_failed = True
                break
            coeffs[start:stop] = solved
        if qp_failed:
            solver_used = "scipy_lsq_linear_temporal_projection"
            msg = "qpsolvers/cvxopt solve unavailable or failed; used bounded SciPy least-squares with temporal projection."
            warnings_out.append(msg)
            coeffs, frame_failed = _solve_framewise(A, centered_targets)
            failed |= frame_failed
            coeffs = _temporal_delta_projection(coeffs, temporal_delta)
    else:
        coeffs, frame_failed = _solve_framewise(A, centered_targets)
        failed |= frame_failed
        coeffs = _temporal_delta_projection(coeffs, temporal_delta)

    reconstructed = reconstruct_vertices(coeffs, basis)
    diagnostics = compute_diagnostics(
        target_vertices=target_vertices,
        reconstructed_vertices=reconstructed,
        failed_frame_mask=failed,
        solver=solver_used,
        warnings_out=warnings_out,
    )
    return coeffs, diagnostics


def save_diagnostics(path: Path | str, diagnostics: FitDiagnostics, basis: ARKitBasis) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = diagnostics.to_json_dict()
    payload.update(
        {
            "arkit_names": basis.names,
            "basis_person_id": basis.person_id,
            "basis_vertex_count": basis.num_vertices,
        }
    )
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def fit_smirk_vertices_file(
    vertices_npz: Path | str,
    said_data_dir: Path | str,
    coeffs_csv: Path | str,
    diagnostics_json: Path | str,
    temporal_delta: float = 0.1,
    chunk_size: int = 120,
    said_person_id: str | None = None,
    fps: float | None = None,
) -> tuple[np.ndarray, FitDiagnostics, ARKitBasis]:
    """Fit a saved SMIRK vertex NPZ and persist ARKit coefficients + diagnostics."""

    data = np.load(vertices_npz, allow_pickle=True)
    vertices = np.asarray(data["vertices"], dtype=np.float32)
    neutral = np.asarray(data["neutral_vertices"], dtype=np.float32)
    if fps is None and "fps" in data:
        fps = float(data["fps"])

    basis = build_basis_from_said(said_data_dir, neutral, person_id=said_person_id)
    coeffs, diagnostics = fit_coefficients(
        vertices,
        basis,
        temporal_delta=temporal_delta,
        chunk_size=chunk_size,
        prefer_qp=True,
    )
    if "valid_frames" in data:
        valid_frames = np.asarray(data["valid_frames"], dtype=bool)
        if valid_frames.shape[0] == diagnostics.failed_frame_mask.shape[0]:
            diagnostics.failed_frame_mask |= ~valid_frames
    write_arkit_coeffs_csv(coeffs_csv, basis.names, coeffs, fps=fps)
    save_diagnostics(diagnostics_json, diagnostics, basis)
    return coeffs, diagnostics, basis
