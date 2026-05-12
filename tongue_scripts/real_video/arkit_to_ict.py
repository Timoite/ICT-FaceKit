"""Convert SAiD/Apple ARKit coefficients into ICT FaceKit coefficients."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


SAID_ARKIT_NAMES: list[str] = [
    "jawForward",
    "jawLeft",
    "jawRight",
    "jawOpen",
    "mouthClose",
    "mouthFunnel",
    "mouthPucker",
    "mouthLeft",
    "mouthRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "noseSneerLeft",
    "noseSneerRight",
]


@dataclass(frozen=True)
class ConversionReport:
    """Small audit trail for ARKit to ICT conversion."""

    mapped_channels: dict[str, list[str]]
    missing_ict_channels: dict[str, list[str]]
    zero_filled_ict_channels: list[str]


def arkit_name_to_ict_names(arkit_name: str) -> list[str]:
    """Map a SAiD/Apple ARKit channel name to one or more ICT channel names."""

    if arkit_name == "cheekPuff":
        return ["cheekPuff_L", "cheekPuff_R"]

    direct_names = {
        "jawForward",
        "jawLeft",
        "jawOpen",
        "jawRight",
        "mouthClose",
        "mouthFunnel",
        "mouthLeft",
        "mouthPucker",
        "mouthRight",
        "mouthRollLower",
        "mouthRollUpper",
        "mouthShrugLower",
        "mouthShrugUpper",
    }
    if arkit_name in direct_names:
        return [arkit_name]

    if arkit_name.endswith("Left"):
        return [f"{arkit_name[:-4]}_L"]
    if arkit_name.endswith("Right"):
        return [f"{arkit_name[:-5]}_R"]

    return [arkit_name]


def convert_arkit_to_ict(
    arkit_coeffs: np.ndarray,
    arkit_names: Iterable[str],
    ict_names: Iterable[str],
) -> tuple[np.ndarray, ConversionReport]:
    """Convert an ARKit coefficient matrix into ICT FaceKit channel order."""

    coeffs = np.asarray(arkit_coeffs, dtype=np.float32)
    if coeffs.ndim != 2:
        raise ValueError(f"Expected (frames, channels) coeffs, got shape {coeffs.shape}")

    arkit_names = list(arkit_names)
    ict_names = list(ict_names)
    if coeffs.shape[1] != len(arkit_names):
        raise ValueError(
            f"Coefficient width {coeffs.shape[1]} does not match {len(arkit_names)} ARKit names"
        )

    ict_name_to_idx = {name: idx for idx, name in enumerate(ict_names)}
    ict_coeffs = np.zeros((coeffs.shape[0], len(ict_names)), dtype=np.float32)
    mapped_channels: dict[str, list[str]] = {}
    missing_ict_channels: dict[str, list[str]] = {}

    for src_idx, src_name in enumerate(arkit_names):
        targets = arkit_name_to_ict_names(src_name)
        mapped_channels[src_name] = targets
        missing = [name for name in targets if name not in ict_name_to_idx]
        if missing:
            missing_ict_channels[src_name] = missing

        for target_name in targets:
            target_idx = ict_name_to_idx.get(target_name)
            if target_idx is not None:
                ict_coeffs[:, target_idx] = np.maximum(ict_coeffs[:, target_idx], coeffs[:, src_idx])

    zero_filled = [
        name
        for name, idx in ict_name_to_idx.items()
        if not np.any(np.abs(ict_coeffs[:, idx]) > 1e-8)
    ]
    report = ConversionReport(
        mapped_channels=mapped_channels,
        missing_ict_channels=missing_ict_channels,
        zero_filled_ict_channels=zero_filled,
    )
    return ict_coeffs, report


def load_ict_expression_names(face_model_dir: Path | str) -> list[str]:
    """Load the canonical 53 ICT expression names from FaceXModel metadata."""

    config_path = Path(face_model_dir) / "vertex_indices.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    names = config.get("expressions")
    if not names:
        raise ValueError(f"No expressions list found in {config_path}")
    return list(names)


def read_arkit_coeffs_csv(csv_path: Path | str) -> tuple[list[str], np.ndarray]:
    """Read an ARKit coefficient CSV written by this pipeline or SAiD."""

    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty coefficient CSV: {csv_path}") from exc
        rows = [[float(value) for value in row] for row in reader if row]

    names = [name for name in header if name != "time"]
    coeffs = np.asarray(rows, dtype=np.float32)
    if header and header[0] == "time":
        coeffs = coeffs[:, 1:] if coeffs.size else np.zeros((0, len(names)), dtype=np.float32)
    return names, coeffs


def write_arkit_coeffs_csv(
    csv_path: Path | str,
    arkit_names: Iterable[str],
    arkit_coeffs: np.ndarray,
    fps: float | None = None,
) -> None:
    """Write ARKit coefficients as a simple CSV."""

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(arkit_names)
    coeffs = np.asarray(arkit_coeffs, dtype=np.float32)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if fps is None:
            writer.writerow(names)
            writer.writerows(coeffs.tolist())
        else:
            writer.writerow(["time", *names])
            for frame_idx, row in enumerate(coeffs):
                writer.writerow([frame_idx / fps, *row.tolist()])


def write_ict_coeffs_npz(
    npz_path: Path | str,
    ict_names: Iterable[str],
    ict_coeffs: np.ndarray,
    fps: float,
    report: ConversionReport | None = None,
) -> None:
    """Persist ICT coefficients and conversion metadata."""

    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        coeffs=np.asarray(ict_coeffs, dtype=np.float32),
        names=np.asarray(list(ict_names), dtype=object),
        fps=np.float32(fps),
        mapped_channels=json.dumps(report.mapped_channels if report else {}),
        missing_ict_channels=json.dumps(report.missing_ict_channels if report else {}),
        zero_filled_ict_channels=json.dumps(report.zero_filled_ict_channels if report else []),
    )


def write_arkit_motion_json(
    json_path: Path | str,
    arkit_names: Iterable[str],
    arkit_coeffs: np.ndarray,
    fps: float,
    source_video: Path | str | None = None,
) -> None:
    """Write a BEAT-shaped JSON that existing loaders can read during transition."""

    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(arkit_names)
    coeffs = np.asarray(arkit_coeffs, dtype=np.float32)
    frames = [
        {"time": frame_idx / fps, "weights": row.tolist()}
        for frame_idx, row in enumerate(coeffs)
    ]
    payload = {
        "source": "smirk_said_arkit",
        "fps": fps,
        "names": names,
        "frames": frames,
    }
    if source_video is not None:
        payload["source_video"] = str(source_video)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def convert_csv_to_ict_outputs(
    arkit_csv: Path | str,
    face_model_dir: Path | str,
    ict_npz: Path | str,
    motion_json: Path | str,
    fps: float,
    source_video: Path | str | None = None,
) -> tuple[np.ndarray, ConversionReport]:
    """Convert an ARKit CSV to both ICT NPZ and transition JSON outputs."""

    arkit_names, arkit_coeffs = read_arkit_coeffs_csv(arkit_csv)
    ict_names = load_ict_expression_names(face_model_dir)
    ict_coeffs, report = convert_arkit_to_ict(arkit_coeffs, arkit_names, ict_names)
    write_ict_coeffs_npz(ict_npz, ict_names, ict_coeffs, fps=fps, report=report)
    write_arkit_motion_json(motion_json, arkit_names, arkit_coeffs, fps=fps, source_video=source_video)
    return ict_coeffs, report

