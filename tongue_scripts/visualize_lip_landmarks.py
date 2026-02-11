#!/usr/bin/env python3
"""
Visualize lip landmark indices on the ICT face mesh.
Outputs a PNG with upper/lower lip points highlighted using the same
placement style as test.py (pyrender camera + optional cutout).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
import pyrender
import trimesh

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from face_model_io_trimesh import load_face_model_trimesh
    from test import process_beat_data
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT))
    from tongue_scripts.face_model_io_trimesh import load_face_model_trimesh
    from tongue_scripts.test import process_beat_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ICT face with highlighted lip landmark points."
    )
    parser.add_argument(
        "--face-model-dir",
        default=str(PROJECT_ROOT / "FaceXModel"),
        help="Face model directory",
    )
    parser.add_argument(
        "--dataset-id",
        default="1_wayne_0_75_75",
        help="BEAT clip id for deformation",
    )
    parser.add_argument(
        "--beat-root",
        default=str(
            PROJECT_ROOT
            / "ADFA_EVALUATION"
            / "data"
            / "beat_cache_speaker1"
            / "beat_english_v0.2.1"
            / "beat_english_v0.2.1"
            / "1"
        ),
        help="Root folder containing BEAT JSON/TextGrid files",
    )
    parser.add_argument(
        "--analysis-fps",
        type=float,
        default=50.0,
        help="Target FPS for BEAT resampling",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="Time (s) to render deformed face; omit for neutral",
    )
    parser.add_argument(
        "--lip-mode",
        choices=["center", "inner-avg", "outer-avg"],
        default="inner-avg",
        help="Lip point group mode",
    )
    parser.add_argument(
        "--upper-idx",
        type=int,
        default=None,
        help="Override upper-lip landmark index (0-67)",
    )
    parser.add_argument(
        "--lower-idx",
        type=int,
        default=None,
        help="Override lower-lip landmark index (0-67)",
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "outputs" / "lip_landmarks.png"),
        help="Output PNG path",
    )
    parser.add_argument(
        "--marker-radius",
        type=float,
        default=0.3,
        help="Sphere radius for landmark markers",
    )
    parser.add_argument(
        "--cutout",
        action="store_true",
        default=True,
        help="Render a sagittal cut view like test.py (default: enabled)",
    )
    parser.add_argument(
        "--no-cutout",
        dest="cutout",
        action="store_false",
        help="Disable sagittal cut view",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=(900, 700),
        metavar=("W", "H"),
        help="Output image size in pixels",
    )
    return parser.parse_args()


def load_landmark_indices(face_model_dir: Path) -> List[int]:
    config_path = face_model_dir / "vertex_indices.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing vertex_indices.json at {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    indices = config.get("idx_to_landmark_verts")
    if not indices or len(indices) != 68:
        raise ValueError("Expected 68 landmark indices in vertex_indices.json")
    return [int(v) for v in indices]


def select_lip_groups(mode: str) -> Tuple[List[int], List[int]]:
    if mode == "center":
        return [50], [56]
    if mode == "outer-avg":
        upper = list(range(48, 55))
        lower = list(range(54, 60))
        return upper, lower
    upper = list(range(60, 64))
    lower = list(range(64, 68))
    return upper, lower


def build_sphere_cloud(points: np.ndarray, radius: float) -> trimesh.Trimesh:
    base = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    verts = []
    faces = []
    v_offset = 0
    for point in points:
        verts.append(base.vertices + point)
        faces.append(base.faces + v_offset)
        v_offset += len(base.vertices)
    if not verts:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int), process=False)
    return trimesh.Trimesh(vertices=np.vstack(verts), faces=np.vstack(faces), process=False)


def render_scene(
    verts: np.ndarray,
    faces: np.ndarray,
    landmark_points: np.ndarray,
    upper_points: np.ndarray,
    lower_points: np.ndarray,
    marker_radius: float,
    output_path: Path,
    cutout: bool,
    image_size: Tuple[int, int],
) -> None:
    width, height = image_size
    renderer = pyrender.OffscreenRenderer(width, height)

    if cutout:
        eye = np.array([20, -2, 4], dtype=np.float32)
        target = np.array([0, -3, 2], dtype=np.float32)
    else:
        eye = np.array([0, 0, 35], dtype=np.float32)
        target = np.array([0, -2, 0], dtype=np.float32)

    up = np.array([0, 1, 0], dtype=np.float32)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = np.column_stack((x, y, z))
    cam_pose[:3, 3] = eye

    scene = pyrender.Scene(bg_color=[0, 0, 0])
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=cam_pose)

    spot_pose = cam_pose.copy()
    spot_pose[:3, 3] += [0, 10, -5]
    spot_light = pyrender.SpotLight(
        color=np.ones(3), intensity=100, innerConeAngle=np.pi / 8, outerConeAngle=np.pi / 4
    )
    fill_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=400)
    scene.add(spot_light, pose=spot_pose)
    scene.add(fill_light, pose=cam_pose)

    current_faces = faces
    if cutout:
        valid_mask = verts[:, 0] < 0.1
        valid_faces_mask = valid_mask[current_faces].all(axis=1)
        current_faces = current_faces[valid_faces_mask]

    tm = trimesh.Trimesh(verts, current_faces, process=False)
    mat_skin = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.5, 0.5, 0.5, 1.0], metallicFactor=0.0, roughnessFactor=0.8, alphaMode="OPAQUE"
    )
    mesh = pyrender.Mesh.from_trimesh(tm, material=mat_skin, smooth=True)
    if mesh.primitives:
        for prim in mesh.primitives:
            prim.material.doubleSided = True
    scene.add(mesh)

    def add_markers(points: np.ndarray, color: Tuple[float, float, float, float], radius: float) -> None:
        if points.size == 0:
            return
        marker_tm = build_sphere_cloud(points, radius)
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=list(color), metallicFactor=0.0, roughnessFactor=0.5, alphaMode="OPAQUE"
        )
        marker_mesh = pyrender.Mesh.from_trimesh(marker_tm, material=material, smooth=True)
        if marker_mesh.primitives:
            for prim in marker_mesh.primitives:
                prim.material.doubleSided = True
        scene.add(marker_mesh)

    add_markers(landmark_points, (0.6, 0.6, 0.6, 1.0), marker_radius * 0.7)
    add_markers(upper_points, (0.85, 0.15, 0.2, 1.0), marker_radius * 1.1)
    add_markers(lower_points, (0.12, 0.45, 0.85, 1.0), marker_radius * 1.1)

    color, _ = renderer.render(scene)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    renderer.delete()


def main() -> None:
    args = parse_args()
    face_model_dir = Path(args.face_model_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    face_model = load_face_model_trimesh(str(face_model_dir))
    verts = face_model.neutral_verts

    if args.time is not None:
        json_path = Path(args.beat_root) / f"{args.dataset_id}.json"
        face_seq = process_beat_data(str(json_path), face_model, target_fps=args.analysis_fps)
        if face_seq.size == 0:
            raise RuntimeError("Empty face sequence from BEAT data")
        frame = int(round(args.time * args.analysis_fps))
        frame = max(0, min(frame, face_seq.shape[0] - 1))
        weights = {
            name: val for name, val in zip(face_model.expression_names, face_seq[frame])
        }
        verts = face_model.deform(weights).copy()

    landmark_indices = load_landmark_indices(face_model_dir)
    upper_group, lower_group = select_lip_groups(args.lip_mode)
    if args.upper_idx is not None:
        upper_group = [args.upper_idx]
    if args.lower_idx is not None:
        lower_group = [args.lower_idx]

    upper_verts_idx = [landmark_indices[i] for i in upper_group]
    lower_verts_idx = [landmark_indices[i] for i in lower_group]
    landmark_verts = verts[landmark_indices]
    upper_points = verts[upper_verts_idx]
    lower_points = verts[lower_verts_idx]

    render_scene(
        verts,
        face_model.faces,
        landmark_verts,
        upper_points,
        lower_points,
        args.marker_radius,
        output_path,
        args.cutout,
        tuple(args.image_size),
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
