#!/usr/bin/env python3
"""
Render a brighter version of the tongue animation video.

Changes from original:
1. Gray background instead of black
2. Increased light intensity
3. Brighter material colors
"""

import sys
import numpy as np
import trimesh
import pyrender
import cv2
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from face_model_io_trimesh import load_face_model_trimesh
from render_face_animation_trimesh import map_beat_to_ict_names, load_animation
from test import process_beat_data, load_ema_motion, FaceKitTongueRig, TONGUE_CONFIG
from scipy.interpolate import interp1d
import subprocess


def resample_ema_motion(ema_seq, source_fps=50, target_fps=25):
    """Resample EMA motion from source_fps to target_fps."""
    n_frames = len(ema_seq)
    duration = n_frames / source_fps
    n_target_frames = int(duration * target_fps)

    x_source = np.linspace(0, duration, n_frames)
    x_target = np.linspace(0, duration, n_target_frames)

    # Reshape for interpolation: (n_frames, 12) -> interpolate -> reshape back
    ema_flat = ema_seq.reshape(n_frames, -1)
    f_interp = interp1d(
        x_source, ema_flat, axis=0, kind="cubic", fill_value="extrapolate"
    )
    ema_resampled = f_interp(x_target)

    return ema_resampled.reshape(n_target_frames, 4, 3)


# Paths
PROJECT_ROOT = SCRIPT_DIR.parent
FACE_MODEL_DIR = str(PROJECT_ROOT / "FaceXModel")
MOTION_PATH = str(SCRIPT_DIR / "outputs" / "1_wayne_0_75_75.npy")
BS_JSON_PATH = str(SCRIPT_DIR / "inputs" / "1_wayne_0_75_75.json")
AUDIO_PATH = str(SCRIPT_DIR / "inputs" / "1_wayne_0_75_75.wav")
STD_PATH = str(SCRIPT_DIR / "normalising_vectors" / "JW13_4points_std.npy")
OUTPUT_VIDEO = str(SCRIPT_DIR / "outputs" / "tongue_hybrid_deformation_bright.mp4")
TEMP_VIDEO = str(SCRIPT_DIR / "outputs" / "temp_bright.mp4")

# Config - Match the original 25fps video timing
FPS = 25
MAX_SECONDS = 7.5  # Match sample.mp4 duration (~7.4s)
TONGUE_SLICE = slice(16611, 17039)
ANCHOR_INDICES = [16661, 16696, 16755, 16758]
BONE_INDICES = [16661, 16757]


def render_bright_video():
    print("=" * 60)
    print("RENDERING BRIGHTER VIDEO")
    print("=" * 60)

    # Load face model
    print("Loading face model...")
    face_model = load_face_model_trimesh(FACE_MODEL_DIR)

    # Load animation data
    print("Loading animation data...")
    face_seq = process_beat_data(BS_JSON_PATH, face_model, target_fps=FPS)

    # Setup tongue rig
    print("Setting up tongue rig...")
    tongue_rig = FaceKitTongueRig(
        face_model.neutral_verts,
        face_model.faces,
        TONGUE_SLICE,
        ANCHOR_INDICES,
        BONE_INDICES,
        TONGUE_CONFIG,
    )

    # Load and resample EMA motion to match target FPS
    print("Loading EMA motion...")
    ema_seq_raw = load_ema_motion(
        MOTION_PATH, STD_PATH, tongue_rig.anchors, TONGUE_CONFIG["std_scalar"]
    )
    print(f"  Raw EMA frames: {len(ema_seq_raw)} at ~50fps")

    # Resample EMA from 50fps to target FPS
    ema_seq = resample_ema_motion(ema_seq_raw, source_fps=50, target_fps=FPS)
    print(f"  Resampled EMA frames: {len(ema_seq)} at {FPS}fps")

    # Setup rendering
    print("Setting up renderer...")
    W, H = 800, 600
    r = pyrender.OffscreenRenderer(W, H)
    video = cv2.VideoWriter(TEMP_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    # Camera setup
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

    # BRIGHTER MATERIALS
    mat_skin = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.7, 0.7, 1.0],  # Brighter gray (was 0.5)
        metallicFactor=0.0,
        roughnessFactor=0.8,
        alphaMode="OPAQUE",
    )

    mat_tongue = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 0.7, 0.7, 1.0],  # Brighter pink (was 0.6)
        metallicFactor=0.0,
        roughnessFactor=0.2,
        alphaMode="OPAQUE",
    )

    mat_gums = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.8, 0.4, 0.4, 1.0],  # Brighter red (was 0.7, 0.2, 0.2)
        metallicFactor=0.0,
        roughnessFactor=0.2,
        alphaMode="OPAQUE",
    )

    # Camera and light setup (will reuse for each frame)
    spot_pose = cam_pose.copy()
    spot_pose[:3, 3] += [0, 10, -5]

    # Increased light intensity
    spot_light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=300,  # Increased from 100
        innerConeAngle=np.pi / 8,
        outerConeAngle=np.pi / 4,
    )
    fill_light = pyrender.PointLight(
        color=[1.0, 1.0, 1.0],
        intensity=800,  # Increased from 400
    )

    # Pre-calculate masks
    is_tongue_vert = np.zeros(len(face_model.neutral_verts), dtype=bool)
    is_tongue_vert[tongue_rig.global_indices] = True

    is_gum_vert = np.zeros(len(face_model.neutral_verts), dtype=bool)
    is_gum_vert[14062:17039] = True
    is_gum_vert[is_tongue_vert] = False

    frames = min(len(ema_seq), len(face_seq), int(MAX_SECONDS * FPS))

    print(f"Rendering {frames} frames...")

    for i in range(frames):
        if i % 25 == 0:
            print(f"  Frame {i}/{frames}...")

        # Deform face
        weights = {
            name: val for name, val in zip(face_model.expression_names, face_seq[i])
        }
        verts = face_model.deform(weights).copy()

        # Deform tongue
        t_verts, _, _ = tongue_rig.deform(ema_seq[i])
        verts[tongue_rig.global_indices] = t_verts

        current_faces = face_model.faces

        # Split geometry
        face_vert_is_tongue = is_tongue_vert[current_faces]
        is_tongue_face = face_vert_is_tongue.all(axis=1)

        face_vert_is_gum = is_gum_vert[current_faces]
        is_gum_face = face_vert_is_gum.all(axis=1)

        is_skin_face = ~(is_tongue_face | is_gum_face)

        faces_tongue = current_faces[is_tongue_face]
        faces_gum = current_faces[is_gum_face]
        faces_skin = current_faces[is_skin_face]

        # Create new scene for this frame
        scene = pyrender.Scene(bg_color=[0.3, 0.3, 0.3])  # Gray background
        scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=cam_pose)
        scene.add(spot_light, pose=spot_pose)
        scene.add(fill_light, pose=cam_pose)
        nodes = []

        # Add meshes
        if len(faces_skin) > 0:
            tm_skin = trimesh.Trimesh(verts, faces_skin, process=False)
            mesh_skin = pyrender.Mesh.from_trimesh(
                tm_skin, material=mat_skin, smooth=True
            )
            for p in mesh_skin.primitives:
                p.material.doubleSided = True
            nodes.append(scene.add(mesh_skin))

        if len(faces_tongue) > 0:
            tm_tongue = trimesh.Trimesh(verts, faces_tongue, process=False)
            mesh_tongue = pyrender.Mesh.from_trimesh(
                tm_tongue, material=mat_tongue, smooth=True
            )
            for p in mesh_tongue.primitives:
                p.material.doubleSided = True
            nodes.append(scene.add(mesh_tongue))

        if len(faces_gum) > 0:
            tm_gum = trimesh.Trimesh(verts, faces_gum, process=False)
            mesh_gum = pyrender.Mesh.from_trimesh(
                tm_gum, material=mat_gums, smooth=True
            )
            for p in mesh_gum.primitives:
                p.material.doubleSided = True
            nodes.append(scene.add(mesh_gum))

        # Render
        color, _ = r.render(scene)
        video.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

        # Cleanup
        for n in nodes:
            scene.remove_node(n)

    video.release()
    r.delete()

    # Merge audio
    print("Merging audio...")
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "quiet",
        "-i",
        TEMP_VIDEO,
        "-i",
        AUDIO_PATH,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        OUTPUT_VIDEO,
    ]
    subprocess.run(cmd)

    # Clean up temp file
    import os

    if os.path.exists(TEMP_VIDEO):
        os.remove(TEMP_VIDEO)

    print(f"✓ Video saved to: {OUTPUT_VIDEO}")
    return OUTPUT_VIDEO


if __name__ == "__main__":
    render_bright_video()
