#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "scipy>=1.11",
#     "trimesh",
#     "pyrender",
#     "opencv-python",
# ]
# ///
"""
Render two videos for tongue animation comparison:
1. With dynamic tongue (WavLM-driven EMA motion)
2. With passive tongue (generic_neutral_mesh only)

Both videos apply jawOpen offset correction for proper lip closure.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrender
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
TONGUE_SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TONGUE_SCRIPTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tongue_scripts.tongue_animation.face_model_io_trimesh import load_face_model_trimesh
from tongue_scripts.tongue_animation.generate_tongue_animation import (
    load_blendshape_json_sequence,
    load_ema_motion,
    FaceKitTongueRig,
    TONGUE_CONFIG,
)

# ==========================================
# CONFIGURATION
# ==========================================
RENDER_MODE = "FULL_FACE"  # FULL_FACE or MATPLOTLIB
FPS = 25
MAX_SECONDS = None
TONGUE_SHIFT_SECONDS = 0.120  # Positive delay for tongue motion (120ms)

# Dataset configuration
DATASET_ID = "1_wayne_0_75_75"
FACE_MODEL_DIR = str(PROJECT_ROOT / "FaceXModel")

# Try multiple possible BEAT data paths
BEAT_DATA_ROOTS = [
    TONGUE_SCRIPTS_DIR / "inputs",
    PROJECT_ROOT / "data" / "beat_cache" / "beat_english_v0.2.1" / "beat_english_v0.2.1" / "1",
    PROJECT_ROOT / "ADFA_EVALUATION" / "data" / "beat_cache_speaker1" / "beat_english_v0.2.1" / "beat_english_v0.2.1" / "1",
    PROJECT_ROOT / "ADFA_EVALUATION" / "Visual_Speech_Recognition_for_Multiple_Languages" / "data" / "beat_english_v0.2.1" / "1",
]

# Find the first valid path
BEAT_DATA_ROOT = None
for root in BEAT_DATA_ROOTS:
    if (root / f"{DATASET_ID}.json").exists():
        BEAT_DATA_ROOT = root
        break

if BEAT_DATA_ROOT is None:
    # Check for files with .TextGrid extension
    for root in BEAT_DATA_ROOTS:
        if (root / f"{DATASET_ID}.TextGrid").exists():
            BEAT_DATA_ROOT = root
            print(f"  Note: Found .TextGrid file, using path: {root}")
            break

if BEAT_DATA_ROOT is None:
    print("Warning: BEAT data files not found in expected locations.")
    print("Searched paths:")
    for root in BEAT_DATA_ROOTS:
        print(f"  - {root}")
    BEAT_DATA_ROOT = BEAT_DATA_ROOTS[0]  # Fallback to first path

BEAT_JSON_PATH = str(BEAT_DATA_ROOT / f"{DATASET_ID}.json")
AUDIO_PATH = str(BEAT_DATA_ROOT / f"{DATASET_ID}.wav")
MOTION_PATH = str(TONGUE_SCRIPTS_DIR / "outputs" / f"{DATASET_ID}.npy")
STD_PATH = str(TONGUE_SCRIPTS_DIR / "normalising_vectors" / "JW13_4points_std.npy")

# Output directory
OUTPUT_DIR = TONGUE_SCRIPTS_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_VIDEO_WITH_TONGUE = str(OUTPUT_DIR / f"{DATASET_ID}_with_tongue.mp4")
OUTPUT_VIDEO_PASSIVE_TONGUE = str(OUTPUT_DIR / f"{DATASET_ID}_passive_tongue.mp4")

# Tongue rig configuration
TONGUE_SLICE = slice(16611, 17039)
ANCHOR_INDICES = [16661, 16696, 16755, 16758]
BONE_INDICES = [16661, 16757]

# ==========================================
# JAWOPEN OFFSET CORRECTION
# ==========================================
def apply_jawopen_offset_correction(face_seq, face_model):
    """
    Apply jawOpen offset correction to ensure proper lip closure.

    This function shifts the jawOpen blendshape so that its minimum value
    is 0.0 (fully closed). This is a systematic error correction that
    ensures the lips can fully close.

    Args:
        face_seq: (N, M) array of facial expression weights
        face_model: TrimeshFaceModel instance

    Returns:
        Corrected face_seq
    """
    if "jawOpen" in face_model.expression_names:
        jaw_idx = face_model.expression_names.index("jawOpen")
        raw_vals = face_seq[:, jaw_idx]
        min_val = float(np.min(raw_vals))
        if min_val != 0.0:
            print(f"  ★ JawOpen offset detected: min_value={min_val:.4f}")
            print(f"  ★ Applying correction: shifting by {-min_val:.4f} to ensure full closure")
        face_seq[:, jaw_idx] = np.maximum(0.0, raw_vals - min_val)
    else:
        print("  [Warning] 'jawOpen' not found in expression names. Skipping lip correction.")
    return face_seq

def shift_sequence(seq: np.ndarray, shift_frames: int) -> np.ndarray:
    """
    Shift sequence by frames (positive = delay/pad at start).
    """
    if shift_frames == 0:
        return seq
    n = len(seq)
    if shift_frames > 0:
        # Delay: repeat first frame and prepend
        pad = np.repeat(seq[:1], shift_frames, axis=0)
        shifted = np.concatenate([pad, seq], axis=0)[:n]
    else:
        # Advance: repeat last frame and append
        shift_frames = abs(shift_frames)
        pad = np.repeat(seq[-1:], shift_frames, axis=0)
        shifted = np.concatenate([seq[shift_frames:], pad], axis=0)
    return shifted

# ==========================================
# DATA LOADING
# ==========================================
def load_data():
    """Load face model, BEAT blendshapes, and EMA motion."""
    print("Loading face model...")
    face_model = load_face_model_trimesh(FACE_MODEL_DIR)

    print(f"Loading BEAT blendshapes from: {BEAT_JSON_PATH}")
    if not Path(BEAT_JSON_PATH).exists():
        print(f"  Warning: BEAT JSON not found. Using neutral face.")
        print(f"  Searched in: {BEAT_JSON_PATH}")
        face_seq = np.zeros((375, len(face_model.expression_names)), dtype=np.float32)
    else:
        face_seq = load_blendshape_json_sequence(
            BEAT_JSON_PATH,
            face_model,
            source_fps=60,
            target_fps=FPS,
        )

    # Apply jawOpen offset correction
    print("\n" + "="*60)
    print("JAWOPEN OFFSET CORRECTION")
    print("="*60)
    face_seq = apply_jawopen_offset_correction(face_seq, face_model)
    print("="*60 + "\n")

    print("Setting up tongue rig...")
    tongue_rig = FaceKitTongueRig(
        face_model.neutral_verts,
        face_model.faces,
        TONGUE_SLICE,
        ANCHOR_INDICES,
        BONE_INDICES,
        TONGUE_CONFIG,
    )

    print(f"Loading EMA motion from: {MOTION_PATH}")
    if not Path(MOTION_PATH).exists():
        raise FileNotFoundError(f"EMA motion file not found: {MOTION_PATH}")

    ema_seq = load_ema_motion(
        MOTION_PATH,
        STD_PATH,
        tongue_rig.anchors,
        TONGUE_CONFIG["std_scalar"],
    )

    # Resample from 50fps to 25fps if needed
    if len(ema_seq) > len(face_seq):
        from scipy.interpolate import interp1d
        n_frames = len(face_seq)
        duration = len(ema_seq) / 50.0
        x_source = np.linspace(0, duration, len(ema_seq))
        x_target = np.linspace(0, duration, n_frames)
        ema_flat = ema_seq.reshape(len(ema_seq), -1)
        ema_resampled = np.zeros((n_frames, ema_flat.shape[1]))
        for i in range(ema_flat.shape[1]):
            ema_resampled[:, i] = interp1d(x_source, ema_flat[:, i], kind='cubic')(x_target)
        ema_seq = ema_resampled.reshape(n_frames, 4, 3)

    # Apply tongue shift delay (120ms)
    print("\n" + "="*60)
    print("TONGUE SHIFT DELAY")
    print("="*60)
    if TONGUE_SHIFT_SECONDS != 0.0:
        shift_frames = int(round(TONGUE_SHIFT_SECONDS * FPS))
        print(f"  ★ Applying {TONGUE_SHIFT_SECONDS}s delay ({shift_frames} frames @ {FPS}fps)")
        print(f"  ★ This delays tongue motion relative to jaw/expression motion")
        ema_seq = shift_sequence(ema_seq, shift_frames)
    else:
        print("  No tongue shift applied (TONGUE_SHIFT_SECONDS = 0.0)")
    print("="*60 + "\n")

    return face_model, face_seq, tongue_rig, ema_seq

# ==========================================
# RENDERING
# ==========================================
def render_video_with_dynamic_tongue(face_model, face_seq, tongue_rig, ema_seq, output_path, fps=FPS, max_seconds=MAX_SECONDS):
    """Render video with dynamic tongue driven by EMA motion."""
    print(f"Rendering video with DYNAMIC tongue to {output_path}...")

    W, H = 800, 600
    renderer = pyrender.OffscreenRenderer(W, H)
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # Camera setup (front view)
    eye = np.array([0.0, -2.0, 35.0], dtype=np.float32)
    target = np.array([0.0, -2.0, 0.0], dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = np.column_stack((x, y, z))
    cam_pose[:3, 3] = eye

    # Materials
    mat_skin = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.5, 0.5, 0.5, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.8,
        alphaMode="OPAQUE",
    )
    mat_tongue = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 0.6, 0.6, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.2,
        alphaMode="OPAQUE",
    )
    mat_gums = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.2, 0.2, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.2,
        alphaMode="OPAQUE",
    )

    # Scene setup
    scene = pyrender.Scene(bg_color=[0, 0, 0])
    spot_pose = cam_pose.copy()
    spot_pose[:3, 3] += [0, 10, -5]
    spot_light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=100,
        innerConeAngle=np.pi / 8,
        outerConeAngle=np.pi / 4,
    )
    fill_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=400)

    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=cam_pose)
    scene.add(spot_light, pose=spot_pose)
    scene.add(fill_light, pose=cam_pose)

    # Prepare masks
    is_tongue_vert = np.zeros(len(face_model.neutral_verts), dtype=bool)
    is_tongue_vert[tongue_rig.global_indices] = True

    is_gum_vert = np.zeros(len(face_model.neutral_verts), dtype=bool)
    is_gum_vert[14062:17039] = True
    is_gum_vert[is_tongue_vert] = False

    if max_seconds is None:
        frames = min(len(face_seq), len(ema_seq))
    else:
        frames = min(len(face_seq), len(ema_seq), int(max_seconds * fps))

    for i in range(frames):
        if i % max(1, int(round(fps))) == 0:
            print(f"  Frame {i}/{frames}...")

        # Deform face
        weights = {name: val for name, val in zip(face_model.expression_names, face_seq[i])}
        verts = face_model.deform(weights).copy()

        # Apply dynamic tongue deformation
        t_verts, _, _ = tongue_rig.deform(ema_seq[i])
        verts[tongue_rig.global_indices] = t_verts

        # Split geometry
        current_faces = face_model.faces
        face_vert_is_tongue = is_tongue_vert[current_faces]
        is_tongue_face = face_vert_is_tongue.all(axis=1)

        face_vert_is_gum = is_gum_vert[current_faces]
        is_gum_face = face_vert_is_gum.all(axis=1)

        is_skin_face = ~(is_tongue_face | is_gum_face)

        faces_tongue = current_faces[is_tongue_face]
        faces_gum = current_faces[is_gum_face]
        faces_skin = current_faces[is_skin_face]

        nodes = []

        if len(faces_skin) > 0:
            tm_skin = trimesh.Trimesh(verts, faces_skin, process=False)
            mesh_skin = pyrender.Mesh.from_trimesh(tm_skin, material=mat_skin, smooth=True)
            if mesh_skin.primitives:
                for p in mesh_skin.primitives:
                    p.material.doubleSided = True
            nodes.append(scene.add(mesh_skin))

        if len(faces_tongue) > 0:
            tm_tongue = trimesh.Trimesh(verts, faces_tongue, process=False)
            mesh_tongue = pyrender.Mesh.from_trimesh(tm_tongue, material=mat_tongue, smooth=True)
            if mesh_tongue.primitives:
                for p in mesh_tongue.primitives:
                    p.material.doubleSided = True
            nodes.append(scene.add(mesh_tongue))

        if len(faces_gum) > 0:
            tm_gum = trimesh.Trimesh(verts, faces_gum, process=False)
            mesh_gum = pyrender.Mesh.from_trimesh(tm_gum, material=mat_gums, smooth=True)
            if mesh_gum.primitives:
                for p in mesh_gum.primitives:
                    p.material.doubleSided = True
            nodes.append(scene.add(mesh_gum))

        color, _ = renderer.render(scene)
        video.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

        for n in nodes:
            scene.remove_node(n)

    video.release()
    renderer.delete()
    print(f"  ✓ Saved: {output_path}")

def render_video_with_passive_tongue(face_model, face_seq, tongue_rig, output_path, fps=FPS, max_seconds=MAX_SECONDS):
    """Render video with passive tongue from generic_neutral_mesh."""
    print(f"Rendering video with PASSIVE tongue to {output_path}...")

    W, H = 800, 600
    renderer = pyrender.OffscreenRenderer(W, H)
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # Camera setup (front view)
    eye = np.array([0.0, -2.0, 35.0], dtype=np.float32)
    target = np.array([0.0, -2.0, 0.0], dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = np.column_stack((x, y, z))
    cam_pose[:3, 3] = eye

    # Materials
    mat_skin = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.5, 0.5, 0.5, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.8,
        alphaMode="OPAQUE",
    )
    mat_tongue = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 0.6, 0.6, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.2,
        alphaMode="OPAQUE",
    )
    mat_gums = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.2, 0.2, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.2,
        alphaMode="OPAQUE",
    )

    # Scene setup
    scene = pyrender.Scene(bg_color=[0, 0, 0])
    spot_pose = cam_pose.copy()
    spot_pose[:3, 3] += [0, 10, -5]
    spot_light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=100,
        innerConeAngle=np.pi / 8,
        outerConeAngle=np.pi / 4,
    )
    fill_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=400)

    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=cam_pose)
    scene.add(spot_light, pose=spot_pose)
    scene.add(fill_light, pose=cam_pose)

    # Prepare masks
    is_tongue_vert = np.zeros(len(face_model.neutral_verts), dtype=bool)
    is_tongue_vert[tongue_rig.global_indices] = True

    is_gum_vert = np.zeros(len(face_model.neutral_verts), dtype=bool)
    is_gum_vert[14062:17039] = True
    is_gum_vert[is_tongue_vert] = False

    if max_seconds is None:
        frames = len(face_seq)
    else:
        frames = min(len(face_seq), int(max_seconds * fps))

    for i in range(frames):
        if i % max(1, int(round(fps))) == 0:
            print(f"  Frame {i}/{frames}...")

        # Deform face (NO dynamic tongue deformation)
        weights = {name: val for name, val in zip(face_model.expression_names, face_seq[i])}
        verts = face_model.deform(weights).copy()

        # NOTE: We do NOT apply tongue_rig.deform() here
        # The tongue remains in its passive neutral state from generic_neutral_mesh

        # Split geometry
        current_faces = face_model.faces
        face_vert_is_tongue = is_tongue_vert[current_faces]
        is_tongue_face = face_vert_is_tongue.all(axis=1)

        face_vert_is_gum = is_gum_vert[current_faces]
        is_gum_face = face_vert_is_gum.all(axis=1)

        is_skin_face = ~(is_tongue_face | is_gum_face)

        faces_tongue = current_faces[is_tongue_face]
        faces_gum = current_faces[is_gum_face]
        faces_skin = current_faces[is_skin_face]

        nodes = []

        if len(faces_skin) > 0:
            tm_skin = trimesh.Trimesh(verts, faces_skin, process=False)
            mesh_skin = pyrender.Mesh.from_trimesh(tm_skin, material=mat_skin, smooth=True)
            if mesh_skin.primitives:
                for p in mesh_skin.primitives:
                    p.material.doubleSided = True
            nodes.append(scene.add(mesh_skin))

        if len(faces_tongue) > 0:
            tm_tongue = trimesh.Trimesh(verts, faces_tongue, process=False)
            mesh_tongue = pyrender.Mesh.from_trimesh(tm_tongue, material=mat_tongue, smooth=True)
            if mesh_tongue.primitives:
                for p in mesh_tongue.primitives:
                    p.material.doubleSided = True
            nodes.append(scene.add(mesh_tongue))

        if len(faces_gum) > 0:
            tm_gum = trimesh.Trimesh(verts, faces_gum, process=False)
            mesh_gum = pyrender.Mesh.from_trimesh(tm_gum, material=mat_gums, smooth=True)
            if mesh_gum.primitives:
                for p in mesh_gum.primitives:
                    p.material.doubleSided = True
            nodes.append(scene.add(mesh_gum))

        color, _ = renderer.render(scene)
        video.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

        for n in nodes:
            scene.remove_node(n)

    video.release()
    renderer.delete()
    print(f"  ✓ Saved: {output_path}")

def merge_audio(video_path, audio_path, output_path):
    """Merge audio into video using ffmpeg."""
    if not Path(audio_path).exists():
        print(f"  [Warning] Audio not found: {audio_path}")
        return

    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "quiet",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd)
    print(f"  ✓ Merged audio: {output_path}")

# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 80)
    print("DUAL TONGUE ANIMATION COMPARISON RENDERER")
    print("=" * 80)
    print(f"Dataset ID: {DATASET_ID}")
    print(f"Render Mode: {RENDER_MODE}")
    print(f"FPS: {FPS}")
    print(f"Max Seconds: {MAX_SECONDS}")
    print(f"Tongue Shift Delay: {TONGUE_SHIFT_SECONDS}s")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 80 + "\n")

    # Load data
    face_model, face_seq, tongue_rig, ema_seq = load_data()

    # Render video with dynamic tongue
    print("\n" + "=" * 80)
    print("RENDERING VIDEO 1: WITH DYNAMIC TONGUE")
    print("=" * 80)
    render_video_with_dynamic_tongue(
        face_model, face_seq, tongue_rig, ema_seq, OUTPUT_VIDEO_WITH_TONGUE
    )

    # Merge audio
    video_with_audio = OUTPUT_VIDEO_WITH_TONGUE.replace(".mp4", "_with_audio.mp4")
    merge_audio(OUTPUT_VIDEO_WITH_TONGUE, AUDIO_PATH, video_with_audio)

    # Render video with passive tongue
    print("\n" + "=" * 80)
    print("RENDERING VIDEO 2: WITH PASSIVE TONGUE")
    print("=" * 80)
    render_video_with_passive_tongue(face_model, face_seq, tongue_rig, OUTPUT_VIDEO_PASSIVE_TONGUE)

    # Merge audio
    passive_with_audio = OUTPUT_VIDEO_PASSIVE_TONGUE.replace(".mp4", "_with_audio.mp4")
    merge_audio(OUTPUT_VIDEO_PASSIVE_TONGUE, AUDIO_PATH, passive_with_audio)

    print("\n" + "=" * 80)
    print("RENDERING COMPLETE")
    print("=" * 80)
    print(f"✓ Dynamic tongue: {video_with_audio}")
    print(f"✓ Passive tongue: {passive_with_audio}")
    print("=" * 80)

if __name__ == "__main__":
    main()
