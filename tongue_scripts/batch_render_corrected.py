#!/usr/bin/env python3
"""
Batch render corrected videos with mouth bias removed.
"""

import sys
import numpy as np
import trimesh
import pyrender
import cv2
from pathlib import Path
import subprocess

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from face_model_io_trimesh import load_face_model_trimesh
from render_face_animation_trimesh import map_beat_to_ict_names, load_animation
from test import process_beat_data, load_ema_motion, FaceKitTongueRig, TONGUE_CONFIG
from scipy.interpolate import interp1d


def resample_ema_motion(ema_seq, source_fps=50, target_fps=25):
    """Resample EMA motion from source_fps to target_fps."""
    n_frames = len(ema_seq)
    duration = n_frames / source_fps
    n_target_frames = int(duration * target_fps)

    x_source = np.linspace(0, duration, n_frames)
    x_target = np.linspace(0, duration, n_target_frames)

    ema_flat = ema_seq.reshape(n_frames, -1)
    f_interp = interp1d(
        x_source, ema_flat, axis=0, kind="cubic", fill_value="extrapolate"
    )
    ema_resampled = f_interp(x_target)

    return ema_resampled.reshape(n_target_frames, 4, 3)


# Paths
PROJECT_ROOT = SCRIPT_DIR.parent
FACE_MODEL_DIR = str(PROJECT_ROOT / "FaceXModel")
STD_PATH = str(SCRIPT_DIR / "normalising_vectors" / "JW13_4points_std.npy")

# Config
FPS = 25
MAX_SECONDS = 7.5
TONGUE_SLICE = slice(16611, 17039)
ANCHOR_INDICES = [16661, 16696, 16755, 16758]
BONE_INDICES = [16661, 16757]


def render_bright_video(dataset_id, wav_path=None, json_path=None, npy_path=None, output_path=None):
    """Render video for a specific dataset with provided paths."""
    print("=" * 80)
    print(f"RENDERING: {dataset_id}")
    print("=" * 80)

    # Use provided paths or fallback (but we expect provided paths now)
    MOTION_PATH = str(npy_path) if npy_path else str(SCRIPT_DIR / "outputs" / f"{dataset_id}.npy")
    BS_JSON_PATH = str(json_path) if json_path else str(SCRIPT_DIR / "inputs" / f"{dataset_id}.json")
    AUDIO_PATH = str(wav_path) if wav_path else str(SCRIPT_DIR / "inputs" / f"{dataset_id}.wav")
    OUTPUT_VIDEO = str(output_path) if output_path else str(SCRIPT_DIR / "batch_videos" / f"{dataset_id}_corrected_7.5s.mp4")
    TEMP_VIDEO = str(Path(OUTPUT_VIDEO).parent / f"{dataset_id}_temp.mp4")
    
    Path(OUTPUT_VIDEO).parent.mkdir(parents=True, exist_ok=True)

    if not Path(MOTION_PATH).exists():
        print(f"❌ EMA file not found: {MOTION_PATH}")
        return None
    if not Path(BS_JSON_PATH).exists():
        print(f"❌ JSON file not found: {BS_JSON_PATH}")
        return None
    if not Path(AUDIO_PATH).exists():
        print(f"❌ Audio file not found: {AUDIO_PATH}")
        return None

    print(f"Dataset: {dataset_id}")
    print(f"Motion: {MOTION_PATH}")
    print(f"Blendshapes: {BS_JSON_PATH}")
    print(f"Audio: {AUDIO_PATH}")
    print(f"Output: {OUTPUT_VIDEO}")

    face_model = load_face_model_trimesh(FACE_MODEL_DIR)

    face_seq = process_beat_data(BS_JSON_PATH, face_model, target_fps=FPS)

    # --- FIX: LIP CLOSURE (Systematic Error Correction) ---
    # Shift 'jawOpen' so that its minimum value is 0.0 (fully closed)
    if 'jawOpen' in face_model.expression_names:
        idx = face_model.expression_names.index('jawOpen')
        raw_vals = face_seq[:, idx]
        min_val = np.min(raw_vals)
        print(f"  [Correction] jawOpen min value: {min_val:.4f}. shifting by {-min_val:.4f} to 0.0")
        face_seq[:, idx] = np.maximum(0, raw_vals - min_val)
    else:
        print("  [Warning] 'jawOpen' not found in expression names. Skipping lip correction.")


    tongue_rig = FaceKitTongueRig(
        face_model.neutral_verts,
        face_model.faces,
        TONGUE_SLICE,
        ANCHOR_INDICES,
        BONE_INDICES,
        TONGUE_CONFIG,
    )

    ema_seq_raw = load_ema_motion(
        MOTION_PATH, STD_PATH, tongue_rig.anchors, TONGUE_CONFIG["std_scalar"]
    )
    print(f"  Raw EMA frames: {len(ema_seq_raw)} at ~50fps")

    ema_seq = resample_ema_motion(ema_seq_raw, source_fps=50, target_fps=FPS)
    print(f"  Resampled EMA frames: {len(ema_seq)} at {FPS}fps")

    W, H = 800, 600
    r = pyrender.OffscreenRenderer(W, H)
    video = cv2.VideoWriter(TEMP_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    # --- VISUALS: Moving Camera & Humane Skin ---
    def get_camera_matrix(eye, target, up=None):
        if up is None: up = np.array([0, 1, 0], dtype=np.float32)
        z = eye - target; z /= np.linalg.norm(z)
        x = np.cross(up, z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        mat = np.eye(4)
        mat[:3, :3] = np.column_stack((x, y, z))
        mat[:3, 3] = eye
        return mat

    # Materials (Humane)
    mat_skin = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.82, 0.65, 0.55, 1.0], # Humane
        metallicFactor=0.0,
        roughnessFactor=0.65,
        alphaMode="OPAQUE",
    )
    mat_tongue = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 0.7, 0.7, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.2,
        alphaMode="OPAQUE",
    )
    mat_gums = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.8, 0.4, 0.4, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.2,
        alphaMode="OPAQUE",
    )

    spot_light = pyrender.SpotLight(
        color=np.ones(3), intensity=300, innerConeAngle=np.pi / 8, outerConeAngle=np.pi / 4,
    )
    fill_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=800)

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

        weights = {
            name: val for name, val in zip(face_model.expression_names, face_seq[i])
        }
        # NOTE: Using original logic (no manual bias added)
        
        verts = face_model.deform(weights).copy()

        t_verts, _, _ = tongue_rig.deform(ema_seq[i])
        verts[tongue_rig.global_indices] = t_verts

        current_faces = face_model.faces

        face_vert_is_tongue = is_tongue_vert[current_faces]
        is_tongue_face = face_vert_is_tongue.all(axis=1)
        face_vert_is_gum = is_gum_vert[current_faces]
        is_gum_face = face_vert_is_gum.all(axis=1)
        is_skin_face = ~(is_tongue_face | is_gum_face)

        faces_tongue = current_faces[is_tongue_face]
        faces_gum = current_faces[is_gum_face]
        faces_skin = current_faces[is_skin_face]

        # Moving Camera Logic
        t = i / 50.0 
        angle = np.radians(25 * np.sin(t * np.pi))
        orbit_r = 35.0
        ex = np.sin(angle) * orbit_r
        ez = np.cos(angle) * orbit_r
        cam_pose = get_camera_matrix(np.array([ex, 0, ez], dtype=np.float32), 
                                     np.array([0, -2, 0], dtype=np.float32))
        
        spot_pose = cam_pose.copy()
        spot_pose[:3, 3] += [0, 10, -5]

        scene = pyrender.Scene(bg_color=[0.3, 0.3, 0.3])
        scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=cam_pose)
        scene.add(spot_light, pose=spot_pose)
        scene.add(fill_light, pose=cam_pose)
        nodes = []

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

        color, _ = r.render(scene)
        video.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

        for n in nodes:
            scene.remove_node(n)

    video.release()
    r.delete()

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

    if Path(TEMP_VIDEO).exists():
        Path(TEMP_VIDEO).unlink()

    print(f"✓ Video saved: {OUTPUT_VIDEO}")
    return OUTPUT_VIDEO


def main():
    datasets = [
        "1_wayne_0_75_75",
        "1_wayne_0_100_100",
        "1_wayne_0_10_10",
        "1_wayne_0_101_101",
    ]

    results = {}
    for dataset_id in datasets:
        try:
            output_path = render_bright_video(dataset_id)
            if output_path:
                results[dataset_id] = output_path
        except Exception as e:
            print(f"❌ Error rendering {dataset_id}: {e}")
            results[dataset_id] = None

    print("\n" + "=" * 80)
    print("BATCH RENDERING COMPLETE")
    print("=" * 80)
    for dataset_id, path in results.items():
        if path:
            print(f"✓ {dataset_id}: {path}")
        else:
            print(f"✗ {dataset_id}: FAILED")


if __name__ == "__main__":
    main()
