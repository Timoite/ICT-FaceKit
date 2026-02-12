#!/usr/bin/env python3
"""
Visualization Setup Script
--------------------------
Generates sample images and short videos to visualize:
1. Different Camera Angles
2. Different Lighting Setups
3. Moving Camera Trajectory
4. "Human-like" skin material (User Request)

Outputs will be saved to 'tongue_scripts/vis_output/'
"""

import sys
import numpy as np
import trimesh
import pyrender
import cv2
import os
from pathlib import Path
from scipy.interpolate import interp1d

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
TONGUE_SCRIPTS_DIR = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TONGUE_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(TONGUE_SCRIPTS_DIR))

from face_model_io_trimesh import load_face_model_trimesh
from generate_tongue_animation import (
    process_beat_data,
    load_ema_motion,
    FaceKitTongueRig,
    TONGUE_CONFIG,
)

# Paths
FACE_MODEL_DIR = str(PROJECT_ROOT / "FaceXModel")
MOTION_PATH = str(SCRIPT_DIR / "outputs" / "1_wayne_0_75_75.npy")
BS_JSON_PATH = str(SCRIPT_DIR / "inputs" / "1_wayne_0_75_75.json")
STD_PATH = str(SCRIPT_DIR / "normalising_vectors" / "JW13_4points_std.npy")
OUTPUT_DIR = SCRIPT_DIR / "vis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Config
FPS = 25
TONGUE_SLICE = slice(16611, 17039)
ANCHOR_INDICES = [16661, 16696, 16755, 16758]
BONE_INDICES = [16661, 16757]

def get_camera_matrix(eye, target, up=None):
    if up is None:
        up = np.array([0, 1, 0], dtype=np.float32)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    mat = np.eye(4)
    mat[:3, :3] = np.column_stack((x, y, z))
    mat[:3, 3] = eye
    return mat

def create_scene_for_frame(face_model, tongue_rig, face_seq_frame, ema_seq_frame, 
                           cam_pose, light_config, skin_color):
    """
    Creates a pyrender scene for a single frame.
    """
    # Deform face
    weights = {
        name: val for name, val in zip(face_model.expression_names, face_seq_frame)
    }
    verts = face_model.deform(weights).copy()

    # Deform tongue
    t_verts, _, _ = tongue_rig.deform(ema_seq_frame)
    verts[tongue_rig.global_indices] = t_verts

    # Materials
    mat_skin = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.82, 0.65, 0.55, 1.0],  # "Humane" skin tone (Warmer/Peach)
        metallicFactor=0.0,
        roughnessFactor=0.65, # Slightly rougher to look like skin, not plastic
        alphaMode="OPAQUE",
    )
    
    mat_tongue = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 0.7, 0.7, 1.0],  # Pink tongue
        metallicFactor=0.0,
        roughnessFactor=0.2, # Wet
        alphaMode="OPAQUE",
    )

    mat_gums = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.8, 0.4, 0.4, 1.0],  # Red gums
        metallicFactor=0.0,
        roughnessFactor=0.2, # Wet
        alphaMode="OPAQUE",
    )

    # Scene
    scene = pyrender.Scene(bg_color=[0.3, 0.3, 0.3]) # Gray BG
    
    # Camera
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=cam_pose)

    # Lighting
    for light_def in light_config:
        l_type = light_def.get('type', 'point')
        color = light_def.get('color', [1.0, 1.0, 1.0])
        intensity = light_def.get('intensity', 400)
        pose = light_def.get('pose', np.eye(4))
        
        if l_type == 'spot':
            light = pyrender.SpotLight(color=color, intensity=intensity, 
                                      innerConeAngle=np.pi/8, outerConeAngle=np.pi/4)
        else:
            light = pyrender.PointLight(color=color, intensity=intensity)
            
        scene.add(light, pose=pose)

    # Masks
    current_faces = face_model.faces
    is_tongue_vert = np.zeros(len(face_model.neutral_verts), dtype=bool)
    is_tongue_vert[tongue_rig.global_indices] = True
    is_gum_vert = np.zeros(len(face_model.neutral_verts), dtype=bool)
    is_gum_vert[14062:17039] = True
    is_gum_vert[is_tongue_vert] = False

    face_vert_is_tongue = is_tongue_vert[current_faces]
    is_tongue_face = face_vert_is_tongue.all(axis=1)
    face_vert_is_gum = is_gum_vert[current_faces]
    is_gum_face = face_vert_is_gum.all(axis=1)
    is_skin_face = ~(is_tongue_face | is_gum_face)

    # Add Meshes
    if np.any(is_skin_face):
        mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts, current_faces[is_skin_face], process=False), material=mat_skin, smooth=True)
        for p in mesh.primitives: p.material.doubleSided = True
        scene.add(mesh)
    
    if np.any(is_tongue_face):
        mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts, current_faces[is_tongue_face], process=False), material=mat_tongue, smooth=True)
        for p in mesh.primitives: p.material.doubleSided = True
        scene.add(mesh)

    if np.any(is_gum_face):
        mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts, current_faces[is_gum_face], process=False), material=mat_gums, smooth=True)
        for p in mesh.primitives: p.material.doubleSided = True
        scene.add(mesh)

    return scene

def main():
    print("Loading data...")
    face_model = load_face_model_trimesh(FACE_MODEL_DIR)
    face_seq = process_beat_data(BS_JSON_PATH, face_model, target_fps=FPS)
    
    tongue_rig = FaceKitTongueRig(
        face_model.neutral_verts, face_model.faces, TONGUE_SLICE, 
        ANCHOR_INDICES, BONE_INDICES, TONGUE_CONFIG
    )
    
    raw_ema = np.load(MOTION_PATH)[:, :8].reshape(-1, 4, 2)
    std_raw = np.load(STD_PATH).flatten()[:8].reshape(4, 2)
    
    # Simple manual loading of EMA for this test
    # Just need one frame or a few frames
    denorm_ema_3d = np.zeros((len(raw_ema), 4, 3))
    for i in range(len(raw_ema)):
        delta_2d = raw_ema[i] * std_raw * TONGUE_CONFIG['std_scalar']
        denorm_ema_3d[i, :, 0] = tongue_rig.anchors[:, 0]
        denorm_ema_3d[i, :, 1] = tongue_rig.anchors[:, 1] + delta_2d[:, 1]
        denorm_ema_3d[i, :, 2] = tongue_rig.anchors[:, 2] + delta_2d[:, 0]

    # Pick a frame where mouth is open
    SAMPLE_FRAME = 25 # ~1 second in
    
    ema_sample = denorm_ema_3d[SAMPLE_FRAME]
    face_sample = face_seq[SAMPLE_FRAME]
    
    # Renderer
    W, H = 800, 600
    r = pyrender.OffscreenRenderer(W, H)
    
    # ============================
    # 1. DEFINE CONFIGS
    # ============================
    
    # Skin Color
    SKIN_HUMAN = [0.85, 0.68, 0.58, 1.0] # Fair skin tone
    
    # Default Camera
    eye_def = np.array([0, 0, 35], dtype=np.float32)
    target_def = np.array([0, -2, 0], dtype=np.float32)
    cam_def = get_camera_matrix(eye_def, target_def)
    
    # Angle 1: Side/Elevated (Reduced angle as requested)
    eye_ang1 = np.array([12, 2, 32], dtype=np.float32) # Closer to center (was [20, 5, 25])
    target_ang1 = np.array([0, -2, 0], dtype=np.float32)
    cam_ang1 = get_camera_matrix(eye_ang1, target_ang1)
    
    # Angle 2: Lower/Upwad (Dramtic)
    eye_ang2 = np.array([0, -10, 25], dtype=np.float32)
    target_ang2 = np.array([0, -2, 0], dtype=np.float32)
    cam_ang2 = get_camera_matrix(eye_ang2, target_ang2)
    
    # Lights
    light_pose_def = cam_def.copy()
    light_pose_def[:3, 3] += [0, 10, -5]
    
    lights_default = [
        {'type': 'spot', 'intensity': 300, 'pose': light_pose_def},
        {'type': 'point', 'intensity': 800, 'pose': cam_def}, # Fill
    ]
    
    # Rim Light Setup
    rim_pose = cam_def.copy()
    rim_pose[:3, 3] = [20, 10, -20] # Behind and right
    
    lights_rim = [
        {'type': 'spot', 'intensity': 400, 'pose': light_pose_def},
        {'type': 'point', 'intensity': 300, 'pose': cam_def},
        {'type': 'spot', 'color': [0.9, 0.9, 1.0], 'intensity': 1000, 'pose': rim_pose}
    ]

    # ============================
    # 2. RENDER IMAGES
    # ============================
    configs = [
        ("vis_1_default.png", cam_def, lights_default),
        ("vis_2_side_angle.png", cam_ang1, lights_default),
        ("vis_3_low_angle.png", cam_ang2, lights_default),
        ("vis_4_rim_light.png", cam_def, lights_rim),
        ("vis_5_side_rim.png", cam_ang1, lights_rim),
    ]
    
    print("Rendering static visualizations...")
    for filename, cam, lights in configs:
        scene = create_scene_for_frame(face_model, tongue_rig, face_sample, ema_sample, cam, lights, SKIN_HUMAN)
        color, _ = r.render(scene)
        out_path = OUTPUT_DIR / filename
        cv2.imwrite(str(out_path), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        print(f"  Saved {filename}")

    # ============================
    # 3. RENDER MOVING CAMERA
    # ============================
    print("Rendering moving camera video...")
    video_out = str(OUTPUT_DIR / "vis_moving_cam.mp4")
    video = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    
    # Move camera from center to side over 50 frames (2s)
    frames_mov = 50
    
    for i in range(frames_mov):
        # Interpolate camera pos
        t = i / float(frames_mov - 1)
        # Circular path around center? Or just linear interp?
        # Let's do a semi-circle orbit
        angle = np.radians(25 * np.sin(t * np.pi)) # Reduced to 25 deg (was 45)
        
        orbit_radius = 35.0
        eye_x = np.sin(angle) * orbit_radius
        eye_z = np.cos(angle) * orbit_radius
        eye_curr = np.array([eye_x, 0, eye_z], dtype=np.float32)
        
        target_curr = np.array([0, -2, 0], dtype=np.float32)
        cam_curr = get_camera_matrix(eye_curr, target_curr)
        
        # Keep light attached to camera or fixed?
        # Let's keep main light attached to camera for visibility
        l_pose = cam_curr.copy()
        l_pose[:3, 3] += [0, 10, -5]
        lights_mov = [
        {'type': 'spot', 'intensity': 300, 'pose': l_pose},
        {'type': 'point', 'intensity': 800, 'pose': cam_curr},
        ]
        
        # Use frame from sequence if available, or static
        f_idx = i % len(denorm_ema_3d)
        
        scene = create_scene_for_frame(face_model, tongue_rig, face_seq[f_idx], denorm_ema_3d[f_idx], 
                                      cam_curr, lights_mov, SKIN_HUMAN)
        color, _ = r.render(scene)
        video.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    
    video.release()
    print(f"  Saved vis_moving_cam.mp4")
    
    r.delete()
    print("Done.")

if __name__ == "__main__":
    main()
