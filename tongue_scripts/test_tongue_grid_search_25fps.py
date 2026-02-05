#!/usr/bin/env python3
"""
Comprehensive tongue parameter grid search for optimal speech intelligibility.
Renders at 25fps.
Generates:
1. Standard Frontal Video (for VSR Inference)
2. Moving Camera Video (for User Inspection)
Stores results in 'grid/' folder.
"""

import sys
import numpy as np
import trimesh
import pyrender
import cv2
import subprocess
import os
import json
import shutil
from pathlib import Path
from scipy.interpolate import interp1d

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

# Import existing modules
try:
    from face_model_io_trimesh import load_face_model_trimesh
    from render_face_animation_trimesh import map_beat_to_ict_names, load_animation
    from test import process_beat_data, load_ema_motion, FaceKitTongueRig, TONGUE_CONFIG
except ImportError:
    # Try parent dir
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    from face_model_io_trimesh import load_face_model_trimesh
    from render_face_animation_trimesh import map_beat_to_ict_names, load_animation
    from tongue_scripts.test import process_beat_data, load_ema_motion, FaceKitTongueRig, TONGUE_CONFIG

# Define paths
PROJECT_ROOT = SCRIPT_DIR.parent
FACE_MODEL_DIR = str(PROJECT_ROOT / "FaceXModel")
MOTION_PATH = str(SCRIPT_DIR / "outputs" / "1_wayne_0_75_75.npy")
BS_JSON_PATH = str(SCRIPT_DIR / "inputs" / "1_wayne_0_75_75.json")
AUDIO_PATH = str(SCRIPT_DIR / "inputs" / "1_wayne_0_75_75.wav")
STD_PATH = str(SCRIPT_DIR / "normalising_vectors" / "JW13_4points_std.npy")

OUTPUT_BASE_DIR = SCRIPT_DIR / "grid"
OUTPUT_BASE_DIR.mkdir(exist_ok=True)

# Config
FPS = 25
MAX_SECONDS = 7.5
TONGUE_SLICE = slice(16611, 17039)
ANCHOR_INDICES = [16661, 16696, 16755, 16758]
BONE_INDICES = [16661, 16757]

# Grid Parameters
ROTATION_RANGE = [0, 10, 20]
THICKNESS_RANGE = [1.0, 2.0, 4.0]
STD_SCALAR_RANGE = [0.10, 0.25, 0.40]

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

def run_vsr_inference(video_path):
    """Run VSR inference on the video."""
    vsr_dir = PROJECT_ROOT / "ADFA_EVALUATION" / "Visual_Speech_Recognition_for_Multiple_Languages"
    
    # We will run this as a subprocess to avoid polluting the current python process 
    # with conflicting torch/hydra configs if possible, or we can import if we are careful.
    # Given the previous script imported it, we can try importing.
    
    try:
        sys.path.insert(0, str(vsr_dir))
        # We need to ensure we don't re-import or mess up paths if called multiple times
        # But for safety, let's use the same logic as the previous script:
        # It imported inside the function.
        
        import torch
        from pipelines.model import AVSR
        from pipelines.data.data_module import AVSRDataLoader
        from pipelines.detectors.mediapipe.detector import LandmarksDetector

        device = torch.device("cpu") # Force CPU to avoid CUDA OOM if running other things, or use CUDA if available

        model_path = str(PROJECT_ROOT / "ADFA_EVALUATION" / "Visual_Speech_Recognition_for_Multiple_Languages" / "data" / "LRS3_V_WER19.1" / "model.pth")
        model_conf = str(PROJECT_ROOT / "ADFA_EVALUATION" / "Visual_Speech_Recognition_for_Multiple_Languages" / "data" / "LRS3_V_WER19.1" / "model.json")
        lm_path = None # LM not found
        lm_conf = None

        # Initialize dataloader with speed_rate=1.0 for 25fps
        dataloader = AVSRDataLoader(modality="video", speed_rate=1.0, detector="mediapipe")
        
        model = AVSR(
            modality="video",
            model_path=model_path,
            model_conf=model_conf,
            rnnlm=None, # Disable LM
            rnnlm_conf=None,
            penalty=0.0,
            ctc_weight=0.1,
            lm_weight=0.0, # Zero weight
            beam_size=40,
            device=device
        )
        
        detector = LandmarksDetector()
        landmarks = detector(str(video_path))
        data = dataloader.load_data(str(video_path), landmarks)
        transcript = model.infer(data)
        return transcript.replace("▁", " ").strip()
        
    except Exception as e:
        return f"ERROR: {str(e)}"

def merge_audio(video_path, audio_path, output_path):
    cmd = [
        "ffmpeg", "-y", "-v", "quiet",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(output_path)
    ]
    subprocess.run(cmd)

def main():
    print("Loading resources...")
    face_model = load_face_model_trimesh(FACE_MODEL_DIR)
    
    # Load face sequence once
    face_seq = process_beat_data(BS_JSON_PATH, face_model, target_fps=FPS)
    
    # Pre-calc logical masks
    is_tongue_vert = np.zeros(len(face_model.neutral_verts), dtype=bool)
    is_gum_vert = np.zeros(len(face_model.neutral_verts), dtype=bool)
    is_gum_vert[14062:17039] = True
    
    # Renderer
    W, H = 800, 600
    r = pyrender.OffscreenRenderer(W, H)
    
    total_runs = len(ROTATION_RANGE) * len(THICKNESS_RANGE) * len(STD_SCALAR_RANGE)
    current_run = 0
    
    results = []

    # Iterate Grid
    for rot in ROTATION_RANGE:
        for thick in THICKNESS_RANGE:
            for scalar in STD_SCALAR_RANGE:
                current_run += 1
                run_name = f"rot{rot}_thick{thick}_std{scalar}"
                run_dir = OUTPUT_BASE_DIR / run_name
                run_dir.mkdir(exist_ok=True)
                
                print(f"[{current_run}/{total_runs}] Running {run_name}...")
                
                # Config
                config = TONGUE_CONFIG.copy()
                config['rotation_deg'] = rot
                config['thickness'] = thick
                config['std_scalar'] = scalar
                
                # Setup Rig
                tongue_rig = FaceKitTongueRig(
                    face_model.neutral_verts, face_model.faces, TONGUE_SLICE,
                    ANCHOR_INDICES, BONE_INDICES, config
                )
                
                # Refine masks
                is_tongue_vert[:] = False
                is_tongue_vert[tongue_rig.global_indices] = True
                is_gum_vert[is_tongue_vert] = False # Exclusive
                
                # Load Motion
                ema_seq = load_ema_motion(MOTION_PATH, STD_PATH, tongue_rig.anchors, scalar)
                
                # Resample EMA (50fps -> 25fps)
                # Simple subsampling if exactly half, otherwise interp
                # Assuming EMA is roughly 50Hz and we want 25Hz
                ema_seq_25 = ema_seq[::2] 
                
                # Length
                n_frames = min(len(face_seq), len(ema_seq_25), int(MAX_SECONDS * FPS))
                
                # Video Writers
                vid_vsr_path = run_dir / "temp_vsr.mp4"
                vid_mov_path = run_dir / "temp_mov.mp4"
                vid_low_path = run_dir / "temp_low.mp4"
                
                writer_vsr = cv2.VideoWriter(str(vid_vsr_path), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
                writer_mov = cv2.VideoWriter(str(vid_mov_path), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
                writer_low = cv2.VideoWriter(str(vid_low_path), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
                
                # Materials (Same)
                mat_skin = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[0.82, 0.65, 0.55, 1.0], # Humane
                    metallicFactor=0.0, roughnessFactor=0.65, alphaMode="OPAQUE"
                )
                mat_tongue = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[1.0, 0.7, 0.7, 1.0], 
                    metallicFactor=0.0, roughnessFactor=0.2, alphaMode="OPAQUE"
                )
                mat_gums = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[0.8, 0.4, 0.4, 1.0], 
                    metallicFactor=0.0, roughnessFactor=0.2, alphaMode="OPAQUE"
                )
                
                # 1. Fixed Camera (Frontal)
                cam_front = get_camera_matrix(np.array([0, 0, 35], dtype=np.float32), 
                                              np.array([0, -2, 0], dtype=np.float32))
                                              
                # 3. Fixed Camera (Low Angle)
                # "Further down pointing up" (Less severe)
                cam_low = get_camera_matrix(np.array([0, -5, 32], dtype=np.float32), 
                                            np.array([0, -2, 0], dtype=np.float32))
                
                # Loop Frames
                for i in range(n_frames):
                    # Deform
                    w = {n: v for n, v in zip(face_model.expression_names, face_seq[i])}
                    verts = face_model.deform(w).copy()
                    t_verts, _, _ = tongue_rig.deform(ema_seq_25[i])
                    verts[tongue_rig.global_indices] = t_verts
                    
                    # calc masks once
                    if i == 0:
                        faces_all = face_model.faces
                        f_v_tongue = is_tongue_vert[faces_all]
                        mask_f_tongue = f_v_tongue.all(axis=1)
                        f_v_gum = is_gum_vert[faces_all]
                        mask_f_gum = f_v_gum.all(axis=1)
                        mask_f_skin = ~(mask_f_tongue | mask_f_gum)
                        
                        f_tongue = faces_all[mask_f_tongue]
                        f_gum = faces_all[mask_f_gum]
                        f_skin = faces_all[mask_f_skin]

                    # Helper to render specific camera
                    def render_pass(camera_pose, out_writer):
                        scene = pyrender.Scene(bg_color=[0.3, 0.3, 0.3])
                        scene.add(pyrender.PerspectiveCamera(yfov=np.pi/3.0), pose=camera_pose)
                        
                        # Light follows cam
                        l_pose = camera_pose.copy(); l_pose[:3, 3] += [0, 10, -5]
                        scene.add(pyrender.SpotLight(color=np.ones(3), intensity=300, innerConeAngle=0.4, outerConeAngle=0.8), pose=l_pose)
                        scene.add(pyrender.PointLight(color=np.ones(3), intensity=800), pose=camera_pose)
                        
                        if len(f_skin) > 0:
                            m = pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts, f_skin, process=False), material=mat_skin, smooth=True)
                            for p in m.primitives: p.material.doubleSided = True
                            scene.add(m)
                        if len(f_tongue) > 0:
                            m = pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts, f_tongue, process=False), material=mat_tongue, smooth=True)
                            for p in m.primitives: p.material.doubleSided = True
                            scene.add(m)
                        if len(f_gum) > 0:
                            m = pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts, f_gum, process=False), material=mat_gums, smooth=True)
                            for p in m.primitives: p.material.doubleSided = True
                            scene.add(m)
                        
                        c, _ = r.render(scene)
                        out_writer.write(cv2.cvtColor(c, cv2.COLOR_RGB2BGR))

                    # 1. Frontal
                    render_pass(cam_front, writer_vsr)
                    
                    # 2. Low Angle
                    render_pass(cam_low, writer_low)

                    # 3. Moving Camera (Circular, Frontal, Slower)
                    t = i /FPS 
                    # "Circle in front" -> Orbit in XY plane around Z axis? Or just small circular motion of eye position?
                    # Let's do small circular motion around (0,0,35)
                    # Radius 5, Period 4s
                    theta = (2 * np.pi * t) / 4.0
                    ex = 5.0 * np.cos(theta)
                    ey = 5.0 * np.sin(theta)
                    cam_mov = get_camera_matrix(np.array([ex, ey, 35], dtype=np.float32), 
                                                np.array([0, -2, 0], dtype=np.float32))
                    render_pass(cam_mov, writer_mov)
                    
                writer_vsr.release()
                writer_mov.release()
                writer_low.release()
                
                # Audio Merge
                final_vsr = run_dir / "vsr_input.mp4"
                final_mov = run_dir / "visualization.mp4"
                final_low = run_dir / "low_angle.mp4"
                
                merge_audio(vid_vsr_path, AUDIO_PATH, final_vsr)
                merge_audio(vid_mov_path, AUDIO_PATH, final_mov)
                merge_audio(vid_low_path, AUDIO_PATH, final_low)
                
                # Cleanup temps
                if final_vsr.exists(): vid_vsr_path.unlink()
                if final_mov.exists(): vid_mov_path.unlink()
                if final_low.exists(): vid_low_path.unlink()
                
                # Inference - Run on ALL
                print(f"  Running Inference on {run_name} (Frontal)...")
                transcript_vsr = run_vsr_inference(final_vsr)
                print(f"    VSR: {transcript_vsr}")
                
                print(f"  Running Inference on {run_name} (Moving)...")
                transcript_mov = run_vsr_inference(final_mov)
                print(f"    MOV: {transcript_mov}")
                
                print(f"  Running Inference on {run_name} (Low)...")
                transcript_low = run_vsr_inference(final_low)
                print(f"    LOW: {transcript_low}")
                
                with open(run_dir / "transcript.txt", "w") as f: f.write(transcript_vsr)
                with open(run_dir / "transcript_mov.txt", "w") as f: f.write(transcript_mov)
                with open(run_dir / "transcript_low.txt", "w") as f: f.write(transcript_low)

                results.append({
                    "run": run_name,
                    "rot": rot, "thick": thick, "scalar": scalar,
                    "transcript_vsr": transcript_vsr,
                    "transcript_mov": transcript_mov,
                    "transcript_low": transcript_low
                })
                
                # Update progress
                with open(OUTPUT_BASE_DIR / "results_progress.json", "w") as f:
                    json.dump(results, f, indent=2)
                    
    r.delete()
    print("Grid Search Complete.")
    
if __name__ == "__main__":
    main()
