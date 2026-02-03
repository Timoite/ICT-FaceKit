#!/usr/bin/env python3
"""
Comprehensive tongue parameter grid search for optimal speech intelligibility.

Tests different combinations of:
- rotation_deg: 0-20 degrees
- thickness: 1.0-4.0
- std_scalar: 0.1-0.4

For each combination, renders a video and extracts VSR transcript.
Results are saved with full metadata for analysis.
"""
import numpy as np
import trimesh
import pyrender
import cv2
import subprocess
import os
import sys
import json
from pathlib import Path
from scipy.interpolate import make_interp_spline, interp1d
from datetime import datetime
from itertools import product
import shutil

# Import local modules
try:
    from face_model_io_trimesh import load_face_model_trimesh
    from render_face_animation_trimesh import map_beat_to_ict_names, load_animation
except ImportError:
    print("CRITICAL: Required python modules not found.")
    sys.exit(1)

# ==========================================
# CONFIGURATION
# ==========================================
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# Paths
FACE_MODEL_DIR = str(PROJECT_ROOT / "FaceXModel")
MOTION_PATH = str(SCRIPT_DIR / "outputs" / "1_wayne_0_75_75.npy")
BS_JSON_PATH = str(SCRIPT_DIR / "inputs" / "1_wayne_0_75_75.json")
AUDIO_PATH = str(SCRIPT_DIR / "inputs" / "1_wayne_0_75_75.wav")
STD_PATH = str(SCRIPT_DIR / "normalising_vectors" / "JW13_4points_std.npy")

# Test parameters
FPS = 50
MAX_SECONDS = 10  # Test with 10 seconds for speed
TONGUE_SLICE = slice(16611, 17039)
ANCHOR_INDICES = [16661, 16696, 16755, 16758]
BONE_INDICES = [16661, 16757]

# Parameter grid - reduced by factor of 4 for faster testing
ROTATION_RANGE = [0, 10, 20]  # degrees (was 5 values, now 3)
THICKNESS_RANGE = [1.0, 2.0, 4.0]  # (was 7 values, now 3)
STD_SCALAR_RANGE = [0.10, 0.25, 0.40]  # (was 7 values, now 3)
# Total: 3 × 3 × 3 = 27 configurations (was 245)

# Output directory
TEST_OUTPUT_DIR = SCRIPT_DIR / "tongue_param_tests"
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================
# DATA LOADING FUNCTIONS
# ==========================================
def process_beat_data(json_path, face_model, target_fps=50):
    if not os.path.exists(json_path): 
        return np.zeros((100, 52))
    
    anim_data = load_animation(json_path)
    beat_names = anim_data['names']
    raw_frames = anim_data['frames']
    
    source_fps = 60
    duration = len(raw_frames) / source_fps
    n_target_frames = int(duration * target_fps)
    
    ict_expr_names = face_model.expression_names
    ict_name_to_idx = {name: i for i, name in enumerate(ict_expr_names)}
    n_ict = len(ict_expr_names)
    
    source_data = np.zeros((len(raw_frames), n_ict), dtype=np.float32)
    
    for f_idx, frame in enumerate(raw_frames):
        for b_idx, weight in enumerate(frame['weights']):
            b_name = beat_names[b_idx]
            mapped_names = map_beat_to_ict_names(b_name)
            for m_name in mapped_names:
                if m_name in ict_name_to_idx:
                    source_data[f_idx, ict_name_to_idx[m_name]] = weight
    
    if len(source_data) < 2: return source_data
    x_source = np.linspace(0, duration, len(source_data))
    x_target = np.linspace(0, duration, n_target_frames)
    f_interp = interp1d(x_source, source_data, axis=0, kind='cubic', fill_value="extrapolate")
    return f_interp(x_target)

def load_ema_motion(motion_path, std_path, rig_anchors, scalar):
    if not os.path.exists(motion_path): 
        raise FileNotFoundError(motion_path)
    
    raw_ema = np.load(motion_path)[:, :8].reshape(-1, 4, 2)
    std_raw = np.load(std_path)
    if std_raw.size >= 8: 
        std_raw = std_raw.flatten()[:8].reshape(4, 2)
    
    denorm_ema_3d = np.zeros((len(raw_ema), 4, 3))
    for i in range(len(raw_ema)):
        delta_2d = raw_ema[i] * std_raw * scalar
        denorm_ema_3d[i, :, 0] = rig_anchors[:, 0]
        denorm_ema_3d[i, :, 1] = rig_anchors[:, 1] + delta_2d[:, 1]
        denorm_ema_3d[i, :, 2] = rig_anchors[:, 2] + delta_2d[:, 0]
        
    return denorm_ema_3d

# ==========================================
# TONGUE RIG CLASS
# ==========================================
class FaceKitTongueRig:
    def __init__(self, vertices, faces, tongue_slice, anchor_indices_global, 
                 bone_ends_global, config):
        self.global_offset = tongue_slice.start
        raw_rest = vertices[tongue_slice].copy()
        
        # Apply thickness
        if config['thickness'] != 1.0:
            cy = np.mean(raw_rest[:, 1])
            raw_rest[:, 1] = cy + (raw_rest[:, 1] - cy) * config['thickness']
        
        # Apply rotation
        if abs(config['rotation_deg']) > 1e-5:
            center = raw_rest.mean(axis=0)
            theta = np.radians(config['rotation_deg'])
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            raw_rest = np.dot(raw_rest - center, R.T) + center
        
        raw_rest[:, 1] += config['shift_y']
        raw_rest[:, 2] += config['shift_z']
        
        self.vertices_rest = raw_rest
        self.faces = faces
        self.global_indices = slice(tongue_slice.start, tongue_slice.stop)
        
        # Setup anchors and bones
        local_anchors = [idx - self.global_offset for idx in anchor_indices_global]
        self.anchors = self.vertices_rest[local_anchors]
        
        mesh_len = np.max(self.vertices_rest[:, 2]) - np.min(self.vertices_rest[:, 2])
        self.radius = mesh_len / 2.5
        
        num_bones = 12
        p1 = self.vertices_rest[bone_ends_global[0] - self.global_offset]
        p2 = self.vertices_rest[bone_ends_global[1] - self.global_offset]
        self.bone_pos_rest = np.linspace(p1, p2, num_bones)
        
        self.k = 3
        u = np.linspace(0, 1, len(self.anchors))
        self.bind_spline = make_interp_spline(u, self.anchors, k=self.k)
        
        self.bone_u_params = self._project_points_to_spline(self.bind_spline, self.bone_pos_rest)
        
        spline_pos_at_bind = self.bind_spline(self.bone_u_params)
        self.bone_offsets = self.bone_pos_rest - spline_pos_at_bind
        
        self.bind_matrices = self._compute_native_matrices(
            self.bind_spline,
            self.bone_u_params,
            self.bone_offsets
        )
        
        self.weights = self._calc_weights(self.vertices_rest, self.bind_matrices)
    
    def _project_points_to_spline(self, spline, points):
        u_samples = np.linspace(0, 1, 1000)
        path = spline(u_samples)
        u_params = []
        for pt in points:
            dists = np.linalg.norm(path - pt, axis=1)
            u_params.append(u_samples[np.argmin(dists)])
        return np.array(u_params)
    
    def _compute_native_matrices(self, spline, u_params, offsets=None):
        pos = spline(u_params)
        if offsets is not None:
            pos += offsets
        
        fixed_right = np.array([1, 0, 0])
        fixed_up = np.array([0, 1, 0])
        fixed_fwd = np.array([0, 0, 1])
        
        mats = []
        for i in range(len(u_params)):
            p = pos[i]
            mat = np.eye(4)
            mat[:3, 0] = fixed_right
            mat[:3, 1] = fixed_up
            mat[:3, 2] = fixed_fwd
            mat[:3, 3] = p
            mats.append(mat)
        return np.array(mats)
    
    def _calc_weights(self, verts, mats):
        w = np.zeros((len(verts), len(mats)))
        for i, m in enumerate(mats):
            d = np.linalg.norm(verts - m[:3, 3], axis=1)
            val = np.clip(1 - (d / self.radius), 0, 1)
            w[:, i] = val * val * (3 - 2 * val)
        row_sum = w.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return w / row_sum
    
    def deform(self, target_anchors):
        u = np.linspace(0, 1, len(self.anchors))
        new_spline = make_interp_spline(u, target_anchors, k=self.k)
        
        pose_mats = self._compute_native_matrices(
            new_spline,
            self.bone_u_params,
            self.bone_offsets
        )
        
        inv_bind = np.linalg.inv(self.bind_matrices)
        deform_mats = np.matmul(pose_mats, inv_bind)
        skin_mats = np.einsum('vj,jkl->vkl', self.weights, deform_mats)
        
        v_homo = np.c_[self.vertices_rest, np.ones(len(self.vertices_rest))]
        v_new = np.matmul(skin_mats, v_homo[..., None]).squeeze(-1)
        
        return v_new[:, :3], pose_mats, new_spline

# ==========================================
# RENDERING FUNCTION
# ==========================================
def render_video(face_model, tongue_rig, ema_seq, face_seq, 
                  output_path, audio_path):
    """
    Render full-face animation with tongue deformation.
    """
    W, H = 800, 600
    r = pyrender.OffscreenRenderer(W, H)
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    
    # Setup camera
    eye = np.array([0, 0, 35], dtype=np.float32)
    target = np.array([0, -2, 0], dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)
    z = eye - target; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = np.column_stack((x, y, z))
    cam_pose[:3, 3] = eye
    
    # Materials
    mat_skin = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.5, 0.5, 0.5, 1.0], 
        metallicFactor=0.0,
        roughnessFactor=0.8, 
        alphaMode='OPAQUE'
    )
    
    mat_tongue = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 0.6, 0.6, 1.0], 
        metallicFactor=0.0,
        roughnessFactor=0.2, 
        alphaMode='OPAQUE'
    )
    
    mat_gums = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.2, 0.2, 1.0], 
        metallicFactor=0.0,
        roughnessFactor=0.2, 
        alphaMode='OPAQUE'
    )
    
    # Setup scene
    scene = pyrender.Scene(bg_color=[0, 0, 0])
    spot_pose = cam_pose.copy()
    spot_pose[:3, 3] += [0, 10, -5]
    spot_light = pyrender.SpotLight(color=np.ones(3), intensity=100, 
                                    innerConeAngle=np.pi/8, outerConeAngle=np.pi/4)
    fill_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=400)
    
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi/3.0), pose=cam_pose)
    scene.add(spot_light, pose=spot_pose)
    scene.add(fill_light, pose=cam_pose)
    
    frames = min(len(ema_seq), len(face_seq), int(MAX_SECONDS * FPS))
    
    # Pre-calculate masks
    is_tongue_vert = np.zeros(len(face_model.neutral_verts), dtype=bool)
    is_tongue_vert[tongue_rig.global_indices] = True
    
    is_gum_vert = np.zeros(len(face_model.neutral_verts), dtype=bool)
    is_gum_vert[14062:17039] = True
    is_gum_vert[is_tongue_vert] = False
    
    for i in range(frames):
        if i % 25 == 0: 
            print(f"  Rendering frame {i}/{frames}...")
        
        # Deform face
        weights = {name: val for name, val in zip(face_model.expression_names, face_seq[i])}
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
        
        nodes = []
        
        # Add meshes
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
        
        # Render
        color, _ = r.render(scene)
        video.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        
        # Cleanup
        for n in nodes:
            scene.remove_node(n)
    
    video.release()
    r.delete()

# ==========================================
# VSR INFERENCE
# ==========================================
def run_vsr_inference(video_path, vsr_dir):
    """
    Extract transcript using VSR model WITH language model.
    """
    import torch
    vsr_dir_path = Path(vsr_dir)
    sys.path.insert(0, str(vsr_dir_path))
    
    from pipelines.model import AVSR
    from pipelines.data.data_module import AVSRDataLoader
    from pipelines.detectors.mediapipe.detector import LandmarksDetector
    
    device = torch.device("cpu")
    
    # Model paths - using downloaded language models
    model_path = str(PROJECT_ROOT / "LRS3_V_WER19.1" / "model.pth")
    model_conf = str(PROJECT_ROOT / "LRS3_V_WER19.1" / "model.json")
    lm_path = str(PROJECT_ROOT / "lm_en_subword" / "model.pth")
    lm_conf = str(PROJECT_ROOT / "lm_en_subword" / "model.json")
    
    try:
        # Initialize dataloader
        dataloader = AVSRDataLoader(modality="video", speed_rate=50.0/25.0, detector="mediapipe")
        
        # Load model with language model
        model = AVSR(
            modality="video",
            model_path=model_path,
            model_conf=model_conf,
            rnnlm=lm_path,
            rnnlm_conf=lm_conf,
            penalty=0.0,
            ctc_weight=0.1,
            lm_weight=0.3,  # Use language model! (0.2-0.4 is typical)
            beam_size=40,
            device=device
        )
        
        # Detect landmarks
        detector = LandmarksDetector()
        landmarks = detector(str(video_path))
        
        # Load video data
        data = dataloader.load_data(str(video_path), landmarks)
        
        # Run inference
        transcript = model.infer(data)
        
        # Clean transcript
        transcript = transcript.replace("▁", " ").strip()
        
        return transcript
    except Exception as e:
        import traceback
        return f"ERROR: {str(e)}\n{traceback.format_exc()}"

# ==========================================
# MAIN TEST FUNCTION
# ==========================================
def run_parameter_grid_test():
    """
    Run comprehensive parameter grid test.
    """
    print("="*60)
    print("TONGUE PARAMETER GRID SEARCH")
    print("="*60)
    print(f"Testing {len(ROTATION_RANGE)} × {len(THICKNESS_RANGE)} × {len(STD_SCALAR_RANGE)} = {len(ROTATION_RANGE) * len(THICKNESS_RANGE) * len(STD_SCALAR_RANGE)} configurations")
    print(f"Output directory: {TEST_OUTPUT_DIR}")
    print()
    
    # Load face model once
    print("Loading face model...")
    face_model = load_face_model_trimesh(FACE_MODEL_DIR)
    print(f"✓ Face model loaded: {len(face_model.neutral_verts)} vertices")
    
    # Load animation data once
    print("Loading animation data...")
    face_seq = process_beat_data(BS_JSON_PATH, face_model, target_fps=FPS)
    print(f"✓ Face sequence loaded: {len(face_seq)} frames")
    
    # Load neutral vertices for rig initialization
    neutral_verts = face_model.neutral_verts
    faces = face_model.faces
    
    # Create results tracking
    results = []
    
    # Get total combinations
    total_combinations = len(ROTATION_RANGE) * len(THICKNESS_RANGE) * len(STD_SCALAR_RANGE)
    current_combo = 0
    
    # Iterate through all combinations
    for rotation in ROTATION_RANGE:
        for thickness in THICKNESS_RANGE:
            for std_scalar in STD_SCALAR_RANGE:
                current_combo += 1
                
                # Config name
                config_name = f"rot{rotation:02d}_thick{thickness:.1f}_std{std_scalar:.2f}"
                
                print(f"\n[{current_combo}/{total_combinations}] Testing: {config_name}")
                
                # Create output directory for this config
                config_dir = TEST_OUTPUT_DIR / config_name
                config_dir.mkdir(exist_ok=True)
                
                # Tongue config
                tongue_config = {
                    "rotation_deg": rotation,
                    "thickness": thickness,
                    "shift_y": 0,
                    "shift_z": 0,
                    "std_scalar": std_scalar
                }
                
                # Save config metadata
                with open(config_dir / "config.json", "w") as f:
                    json.dump(tongue_config, f, indent=2)
                
                try:
                    # Initialize rig with this config
                    tongue_rig = FaceKitTongueRig(
                        neutral_verts,
                        faces,
                        TONGUE_SLICE,
                        ANCHOR_INDICES,
                        BONE_INDICES,
                        tongue_config
                    )
                    
                    # Load EMA motion with this std_scalar
                    ema_seq = load_ema_motion(MOTION_PATH, STD_PATH, 
                                              tongue_rig.anchors, std_scalar)
                    
                    # Render video
                    video_path = config_dir / "animation.mp4"
                    print(f"  Rendering video...")
                    render_video(face_model, tongue_rig, ema_seq, face_seq, 
                                   str(video_path), AUDIO_PATH)
                    
                    # Merge audio
                    final_video = config_dir / "animation_with_audio.mp4"
                    cmd = ["ffmpeg", "-y", "-v", "quiet", "-i", str(video_path), 
                           "-i", AUDIO_PATH,
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", 
                           "-c:a", "aac", "-shortest", str(final_video)]
                    subprocess.run(cmd)
                    
                    # Remove temp video
                    if os.path.exists(final_video):
                        os.remove(video_path)
                        video_path = final_video
                    
                    print(f"  ✓ Video rendered: {video_path.name}")
                    
                    # Run VSR inference
                    print(f"  Running VSR inference...")
                    vsr_dir = PROJECT_ROOT / "ADFA_EVALUATION" / "Visual_Speech_Recognition_for_Multiple_Languages"
                    transcript = run_vsr_inference(video_path, vsr_dir)
                    
                    # Save transcript
                    with open(config_dir / "transcript.txt", "w") as f:
                        f.write(transcript)
                    
                    print(f"  ✓ Transcript: {transcript[:100]}...")
                    
                    # Save result
                    result = {
                        "config_name": config_name,
                        "rotation": rotation,
                        "thickness": thickness,
                        "std_scalar": std_scalar,
                        "video_path": str(video_path),
                        "transcript": transcript,
                        "transcript_length": len(transcript)
                    }
                    results.append(result)
                    
                    # Update progress
                    with open(TEST_OUTPUT_DIR / "progress.json", "w") as f:
                        json.dump({
                            "total": total_combinations,
                            "completed": current_combo,
                            "results": results
                        }, f, indent=2)
                    
                except Exception as e:
                    print(f"  ✗ ERROR: {e}")
                    result = {
                        "config_name": config_name,
                        "rotation": rotation,
                        "thickness": thickness,
                        "std_scalar": std_scalar,
                        "error": str(e)
                    }
                    results.append(result)
    
    # Save final results
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    
    with open(TEST_OUTPUT_DIR / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    print("\nGenerating summary...")
    successful = [r for r in results if "transcript" in r]
    failed = [r for r in results if "error" in r]
    
    print(f"Total configurations: {total_combinations}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    # Show sample transcripts
    print("\nSample transcripts (first 5):")
    for i, result in enumerate(successful[:5]):
        print(f"  {i+1}. {result['config_name']}: {result['transcript'][:80]}...")
    
    print(f"\nResults saved to: {TEST_OUTPUT_DIR}")

if __name__ == "__main__":
    run_parameter_grid_test()
