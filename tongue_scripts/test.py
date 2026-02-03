import numpy as np
import pudb
import trimesh
import pyrender
import cv2
import subprocess
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.interpolate import make_interp_spline, interp1d
from pathlib import Path

# --- IMPORT USER MODULES ---
try:
    from face_model_io_trimesh import load_face_model_trimesh
    from render_face_animation_trimesh import map_beat_to_ict_names, load_animation
except ImportError:
    print("CRITICAL: Required python modules not found.")
    sys.exit(1)

# ==========================================
# CONFIGURATION
# ==========================================
RENDER_MODE     = 'FULL_FACE' # Options: 'MATPLOTLIB' (Debug) or 'FULL_FACE' (Video)
CUTOUT_MODE     = False        # Set to True to see the sagittal cut view

# --- PATHS (Absolute paths for robust execution from any directory) ---
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

FACE_MODEL_DIR  = str(PROJECT_ROOT / "FaceXModel")
MOTION_PATH     = str(SCRIPT_DIR / "outputs" / "1_wayne_0_75_75.npy")
BS_JSON_PATH    = str(SCRIPT_DIR / "inputs" / "1_wayne_0_75_75.json")
AUDIO_PATH      = str(SCRIPT_DIR / "inputs" / "1_wayne_0_75_75.wav")
STD_PATH        = str(SCRIPT_DIR / "normalising_vectors" / "JW13_4points_std.npy")

# --- OUTPUTS ---
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_VIDEO    = str(OUTPUT_DIR / "tongue_hybrid_deformation.mp4")
TEMP_VIDEO      = str(OUTPUT_DIR / "temp_visuals.mp4")

# --- SETTINGS ---
FPS             = 50
MAX_SECONDS     = 10
TONGUE_CONFIG   = {
    "rotation_deg": 5,
    "thickness":    1.2,
    "shift_y":      0,
    "shift_z":      0,
    "std_scalar":   0.20 # Scalar from your working file (was 0.15 in uploaded, 0.25 in current)
}

# --- INDICES ---
TONGUE_SLICE    = slice(16611, 17039)
ANCHOR_INDICES  = [16661,16696,16755,16758] 
BONE_INDICES    = [16661, 16757]               

# ==========================================
# PART 1: DATA LOADING
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
    if not os.path.exists(motion_path): raise FileNotFoundError(motion_path)
    
    raw_ema = np.load(motion_path)[:, :8].reshape(-1, 4, 2)
    std_raw = np.load(std_path)
    if std_raw.size >= 8: std_raw = std_raw.flatten()[:8].reshape(4, 2)
    
    denorm_ema_3d = np.zeros((len(raw_ema), 4, 3))
    for i in range(len(raw_ema)):
        delta_2d = raw_ema[i] * std_raw * scalar
        denorm_ema_3d[i, :, 0] = rig_anchors[:, 0]
        denorm_ema_3d[i, :, 1] = rig_anchors[:, 1] + delta_2d[:, 1]
        denorm_ema_3d[i, :, 2] = rig_anchors[:, 2] + delta_2d[:, 0]
        
    return denorm_ema_3d

# ==========================================
# PART 2: HYBRID TONGUE RIG
# ==========================================

class FaceKitTongueRig:
    def __init__(self, vertices, faces, tongue_slice, anchor_indices_global, bone_ends_global, config):
        self.global_offset = tongue_slice.start

        # 1. SETUP MESH
        raw_rest = vertices[tongue_slice].copy()

        if config['thickness'] != 1.0:
            cy = np.mean(raw_rest[:, 1])
            raw_rest[:, 1] = cy + (raw_rest[:, 1] - cy) * config['thickness']

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

        # 2. SETUP ANCHORS & BONES
        local_anchors = [idx - self.global_offset for idx in anchor_indices_global]
        self.anchors = self.vertices_rest[local_anchors]

        mesh_len = np.max(self.vertices_rest[:, 2]) - np.min(self.vertices_rest[:, 2])
        self.radius = mesh_len / 2.5

        # Generate Linear Bones
        num_bones = 12
        p1 = self.vertices_rest[bone_ends_global[0] - self.global_offset]
        p2 = self.vertices_rest[bone_ends_global[1] - self.global_offset]
        self.bone_pos_rest = np.linspace(p1, p2, num_bones) # Store this for offset calc

        # 3. SPLINE & BINDING
        self.k = 3
        u = np.linspace(0, 1, len(self.anchors))
        self.bind_spline = make_interp_spline(u, self.anchors, k=self.k)

        # Project bones to spline to get fixed 'u' parameters
        self.bone_u_params = self._project_points_to_spline(self.bind_spline, self.bone_pos_rest)

        # --- FIX START: Calculate Offsets ---
        # 1. Where is the spline exactly at these u parameters?
        spline_pos_at_bind = self.bind_spline(self.bone_u_params)

        # 2. Calculate vector from Spline -> Real Bone Position
        # We will add this back during deformation so the bone stays "floating" relative to the spline
        self.bone_offsets = self.bone_pos_rest - spline_pos_at_bind

        # 3. Compute bind matrices using the offsets
        self.bind_matrices = self._compute_native_matrices(
            self.bind_spline,
            self.bone_u_params,
            self.bone_offsets
        )
        # --- FIX END ---

        self.weights = self._calc_weights(self.vertices_rest, self.bind_matrices)

    def _project_points_to_spline(self, spline, points):
        u_samples = np.linspace(0, 1, 1000)
        path = spline(u_samples)
        u_params = []
        for pt in points:
            dists = np.linalg.norm(path - pt, axis=1)
            u_params.append(u_samples[np.argmin(dists)])
        return np.array(u_params)

    # --- UPDATED: Accepts optional offsets ---
    def _compute_native_matrices(self, spline, u_params, offsets=None):
        """
        Computes bone matrices with TRANSLATION from the spline + OFFSET.
        """
        # 1. Get Positions (Translation) from Spline
        pos = spline(u_params)

        # Apply offsets if they exist (Restores the "middle" structure)
        if offsets is not None:
            pos += offsets

        # 2. Define Fixed Orientation
        fixed_right = np.array([1, 0, 0])
        fixed_up    = np.array([0, 1, 0])
        fixed_fwd   = np.array([0, 0, 1])

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
        # 1. Fit Spline
        u = np.linspace(0, 1, len(self.anchors))
        new_spline = make_interp_spline(u, target_anchors, k=self.k)

        # 2. Compute Matrices (Pass the stored offsets!)
        pose_mats = self._compute_native_matrices(
            new_spline,
            self.bone_u_params,
            self.bone_offsets
        )

        # 3. LBS Skinning
        inv_bind = np.linalg.inv(self.bind_matrices)
        deform_mats = np.matmul(pose_mats, inv_bind)
        skin_mats = np.einsum('vj,jkl->vkl', self.weights, deform_mats)

        v_homo = np.c_[self.vertices_rest, np.ones(len(self.vertices_rest))]
        v_new = np.matmul(skin_mats, v_homo[..., None]).squeeze(-1)

        return v_new[:, :3], pose_mats, new_spline

# ==========================================
# PART 3: RENDERERS
# ==========================================
def run_matplotlib_debug(tongue_rig, ema_seq):
    print("Starting Matplotlib Debug View...")
    frames = min(len(ema_seq), int(MAX_SECONDS * FPS))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-5, 15); ax.set_ylim(-10, 10)
    ax.set_title("Hybrid Rig Debug")
    
    # Static
    midline = np.abs(tongue_rig.vertices_rest[:, 0]) < 1.0
    rest_prof = tongue_rig.vertices_rest[midline][:, [2, 1]] 
    ax.scatter(rest_prof[:, 0], rest_prof[:, 1], s=2, c='lightgrey', alpha=0.3)
    
    # Dynamic
    scat_mesh    = ax.scatter([], [], s=5, c='blue', alpha=0.3)
    scat_anchors = ax.scatter([], [], s=50, c='red', marker='x')
    scat_bones   = ax.scatter([], [], s=30, c='orange', marker='s', edgecolors='black')
    line_spline, = ax.plot([], [], 'purple', linewidth=2)

    def update(i):
        verts, bone_mats, cur_spline = tongue_rig.deform(ema_seq[i])
        
        prof = verts[midline][:, [2, 1]]
        scat_mesh.set_offsets(prof)
        scat_anchors.set_offsets(ema_seq[i][:, [2, 1]])
        scat_bones.set_offsets(bone_mats[:, :3, 3][:, [2, 1]])
        
        u_plot = np.linspace(0, 1, 100)
        pts = cur_spline(u_plot)[:, [2, 1]]
        line_spline.set_data(pts[:, 0], pts[:, 1])
        
        return scat_mesh, scat_anchors, scat_bones, line_spline

    anim = FuncAnimation(fig, update, frames=frames, interval=1000/FPS, blit=False)
    anim.save(TEMP_VIDEO, writer=FFMpegWriter(fps=FPS))
    plt.close()

def run_pyrender_video(face_model, tongue_rig, ema_seq, face_seq):
    print("Starting Pyrender Video (Tri-Color Split)...")
    W, H = 800, 600
    r = pyrender.OffscreenRenderer(W, H)
    video = cv2.VideoWriter(TEMP_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    
    # --- 1. SETUP CAMERA ---
    if CUTOUT_MODE:
        eye = np.array([20, -2, 4], dtype=np.float32)
        target = np.array([0, -3, 2], dtype=np.float32)
    else:
        eye = np.array([0, 0, 35], dtype=np.float32) 
        target = np.array([0, -2, 0], dtype=np.float32)
        
    up = np.array([0, 1, 0], dtype=np.float32)
    z = eye - target; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = np.column_stack((x, y, z))
    cam_pose[:3, 3] = eye

    # --- 2. DEFINE MATERIALS ---
    
    # A: SKIN (Grey, Matte)
    mat_skin = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.5, 0.5, 0.5, 1.0], 
        metallicFactor=0.0,
        roughnessFactor=0.8, 
        alphaMode='OPAQUE'
    )

    # B: TONGUE (Lighter Pink, Wet)
    # RGB [255, 150, 150] approx -> Light Pink
    mat_tongue = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 0.6, 0.6, 1.0], 
        metallicFactor=0.0,
        roughnessFactor=0.2, 
        alphaMode='OPAQUE'
    )

    # C: GUMS (Darker Pink, Wet)
    # RGB [180, 50, 50] approx -> Deep Red/Pink
    mat_gums = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.2, 0.2, 1.0], 
        metallicFactor=0.0,
        roughnessFactor=0.2, 
        alphaMode='OPAQUE'
    )

    # --- 3. SETUP SCENE & LIGHTS ---
    scene_base = pyrender.Scene(bg_color=[0, 0, 0])
    
    spot_pose = cam_pose.copy()
    spot_pose[:3, 3] += [0, 10, -5] 
    spot_light = pyrender.SpotLight(color=np.ones(3), intensity=100, 
                                    innerConeAngle=np.pi/8, outerConeAngle=np.pi/4)
    fill_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=400)
    
    scene_base.add(pyrender.PerspectiveCamera(yfov=np.pi/3.0), pose=cam_pose)
    scene_base.add(spot_light, pose=spot_pose)
    scene_base.add(fill_light, pose=cam_pose)

    frames = min(len(ema_seq), len(face_seq), int(MAX_SECONDS * FPS))
    
    # --- 4. PRE-CALCULATE MASKS ---
    
    # Mask 1: Tongue Vertices
    is_tongue_vert = np.zeros(len(face_model.neutral_verts), dtype=bool)
    is_tongue_vert[tongue_rig.global_indices] = True
    
    # Mask 2: Gum Vertices (14062 -> 17038)
    is_gum_vert = np.zeros(len(face_model.neutral_verts), dtype=bool)
    is_gum_vert[14062:17039] = True
    
    # (Optional) Ensure mutual exclusivity if ranges overlap
    # Here we prioritize Tongue over Gums if they overlap
    is_gum_vert[is_tongue_vert] = False

    for i in range(frames):
        if i % 25 == 0: print(f"Rendering frame {i}/{frames}...")
        
        # Deform
        weights = {name: val for name, val in zip(face_model.expression_names, face_seq[i])}
        verts = face_model.deform(weights).copy()
        
        t_verts, _, _ = tongue_rig.deform(ema_seq[i])
        verts[tongue_rig.global_indices] = t_verts
        
        # Handle Cutout
        current_faces = face_model.faces
        if CUTOUT_MODE:
            valid_mask = verts[:, 0] < 0.1
            valid_faces_mask = valid_mask[current_faces].all(axis=1)
            current_faces = current_faces[valid_faces_mask]

        # SPLIT GEOMETRY: Tongue vs Gums vs Skin
        
        # Check membership for every face
        # A face belongs to a group if ALL its vertices belong to that group
        
        face_vert_is_tongue = is_tongue_vert[current_faces]
        is_tongue_face = face_vert_is_tongue.all(axis=1)
        
        face_vert_is_gum = is_gum_vert[current_faces]
        is_gum_face = face_vert_is_gum.all(axis=1)
        
        # Anything not Tongue or Gum is Skin
        is_skin_face = ~(is_tongue_face | is_gum_face)
        
        faces_tongue = current_faces[is_tongue_face]
        faces_gum    = current_faces[is_gum_face]
        faces_skin   = current_faces[is_skin_face]
        
        nodes = []
        
        # 1. Skin Mesh
        if len(faces_skin) > 0:
            tm_skin = trimesh.Trimesh(verts, faces_skin, process=False)
            mesh_skin = pyrender.Mesh.from_trimesh(tm_skin, material=mat_skin, smooth=True)
            if mesh_skin.primitives:
                for p in mesh_skin.primitives: p.material.doubleSided = True
            nodes.append(scene_base.add(mesh_skin))
            
        # 2. Tongue Mesh
        if len(faces_tongue) > 0:
            tm_tongue = trimesh.Trimesh(verts, faces_tongue, process=False)
            mesh_tongue = pyrender.Mesh.from_trimesh(tm_tongue, material=mat_tongue, smooth=True)
            if mesh_tongue.primitives:
                for p in mesh_tongue.primitives: p.material.doubleSided = True
            nodes.append(scene_base.add(mesh_tongue))

        # 3. Gum Mesh
        if len(faces_gum) > 0:
            tm_gum = trimesh.Trimesh(verts, faces_gum, process=False)
            mesh_gum = pyrender.Mesh.from_trimesh(tm_gum, material=mat_gums, smooth=True)
            if mesh_gum.primitives:
                for p in mesh_gum.primitives: p.material.doubleSided = True
            nodes.append(scene_base.add(mesh_gum))

        # Render & Cleanup
        color, _ = r.render(scene_base)
        video.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        
        for n in nodes:
            scene_base.remove_node(n)

    video.release()
    r.delete()



def merge_audio():
    if not os.path.exists(TEMP_VIDEO): return
    if not os.path.exists(AUDIO_PATH):
        os.rename(TEMP_VIDEO, OUTPUT_VIDEO)
        return

    cmd = ["ffmpeg", "-y", "-v", "quiet", "-i", TEMP_VIDEO, "-i", AUDIO_PATH,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", OUTPUT_VIDEO]
    subprocess.run(cmd)
    if os.path.exists(OUTPUT_VIDEO): os.remove(TEMP_VIDEO)

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    face_model = load_face_model_trimesh(FACE_MODEL_DIR)
    
    print("Initializing Hybrid Rig...")
    # Automatic Anchor/Bone Placement (Current Script)
    tongue_rig = FaceKitTongueRig(
        face_model.neutral_verts, 
        face_model.faces, 
        TONGUE_SLICE,
        ANCHOR_INDICES,
        BONE_INDICES, 
        TONGUE_CONFIG
    )
    
    face_seq = process_beat_data(BS_JSON_PATH, face_model, target_fps=FPS)
    ema_seq  = load_ema_motion(MOTION_PATH, STD_PATH, tongue_rig.anchors, TONGUE_CONFIG['std_scalar'])
    
    if RENDER_MODE == 'MATPLOTLIB':
        run_matplotlib_debug(tongue_rig, ema_seq)
    else:
        run_pyrender_video(face_model, tongue_rig, ema_seq, face_seq)
        
    merge_audio()
