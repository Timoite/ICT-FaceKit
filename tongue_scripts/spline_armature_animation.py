import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import make_interp_spline
import os
import subprocess
import shutil

# ==========================================
# CONFIGURATION
# ==========================================
MANUAL_BONE_COORDS = np.linspace(
    np.array([3.92, -4.52]), 
    np.array([9.79, -4.02]), 
    7
)

EMA_FPS = 50

# Paths
MESH_PATH = "seperated_tongue.obj"
MOTION_PATH = "outputs/motion.npy"
STD_PATH = "normalising_vectors/JW13_4points_std.npy"
AUDIO_PATH = "inputs/test.wav"

# Intermediate and Final Files
TEMP_GIF = "outputs/temp_visuals.gif"
OUTPUT_VIDEO = "outputs/final_tongue_speech.mp4"

# ==========================================
# RIGGING ENGINE
# ==========================================
class FaceKitTongueRig:
    def __init__(self, vertices, faces, ema_rest_frame, num_bones=12, 
                 envelope_radius=None, manual_bone_pos=None):
        self.vertices_rest = vertices
        self.faces = faces
        self.ema_rest_frame = ema_rest_frame
        
        if envelope_radius is None:
            mesh_len = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
            self.radius = mesh_len / 2.5 
        else:
            self.radius = envelope_radius
        
        if manual_bone_pos is not None:
            start, end = manual_bone_pos[0], manual_bone_pos[-1]
        else:
            start, end = ema_rest_frame[0], ema_rest_frame[-1]
            
        self.virtual_rest_shape = np.linspace(start, end, 4)
        self.rest_offset = ema_rest_frame - self.virtual_rest_shape
        
        self.k = 3
        u = np.linspace(0, 1, 4) 
        self.bind_spline = make_interp_spline(u, self.virtual_rest_shape, k=self.k)
        
        if manual_bone_pos is not None:
            self.num_bones = len(manual_bone_pos)
            self.bone_params = self._project_points_to_spline(self.bind_spline, manual_bone_pos)
        else:
            self.bone_params = np.linspace(0, 1, num_bones + 1)[:-1]

        self.bind_matrices = self._compute_native_matrices(self.bind_spline)
        print(f"Binding {len(vertices)} vertices to {self.num_bones} bones...")
        self.weights = self._calc_weights(vertices, self.bind_matrices)

    def _project_points_to_spline(self, spline, points, samples=1000):
        u_samples = np.linspace(0, 1, samples)
        path_points = spline(u_samples)
        found_params = []
        for pt in points:
            dists = np.linalg.norm(path_points - pt, axis=1)
            found_params.append(u_samples[np.argmin(dists)])
        return np.array(found_params)

    def _compute_native_matrices(self, spline_2d):
        pos_2d = spline_2d(self.bone_params)
        tan_2d = spline_2d.derivative()(self.bone_params)
        matrices = []
        for i in range(len(self.bone_params)):
            user_z, user_y = pos_2d[i]
            tz, ty = tan_2d[i]
            pos_vec = np.array([0, user_y, user_z])
            bone_fwd = np.array([0, ty, tz])
            if np.linalg.norm(bone_fwd) < 1e-6: bone_fwd = np.array([0, 0, 1])
            else: bone_fwd /= np.linalg.norm(bone_fwd)
            bone_right = np.array([1, 0, 0])
            bone_up = np.cross(bone_fwd, bone_right)
            mat = np.eye(4)
            mat[:3, 0] = bone_right
            mat[:3, 1] = bone_up
            mat[:3, 2] = bone_fwd
            mat[:3, 3] = pos_vec
            matrices.append(mat)
        return np.array(matrices)

    def _calc_weights(self, verts, mats):
        w = np.zeros((len(verts), len(mats)))
        for i, m in enumerate(mats):
            d = np.linalg.norm(verts - m[:3, 3], axis=1)
            val = np.clip(1 - (d / self.radius), 0, 1)
            w[:, i] = val * val * (3 - 2 * val)
        row_sum = w.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return w / row_sum

    def deform(self, ema_frame):
        target_shape = ema_frame - self.rest_offset
        u = np.linspace(0, 1, 4)
        new_spline = make_interp_spline(u, target_shape, k=self.k)
        pose_mats = self._compute_native_matrices(new_spline)
        inv_bind = np.linalg.inv(self.bind_matrices)
        deform_mats = np.matmul(pose_mats, inv_bind)
        skin_mats = np.sum(self.weights[:, :, None, None] * deform_mats[None, :, :, :], axis=1)
        v_homo = np.hstack([self.vertices_rest, np.ones((len(self.vertices_rest), 1))])
        v_new = np.matmul(skin_mats, v_homo[:, :, None]).squeeze(-1)
        return v_new[:, :3], pose_mats

# --- HELPERS ---
def get_fixed_anchors():
    return np.array([[4.19, -3.79], [6.15, -3.13], [8.52, -3.26], [9.60, -3.79]])

def load_and_reshape_motion(motion_path):
    d = np.load(motion_path)
    return d[:, :8].reshape(-1, 4, 2)

def denormalize_with_fixed_anchors(ema_z, std_path, anchors, std_scalar=0.25):
    s = np.load(std_path)
    if s.size >= 8: s = s.flatten()[:8].reshape(4, 2)
    return (ema_z * (s * std_scalar)) + anchors

# --- MAIN WORKFLOW ---
def run_pipeline(verts, rig, ema_sequence, anchors, fps=60):
    
    # 1. CHECK FFMPEG
    if shutil.which("ffmpeg") is None:
        print("[Error] FFmpeg is not found. Please install it to use this script.")
        return

    # 2. GENERATE GIF
    print(f"Step 1: Generating Visuals ({len(ema_sequence)} frames)...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Setup Static Background
    mid_mask = np.abs(verts[:, 0]) < 0.5 
    orig_prof = verts[mid_mask]
    
    # We add labels here so they appear in the legend
    ax.scatter(orig_prof[:, 2], orig_prof[:, 1], s=10, c='gray', alpha=0.2, label='Rest Mesh')
    ax.scatter(anchors[:, 0], anchors[:, 1], s=80, c='green', marker='^', label='Anchors')

    # Setup Dynamic Objects
    scat_mesh = ax.scatter([], [], s=10, c='blue', alpha=0.5, label='Deformed')
    scat_bones = ax.scatter([], [], s=50, c='orange', marker='s', label='Bones')
    scat_ema = ax.scatter([], [], s=60, c='red', edgecolors='black', label='EMA')
    
    ax.set_xlim(0, 12)
    ax.set_ylim(-8, 0)
    ax.set_xlabel('Mesh Z (Posterior <-> Anterior)')
    ax.set_ylabel('Mesh Y (Inferior <-> Superior)')
    ax.set_title('Tongue Animation')
    ax.grid(True, alpha=0.3)
    
    # --- FIX: FORCE LEGEND ON ---
    # loc='upper right' ensures it stays in a visible spot
    ax.legend(loc='upper right', framealpha=1.0) 

    def update(frame_idx):
        new_verts, bone_mats = rig.deform(ema_sequence[frame_idx])
        curr_prof = new_verts[mid_mask]
        curr_bones = bone_mats[:, :3, 3][:, [2, 1]]
        curr_ema = ema_sequence[frame_idx]
        
        scat_mesh.set_offsets(curr_prof[:, [2, 1]])
        scat_bones.set_offsets(curr_bones)
        scat_ema.set_offsets(curr_ema)
        
        ax.set_title(f"Frame {frame_idx}/{len(ema_sequence)}")
        return scat_mesh, scat_bones, scat_ema

    # --- FIX: BLIT=FALSE ---
    # blit=False ensures the entire frame (including Legend and Title) is redrawn every time.
    anim = FuncAnimation(fig, update, frames=len(ema_sequence), interval=(1000/fps), blit=False)
    
    print(f"Saving temporary GIF: {TEMP_GIF}")
    anim.save(TEMP_GIF, writer=PillowWriter(fps=fps))
    plt.close(fig)

    # 3. MERGE AUDIO
    print(f"Step 2: Merging audio from {AUDIO_PATH}...")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", TEMP_GIF,
        "-i", AUDIO_PATH,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        OUTPUT_VIDEO
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "="*50)
        print(f"SUCCESS! Video saved to: {OUTPUT_VIDEO}")
        print("="*50)
        
        # Cleanup temp file
        if os.path.exists(TEMP_GIF):
            os.remove(TEMP_GIF)
        
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] FFmpeg failed with error code {e.returncode}.")

# --- MAIN ---
if __name__ == "__main__":
    try:
        print("Loading files...")
        mesh = trimesh.load(MESH_PATH, process=False)
        if isinstance(mesh, trimesh.Scene): mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        verts, faces = mesh.vertices, mesh.faces

        anchors = get_fixed_anchors()
        ema_data = denormalize_with_fixed_anchors(load_and_reshape_motion(MOTION_PATH), STD_PATH, anchors)

        rig = FaceKitTongueRig(verts, faces, anchors, manual_bone_pos=MANUAL_BONE_COORDS)

        run_pipeline(verts, rig, ema_data, anchors, fps=EMA_FPS)

    except Exception as e:
        print(f"Error: {e}")
