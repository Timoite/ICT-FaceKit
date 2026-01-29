"""
Face Animation Renderer using Trimesh & Pyrender
================================================
Render Mode: Grey Mesh, Black Background.
Vertex Logic: Strict 1:1 correspondence.
"""

import os
import sys
import json
import numpy as np
import trimesh
import pyrender
import imageio
from pathlib import Path

# Add Scripts directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    from face_model_io_trimesh import load_face_model_trimesh
except ImportError:
    print("CRITICAL: 'face_model_io_trimesh.py' not found.")
    sys.exit(1)

# ============================================================================
# HELPER: Blendshape Mapping
# ============================================================================

def map_beat_to_ict_names(beat_name):
    truly_bilateral = ['browInnerUp', 'cheekPuff']
    if beat_name in truly_bilateral:
        return [f'{beat_name}_L', f'{beat_name}_R']
    direct_mapping = ['jawLeft', 'jawRight', 'mouthLeft', 'mouthRight']
    if beat_name in direct_mapping:
        return [beat_name]
    if beat_name.endswith('Left'):
        return [f'{beat_name[:-4]}_L']
    elif beat_name.endswith('Right'):
        return [f'{beat_name[:-5]}_R']
    return [beat_name]

def load_animation(json_path):
    with open(json_path, 'r') as f:
        anim_data = json.load(f)
    return anim_data

# ============================================================================
# NUMPY FACE MODEL
# ============================================================================

class FaceModelNumpy:
    def __init__(self, model_data):
        print("Initializing NumPy Face Model...")

        self.neutral_verts = model_data['neutral_verts']
        self.faces = model_data['faces']
        self.expression_names = model_data['expression_names']

        # Stack Deltas
        expressions_dict = model_data.get('expressions', {})
        n_verts = self.neutral_verts.shape[0]
        n_exprs = len(self.expression_names)

        self.deltas = np.zeros((n_exprs, n_verts, 3), dtype=np.float32)

        print(f"Stacking {n_exprs} expression deltas for {n_verts} vertices...")
        for idx, name in enumerate(self.expression_names):
            if name in expressions_dict:
                self.deltas[idx] = expressions_dict[name]

        # Colors (Grey)
        self.vertex_colors = np.ones((n_verts, 3), dtype=np.float32) * 0.5

    def deform(self, active_weights):
        w_vec = np.zeros(len(self.expression_names), dtype=np.float32)
        for name, weight in active_weights.items():
            if name in self.expression_names:
                idx = self.expression_names.index(name)
                w_vec[idx] = weight

        displacement = np.tensordot(w_vec, self.deltas, axes=(0, 0))
        return self.neutral_verts + displacement

# ============================================================================
# PYRENDER RENDERER (Stabilized)
# ============================================================================

class PyrenderRenderer:
    def __init__(self, neutral_verts, image_size=256):
        self.image_size = image_size
        self.r = pyrender.OffscreenRenderer(viewport_width=image_size, viewport_height=image_size)

        # Stabilize Camera Target
        self.neutral_center = neutral_verts.mean(axis=0)
        self.neutral_scale = np.max(np.abs(neutral_verts - self.neutral_center))
        if self.neutral_scale == 0: self.neutral_scale = 1.0

        # Camera
        self.camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.4],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)

        # Lights
        self.light_main = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        self.light_pose_main = np.eye(4)
        self.light_pose_main[:3, 3] = [0, 1, 2]

        self.light_fill = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        self.light_pose_fill = np.eye(4)
        self.light_pose_fill[:3, 3] = [0, 0, 2]

    def render(self, verts, faces, colors):
        # Normalize
        norm_verts = (verts - self.neutral_center) / self.neutral_scale

        # Mesh
        tm = trimesh.Trimesh(vertices=norm_verts, faces=faces, process=False)
        tm.visual.vertex_colors = colors

        mesh = pyrender.Mesh.from_trimesh(tm, smooth=True, wireframe=False)
        if mesh.primitives:
            for p in mesh.primitives:
                p.material.doubleSided = True

        # Scene
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0], ambient_light=[0.4, 0.4, 0.4])

        scene.add(mesh)
        scene.add(self.camera, pose=self.camera_pose)
        scene.add(self.light_main, pose=self.light_pose_main)
        scene.add(self.light_fill, pose=self.light_pose_fill)

        color, _ = self.r.render(scene)
        return color

    def close(self):
        self.r.delete()

def render_animation(face_model, anim_data, renderer, output_dir=None, max_frames=None, video_writer=None):
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    beat_names = anim_data['names']
    frames = anim_data['frames'][:max_frames] if max_frames else anim_data['frames']

    for i, frame_data in enumerate(frames):
        weights = frame_data['weights']

        active_weights = {}
        for beat_name, weight in zip(beat_names, weights):
            ict_names = map_beat_to_ict_names(beat_name)
            for ict_name in ict_names:
                active_weights[ict_name] = weight

        deformed_verts = face_model.deform(active_weights)
        img = renderer.render(deformed_verts, face_model.faces, face_model.vertex_colors)

        if video_writer:
            video_writer.write(img.tobytes())
        elif output_dir:
            imageio.imwrite(output_dir / f'frame_{i:04d}.png', img)

if __name__ == '__main__':
    print("This is a module. Run 'render_batch_speaker_trimesh.py'")
