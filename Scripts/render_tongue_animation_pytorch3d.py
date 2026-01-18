"""
Tongue Animation Renderer using PyTorch3D
=========================================

This script renders tongue animations using the ICTTongueModel and PyTorch3D.
"""

import os
import sys
import torch
import imageio
import numpy as np
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm

# PyTorch3D imports
from pytorch3d.io import load_obj
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    BlendParams
)

# Add Scripts directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from ict_tongue_model import ICTTongueModel, load_tongue_animation_data

class PyTorch3DTongueRenderer:
    """Minimal PyTorch3D renderer for tongue animations"""

    def __init__(self, tongue_model, faces, device=None, image_size=800, visualize_armature=True):
        """
        Initialize PyTorch3D renderer

        Args:
            tongue_model: ICTTongueModel instance
            faces: Tensor (F, 3) of face indices
            device: Device for rendering
            image_size: Output image resolution
            visualize_armature: Whether to render control points
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.tongue_model = tongue_model
        self.faces = faces.to(device)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.image_size = image_size
        self.visualize_armature = visualize_armature

        self._setup_renderer()
        print(f"PyTorch3D Tongue Renderer initialized on {self.device} (Armature Viz: {self.visualize_armature})")

    def _setup_renderer(self):
        """Setup PyTorch3D rendering components"""

        # Compute mesh center and scale for proper viewing
        neutral_verts = self.tongue_model.rest_verts
        mesh_center = neutral_verts.mean(dim=0)
        mesh_scale = (neutral_verts - mesh_center).abs().max()

        self.mesh_center = mesh_center
        self.mesh_scale = mesh_scale

        print(f"  Mesh center: ({mesh_center[0]:.2f}, {mesh_center[1]:.2f}, {mesh_center[2]:.2f})")
        print(f"  Mesh scale: {mesh_scale:.2f}")

        if self.visualize_armature:
            # Create template sphere for control points
            # Normalized mesh is roughly within [-1, 1] box.
            self.template_sphere = ico_sphere(level=1, device=self.device)
            self.sphere_radius = 0.15 # Very thick to ensure visibility
            self.template_sphere.verts_list()[0] *= self.sphere_radius

        # Camera: positioned to view the centered and normalized tongue
        # Tongue is small, so we might need to adjust distance
        R, T = look_at_view_transform(
            dist=6.0,
            elev=15,        # Slight elevation to see top of tongue
            azim=30         # Slight angle
        )

        self.cameras = FoVPerspectiveCameras(
            device=self.device,
            R=R,
            T=T,
            fov=45
        )

        # Lights
        self.lights = PointLights(
            device=self.device,
            location=[[2.0, 2.0, 2.0]],
            ambient_color=[[0.4, 0.4, 0.4]],
            diffuse_color=[[0.6, 0.6, 0.6]],
            specular_color=[[0.1, 0.1, 0.1]]
        )

        # Rasterization settings
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1, # Disable transparency to fix afterimage/ghosting
        )

        # Renderer
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=self.cameras,
                lights=self.lights
                # Removed BlendParams for opaque rendering
            )
        )

    def render_batch(self, deformed_verts_batch, control_points_batch=None):
        """
        Render a batch of deformed vertices

        Args:
            deformed_verts_batch: (B, V, 3)
            control_points_batch: (B, 4, 3) Optional

        Returns:
            Rendered images tensor (B, H, W, 4)
        """
        batch_size = deformed_verts_batch.shape[0]

        # Normalize meshes
        verts_normalized = (deformed_verts_batch - self.mesh_center) / self.mesh_scale

        # Create Meshes object
        # Faces need to be replicated for batch
        faces_batch = self.faces.unsqueeze(0).expand(batch_size, -1, -1)
        tongue_meshes = Meshes(verts=verts_normalized, faces=faces_batch)

        # Add texture (greyish for better contrast with colored spheres)
        verts_rgb = torch.ones_like(tongue_meshes.verts_packed()) * torch.tensor([0.6, 0.6, 0.6], device=self.device)
        tongue_meshes.textures = TexturesVertex(verts_features=[verts_rgb[tongue_meshes.verts_packed_to_mesh_idx() == i]
                                                   for i in range(len(tongue_meshes))])

        if self.visualize_armature and control_points_batch is not None:
            # Normalize control points
            cp_normalized = (control_points_batch - self.mesh_center) / self.mesh_scale

            spheres_list = []
            # Colors for 4 control points: Root(Cyan), Back(Blue), Mid(Green), Tip(Yellow)
            colors = [
                [0.0, 1.0, 1.0], # Root (Cyan)
                [0.2, 0.2, 0.8], # Back
                [0.2, 0.8, 0.2], # Mid
                [0.8, 0.8, 0.2]  # Tip
            ]

            for i in range(4):
                pos = cp_normalized[:, i, :] # (B, 3)

                # Create batch of spheres
                sphere_verts = self.template_sphere.verts_list()[0]
                sphere_faces = self.template_sphere.faces_list()[0]

                b_sphere_verts = sphere_verts.unsqueeze(0).expand(batch_size, -1, -1) + pos.unsqueeze(1)
                b_sphere_faces = sphere_faces.unsqueeze(0).expand(batch_size, -1, -1)

                sphere_mesh = Meshes(verts=b_sphere_verts, faces=b_sphere_faces)

                # Color
                c = torch.tensor(colors[i], device=self.device)
                s_rgb = torch.ones_like(sphere_mesh.verts_packed()) * c
                sphere_mesh.textures = TexturesVertex(verts_features=[s_rgb[sphere_mesh.verts_packed_to_mesh_idx() == k] for k in range(len(sphere_mesh))])

                spheres_list.append(sphere_mesh)

            # Join
            combined_mesh = join_meshes_as_scene([tongue_meshes] + spheres_list)
            images = self.renderer(combined_mesh)
        else:
            images = self.renderer(tongue_meshes)

        return images

def render_tongue_animation(
    tongue_model,
    control_points_seq,
    renderer,
    output_dir,
    batch_size=1 # Render frame by frame to avoid batching artifacts
):
    """
    Render tongue animation sequence
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_frames = control_points_seq.shape[0]
    print(f"\nRendering {num_frames} frames (batch_size={batch_size})...")

    rendered_images = []

    for batch_start in tqdm(range(0, num_frames, batch_size), desc="Rendering batches"):
        batch_end = min(batch_start + batch_size, num_frames)

        # Get batch control points
        batch_ctrl = control_points_seq[batch_start:batch_end].to(renderer.device)

        # Deform
        deformed_verts = tongue_model(batch_ctrl)

        # Render
        images = renderer.render_batch(deformed_verts, control_points_batch=batch_ctrl)

        # Save
        for i, img_tensor in enumerate(images):
            frame_idx = batch_start + i

            img = img_tensor[..., :3].cpu().numpy()
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)

            output_path = output_dir / f'frame_{frame_idx:04d}.png'
            imageio.imwrite(output_path, img)

            rendered_images.append(img)

    return rendered_images

def export_video(output_path, frames_dir, fps=60):
    """Export video using ffmpeg"""
    output_path = Path(output_path)
    frames_dir = Path(frames_dir)

    print(f"\nExporting video to {output_path}...")

    if not shutil.which('ffmpeg'):
        print("FFmpeg not found.")
        return

    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-i', str(frames_dir / 'frame_%04d.png'),
        '-c:v', 'libopenh264',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video exported successfully.")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e}")

def main():
    print("="*60)
    print("PyTorch3D Tongue Animation Renderer")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / 'data' / 'blender_export_lbs'

    # 1. Load Tongue Model
    print("\nPhase 1: Loading Tongue Model")
    model = ICTTongueModel(
        rest_verts_path=str(data_dir / "tongue_verts_rest.npy"),
        weights_path=str(data_dir / "tongue_weights.npy"),
        rest_matrices_path=str(data_dir / "bone_rest_matrices.npy"),
        rest_tails_path=str(data_dir / "bone_rest_tails.npy"),
        device=device
    )

    # 2. Load Faces from OBJ
    print("\nPhase 2: Loading Mesh Topology")
    obj_path = project_root / 'Blender' / 'seperated_tongue.obj'
    verts, faces, aux = load_obj(str(obj_path))
    faces_idx = faces.verts_idx.to(device)
    print(f"Loaded {len(faces_idx)} faces from {obj_path.name}")

    # Verify vertex count matches
    if len(verts) != model.num_verts:
        print(f"Warning: OBJ has {len(verts)} verts, Model has {model.num_verts} verts.")
        # If mismatch is small or order is same, might be fine.
        # But if mismatch, rendering might look weird or crash.
        # Assuming they match based on previous checks.

    # 3. Load Animation
    print("\nPhase 3: Loading Animation")
    npy_path = project_root / 'data' / '26_reamey_0_1_1.npy'
    mu_path = project_root / 'data' / 'JW13_4points_mu.npy'
    std_path = project_root / 'data' / 'JW13_4points_std.npy'

    # Construct rest control points
    p0 = model.rest_heads[0]
    p1 = model.rest_heads[1]
    p2 = model.rest_heads[2]
    p3 = model.rest_tails[2]
    rest_ctrl = torch.stack([p0, p1, p2, p3])

    # Scale down animation by 0.2 for realism
    control_points_seq = load_tongue_animation_data(
        str(npy_path), str(mu_path), str(std_path), rest_ctrl,
        animation_scale=0.09
    )
    print(f"Loaded {len(control_points_seq)} frames of animation")

    # 4. Initialize Renderer
    print("\nPhase 4: Initializing Renderer")
    renderer = PyTorch3DTongueRenderer(model, faces_idx, device=device)

    # 5. Render
    print("\nPhase 5: Rendering")
    frames_dir = project_root / 'sample_data_out' / 'rendered_frames_tongue'
    render_tongue_animation(
        model,
        control_points_seq,
        renderer,
        frames_dir,
        batch_size=1 # Render frame by frame
    )

    # 6. Export Video
    print("\nPhase 6: Exporting Video")
    video_path = project_root / 'sample_data_out' / 'tongue_animation_pytorch3d_scaled.mp4'
    export_video(video_path, frames_dir)

    print("\nDone!")

if __name__ == "__main__":
    main()
