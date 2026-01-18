"""
Face Animation Renderer using PyTorch3D
========================================

This script renders facial animations from BEAT dataset JSON files using the ICT-FaceKit model
with PyTorch3D for GPU-accelerated rendering.

Reuses blendshape mapping logic from render_face_animation.py with PyTorch3D backend.
"""

import os
import sys
import json
import subprocess
import shutil
import torch
import imageio
import numpy as np
from pathlib import Path
from tqdm import tqdm

# PyTorch3D imports
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)

# Try importing MeshRasterizerOpenGL
try:
    from pytorch3d.renderer.opengl import MeshRasterizerOpenGL
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False
    print("MeshRasterizerOpenGL not found, using MeshRasterizer")

# Add Scripts directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from face_model_io_pytorch3d import load_face_model_pytorch3d
from ict_face_model_pytorch3d import FaceModelPyTorch3D


# ============================================================================
# COPIED FROM render_face_animation.py - Blendshape Mapping Logic
# ============================================================================

def map_beat_to_ict_names(beat_name):
    """
    Map BEAT/ARKit blendshape names to ICT-FaceKit expression names

    FIXED: Proper mapping logic:
    - browInnerUp, cheekPuff: bilateral (no Left/Right in BEAT but split in ICT as _L/_R)
    - jawLeft, jawRight, mouthLeft, mouthRight: direct mapping (ICT uses same names)
    - Other Left/Right expressions: convert to _L/_R format (e.g., browDownLeft -> browDown_L)
    - Center movements (jawForward, jawOpen, etc): map directly

    Args:
        beat_name: BEAT blendshape name (e.g., 'browDownLeft', 'browInnerUp', 'jawOpen')

    Returns:
        List of ICT expression names (e.g., ['browDown_L'] or ['browInnerUp_L', 'browInnerUp_R'])
    """
    # ONLY these two are bilateral in BEAT but split in ICT-FaceKit
    truly_bilateral = ['browInnerUp', 'cheekPuff']

    if beat_name in truly_bilateral:
        # Apply to both left and right
        return [f'{beat_name}_L', f'{beat_name}_R']

    # These expressions keep their Left/Right suffix in ICT (no conversion to _L/_R)
    direct_mapping = ['jawLeft', 'jawRight', 'mouthLeft', 'mouthRight']
    if beat_name in direct_mapping:
        return [beat_name]

    # Handle Left/Right suffixes in BEAT names (convert to _L/_R)
    if beat_name.endswith('Left'):
        base_name = beat_name[:-4]  # Remove 'Left'
        return [f'{base_name}_L']
    elif beat_name.endswith('Right'):
        base_name = beat_name[:-5]  # Remove 'Right'
        return [f'{base_name}_R']

    # Center movements (jawForward, jawOpen, etc) map directly
    return [beat_name]


def load_animation(json_path):
    """
    Load animation data from BEAT JSON file

    Args:
        json_path: Path to animation JSON file

    Returns:
        Dict with 'names', 'frames' containing animation data
    """
    print(f"Loading animation from {json_path}...")
    with open(json_path, 'r') as f:
        anim_data = json.load(f)

    num_frames = len(anim_data['frames'])
    num_blendshapes = len(anim_data['names'])
    print(f"Animation: {num_frames} frames, {num_blendshapes} blendshapes")

    return anim_data


# ============================================================================
# NEW: PyTorch3D Rendering Pipeline
# ============================================================================

class PyTorch3DRenderer:
    """Minimal PyTorch3D renderer for facial animations"""

    def __init__(self, face_model, device=None, image_size=256):
        """
        Initialize PyTorch3D renderer

        Args:
            face_model: FaceModelPyTorch3D instance
            device: Device for rendering (str like 'cuda'/'cpu' or torch.device). If None, defaults to 'cuda' if available else 'cpu'
            image_size: Output image resolution
        """
        # Default to CUDA if available, otherwise CPU
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.face_model = face_model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.image_size = image_size

        # Setup camera, lights, and renderer
        self._setup_renderer()

        # Precompute vertex colors
        self.vertex_colors = self._compute_vertex_colors()

        print(f"PyTorch3D Renderer initialized on {self.device}")

    def _setup_renderer(self):
        """Setup PyTorch3D rendering components"""

        # Compute mesh center and scale for proper viewing
        neutral_verts = self.face_model.neutral_verts
        mesh_center = neutral_verts.mean(dim=0)
        mesh_scale = (neutral_verts - mesh_center).abs().max()

        # Store for mesh normalization
        self.mesh_center = mesh_center
        self.mesh_scale = mesh_scale

        print(f"  Mesh center: ({mesh_center[0]:.2f}, {mesh_center[1]:.2f}, {mesh_center[2]:.2f})")
        print(f"  Mesh scale: {mesh_scale:.2f}")

        # Camera: positioned to view the centered and normalized face
        # Look at origin after we center the mesh
        R, T = look_at_view_transform(
            dist=1.5,      # Distance from object (increased to see full face)
            elev=0,        # Elevation angle
            azim=0         # Azimuth angle (0 for front view)
        )

        self.cameras = FoVPerspectiveCameras(
            device=self.device,
            R=R,
            T=T,
            fov=45  # Narrower FOV for better face framing
        )

        # Lights: Enhanced single light with proper 3D shading
        # Using one well-positioned light with good ambient/diffuse/specular balance
        self.lights = PointLights(
            device=self.device,
            location=[[0.0, 0.0, 3.0]],       # Frontal light
            ambient_color=[[0.6, 0.6, 0.6]],  # Higher ambient for softer look
            diffuse_color=[[0.4, 0.4, 0.4]],  # Moderate diffuse
            specular_color=[[0.05, 0.05, 0.05]] # Very low specular to reduce shininess
        )

        # Rasterization settings
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Renderer
        if HAS_OPENGL:
            print("Using MeshRasterizerOpenGL for faster rendering")
            rasterizer = MeshRasterizerOpenGL(
                cameras=self.cameras,
                raster_settings=raster_settings
            )
        else:
            rasterizer = MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            )

        self.renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=SoftPhongShader(
                device=self.device,
                cameras=self.cameras,
                lights=self.lights
            )
        )

    def _compute_vertex_colors(self):
        """Compute vertex colors based on face materials"""
        if not hasattr(self.face_model, 'materials_idx') or \
           self.face_model.materials_idx is None or \
           not hasattr(self.face_model, 'material_colors') or \
           self.face_model.material_colors is None:
            print("Warning: No material information found. Using default skin tone.")
            return None

        print("Computing vertex colors from materials...")

        # Map material names to colors
        # We assume keys in material_colors dict match the indices in materials_idx
        material_names = list(self.face_model.material_colors.keys())

        # Define overrides for better appearance
        overrides = {
            'M_Face': [0.85, 0.72, 0.65],       # Skin tone
            'M_BackHead': [0.2, 0.2, 0.2],      # Darker
            'M_GumsTongue': [0.8, 0.2, 0.2],    # Redder tongue/gums
            'M_Teeth': [0.9, 0.9, 0.85],        # White-ish teeth
            'M_ScleraLeft': [0.95, 0.95, 0.95], # White eyes
            'M_ScleraRight': [0.95, 0.95, 0.95],
            'M_IrisLeft': [0.2, 0.3, 0.4],      # Blue-ish eyes
            'M_IrisRight': [0.2, 0.3, 0.4],
            'M_LacrimalFluid': [0.9, 0.9, 0.9],
            'M_EyeBlend': [0.85, 0.72, 0.65],   # Skin blend
            'M_EyeOcclusion': [0.95, 0.95, 0.95],
            'M_EyeLashes': [0.1, 0.1, 0.1]      # Black
        }

        colors_list = []
        for name in material_names:
            if name in overrides:
                colors_list.append(overrides[name])
            else:
                # Use value from file
                val = self.face_model.material_colors[name]
                # If it's a dict (Kd, Ks etc), extract Kd
                if isinstance(val, dict) and 'Kd' in val:
                    val = val['Kd']
                # Ensure it's a list/tensor of 3 floats
                if isinstance(val, torch.Tensor):
                    val = val.tolist()
                colors_list.append(val[:3]) # RGB

        material_palette = torch.tensor(colors_list, device=self.device, dtype=torch.float32)

        # Map face indices to colors: (F, 3)
        face_colors = material_palette[self.face_model.materials_idx]

        # Compute vertex colors by averaging face colors
        V = self.face_model.neutral_verts.shape[0]
        vertex_colors = torch.zeros((V, 3), device=self.device, dtype=torch.float32)
        vertex_counts = torch.zeros((V, 1), device=self.device, dtype=torch.float32)

        faces = self.face_model.faces

        # Flatten faces and repeat colors
        faces_flat = faces.view(-1) # (F*3)
        colors_repeated = face_colors.repeat_interleave(3, dim=0) # (F*3, 3)

        vertex_colors.index_add_(0, faces_flat, colors_repeated)

        ones = torch.ones((faces_flat.shape[0], 1), device=self.device)
        vertex_counts.index_add_(0, faces_flat, ones)

        # Avoid divide by zero
        vertex_counts[vertex_counts == 0] = 1.0
        vertex_colors = vertex_colors / vertex_counts

        return vertex_colors

    def render_batch(self, meshes):
        """
        Render a batch of meshes

        Args:
            meshes: PyTorch3D Meshes object with B meshes

        Returns:
            Rendered images tensor (B, H, W, 4) with RGBA
        """
        # Normalize meshes: center and scale to unit size
        verts_list = meshes.verts_list()
        normalized_verts = []

        for verts in verts_list:
            # Center at origin
            verts_centered = verts - self.mesh_center
            # Scale to roughly unit size
            verts_normalized = verts_centered / self.mesh_scale
            normalized_verts.append(verts_normalized)

        # Create normalized meshes
        meshes_normalized = meshes.update_padded(torch.stack([v for v in normalized_verts]))

        # Add texture to meshes
        verts_packed = meshes_normalized.verts_packed()

        # Use precomputed vertex colors if available
        if hasattr(self, 'vertex_colors') and self.vertex_colors is not None:
            # Repeat vertex colors for each mesh in batch
            # vertex_colors is (V, 3). verts_packed is (B*V, 3)
            # We need to repeat it B times.
            batch_size = len(meshes)
            verts_rgb = self.vertex_colors.repeat(batch_size, 1)
        else:
            # Create per-vertex colors (skin tone)
            verts_rgb = torch.ones_like(verts_packed) * torch.tensor([0.8, 0.7, 0.6], device=self.device)

        textures = TexturesVertex(verts_features=[verts_rgb[meshes_normalized.verts_packed_to_mesh_idx() == i]
                                                   for i in range(len(meshes_normalized))])
        meshes_normalized.textures = textures

        # Render
        images = self.renderer(meshes_normalized)

        return images

def render_animation(face_model, anim_data, renderer, output_dir=None, max_frames=None, batch_size=1, video_writer=None):
    """
    Render facial animation using the specified renderer

    Args:
        face_model: FaceModelPyTorch3D instance
        anim_data: Animation data from load_animation()
        renderer: PyTorch3DRenderer instance
        output_dir: Directory to save rendered frames (optional if video_writer is provided)
        max_frames: If set, only render first N frames
        batch_size: Number of frames to render in one batch
        video_writer: Optional file-like object to write raw frames to (e.g. ffmpeg stdin)

    Returns:
        List of rendered images (empty if video_writer is used to save memory)
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    beat_names = anim_data['names']
    frames = anim_data['frames'][:max_frames] if max_frames else anim_data['frames']

    num_frames = len(frames)
    print(f"\nRendering {num_frames} frames (batch_size={batch_size})...")

    rendered_images = []
    expression_name_to_idx = {name: idx for idx, name in enumerate(face_model.expression_names)}

    # Process in batches
    for batch_start in tqdm(range(0, num_frames, batch_size), desc="Rendering batches"):
        batch_end = min(batch_start + batch_size, num_frames)
        batch_frames = frames[batch_start:batch_end]

        # Build batch expression weights
        batch_weights = []
        for frame_data in batch_frames:
            weights = frame_data['weights']

            # Map BEAT weights to ICT expression weights
            expression_weights = {}
            for beat_name, weight in zip(beat_names, weights):
                ict_names = map_beat_to_ict_names(beat_name)
                for ict_name in ict_names:
                    if ict_name in expression_name_to_idx:
                        expression_weights[ict_name] = weight

            batch_weights.append(expression_weights)

        # Deform batch on GPU
        meshes = face_model.get_meshes_batch(batch_weights)

        # Render batch on GPU
        images = renderer.render_batch(meshes)

        # Save frames (transfer to CPU only here)
        # OPTIMIZATION: Process whole batch at once
        # Move to CPU and convert to uint8 in one go
        images_cpu = images[..., :3].cpu().numpy()
        images_cpu = np.clip(images_cpu, 0, 1)
        images_cpu = (images_cpu * 255).astype(np.uint8)

        for i, img in enumerate(images_cpu):
            frame_idx = batch_start + i

            if video_writer:
                video_writer.write(img.tobytes())
            elif output_dir:
                # Save
                output_path = output_dir / f'frame_{frame_idx:04d}.png'
                imageio.imwrite(output_path, img)
                rendered_images.append(img)
            else:
                rendered_images.append(img)

    if video_writer:
        print(f"Streamed {num_frames} frames to video writer")
    else:
        print(f"Rendered {len(rendered_images)} frames")

    return rendered_images


def export_video(image_list, output_path='../sample_data_out/face_animation_pytorch3d.mp4', fps=60, frames_dir=None):
    """
    Export rendered frames as video using FFmpeg directly

    Args:
        image_list: List of images (numpy arrays) - unused in ffmpeg mode but kept for signature compatibility
        output_path: Path to save video file
        fps: Frames per second
        frames_dir: Directory containing the frames. If None, defaults to output_path.parent / 'rendered_frames_pytorch3d'

    Returns:
        Path to exported video file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if frames_dir is None:
        # Directory containing frames
        frames_dir = output_path.parent / 'rendered_frames_pytorch3d'
    else:
        frames_dir = Path(frames_dir)

    print(f"\nExporting video using FFmpeg: {len(image_list)} frames at {fps} fps...")

    if not shutil.which('ffmpeg'):
        print("FFmpeg not found in system path. Cannot export video.")
        return output_path

    # FFmpeg command
    # -y: overwrite output
    # -framerate: input frame rate
    # -i: input file pattern
    # -c:v libopenh264: video codec (available in non-GPL ffmpeg)
    # -pix_fmt yuv420p: pixel format for compatibility
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
        # Run ffmpeg
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video exported: {output_path}")
        if output_path.exists():
            print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e}")
        print(f"   Stderr: {e.stderr.decode()}")

    return output_path


def main():
    """Main execution: Load model, render animation, export video"""
    print("="*60)
    print("PyTorch3D Face Animation Renderer")
    print("GPU-Accelerated Rendering Pipeline")
    print("="*60)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_dir = project_root / 'FaceXModel'
    anim_path = project_root / 'data' / '26_reamey_0_2_2.json'

    # Load face model (Phase 1)
    print("\n" + "="*60)
    print("Phase 1: Loading ICT Face Model with PyTorch3D")
    print("="*60)
    # Disable loading identities to save memory and time
    model_data = load_face_model_pytorch3d(str(model_dir), device=str(device), load_identities=False)

    # Initialize face model (Phase 2)
    print("\n" + "="*60)
    print("Phase 2: Initializing Face Model")
    print("="*60)
    face_model = FaceModelPyTorch3D(model_data, device=device)
    print(face_model)

    # Load animation (COPIED logic)
    print("\n" + "="*60)
    print("Phase 3: Loading Animation Data")
    print("="*60)
    anim_data = load_animation(str(anim_path))

    # Verify blendshape mapping
    print("\nVerifying blendshape mapping...")
    expression_name_to_idx = {name: idx for idx, name in enumerate(face_model.expression_names)}
    mapped_count = 0
    unmapped_count = 0

    for beat_name in anim_data['names']:
        ict_names = map_beat_to_ict_names(beat_name)
        if any(name in expression_name_to_idx for name in ict_names):
            mapped_count += 1
        else:
            unmapped_count += 1

    print(f"Mapped: {mapped_count}/{len(anim_data['names'])} blendshapes")
    if unmapped_count > 0:
        print(f"Unmapped: {unmapped_count} blendshapes (will be ignored)")

    # Initialize renderer
    print("\n" + "="*60)
    print("Phase 3: Initializing PyTorch3D Renderer")
    print("="*60)
    renderer = PyTorch3DRenderer(face_model, device=device, image_size=800)

    # Render animation
    print("\n" + "="*60)
    print("Phase 3: Rendering Animation")
    print("="*60)

    rendered_images = render_animation(
        face_model=face_model,
        anim_data=anim_data,
        renderer=renderer,
        output_dir=str(project_root / 'sample_data_out' / 'rendered_frames_pytorch3d'),
        max_frames=None,  # Render all frames
        batch_size=100    # Adjust based on GPU memory
    )

    # Export video
    print("\n" + "="*60)
    print("Phase 3: Exporting Video")
    print("="*60)

    video_path = export_video(
        rendered_images,
        output_path=str(project_root / 'sample_data_out' / 'face_animation_pytorch3d.mp4'),
        fps=60
    )

    # Summary
    print("\n" + "="*60)
    print("Rendering Complete!")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Frames: {project_root / 'sample_data_out' / 'rendered_frames_pytorch3d'}")
    print(f"Total frames: {len(rendered_images)}")
    print(f"Duration: {len(rendered_images) / 60:.2f} seconds")
    print(f"Device: {device}")
    print("="*60)


if __name__ == '__main__':
    main()
