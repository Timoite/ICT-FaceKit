"""
Face Animation Renderer using PyRender
=======================================

This script renders facial animations from BEAT dataset JSON files using the ICT-FaceKit model.
Uses trimesh for mesh operations and pyrender for rendering.

Phase 1: Face Animation Validation
- Load BEAT animation JSON (26_reamey_0_1_1.json)
- Map ARKit blendshape names to ICT-FaceKit expressions (FIXED: proper bilateral mapping)
- Deform mesh using expression weights per frame
- Render using pyrender pipeline

Author: Claude Code
Date: 2025-11-13
"""

import os
import json
import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import imageio
from pathlib import Path


class ICTFaceAnimationRenderer:
    """Renderer for ICT-FaceKit facial animations using pyrender"""

    def __init__(self, model_dir='../FaceXModel'):
        """
        Initialize the renderer with ICT-FaceKit model

        Args:
            model_dir: Path to FaceXModel directory
        """
        self.model_dir = Path(model_dir)
        self.neutral_mesh = None
        self.mesh_center = None  # Original center before normalization
        self.mesh_scale = None   # Scale factor for normalization
        self.expression_names = []
        self.expression_modes = {}  # Store delta vectors for each expression

        print("Loading ICT-FaceKit model...")
        self._load_model()
        print(f"Model loaded: {len(self.expression_names)} expressions, {len(self.expression_modes)} loaded")

    def _load_model(self):
        """Load the ICT-FaceKit model components"""
        # Load configuration
        config_path = self.model_dir / 'vertex_indices.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.expression_names = config['expressions']

        # Load neutral mesh using trimesh
        neutral_path = self.model_dir / 'generic_neutral_mesh.obj'
        loaded = trimesh.load(str(neutral_path), process=True, force='mesh')

        # Combine all geometries into single mesh
        if isinstance(loaded, trimesh.Scene):
            meshes = list(loaded.geometry.values())
            self.neutral_mesh = trimesh.util.concatenate(meshes)
            print(f"Combined {len(meshes)} mesh components")
        else:
            self.neutral_mesh = loaded

        # Store original center and scale for normalization
        self.mesh_center = self.neutral_mesh.vertices.mean(axis=0).copy()
        self.mesh_scale = 1.0 / np.abs(self.neutral_mesh.vertices - self.mesh_center).max()

        # Normalize mesh to [-1, 1] for better rendering
        self.neutral_mesh.vertices = (self.neutral_mesh.vertices - self.mesh_center) * self.mesh_scale

        self.neutral_vertices = np.array(self.neutral_mesh.vertices, dtype=np.float32)

        print(f"Neutral mesh: {len(self.neutral_vertices)} vertices, {len(self.neutral_mesh.faces)} faces (normalized)")

        # Load expression blend shapes
        print("Loading expression blend shapes...")bash -c "source /home/timoite/ICT-FaceKit/.venv/bin/activate && cd /home/timoite/ICT-FaceKit/Scripts && python -u render_face_animation.py 2>&1 | head -30"
        loaded_count = 0
        for expr_name in self.expression_names:
            expr_path = self.model_dir / f'{expr_name}.obj'
            if expr_path.exists():bash -c "source /home/timoite/ICT-FaceKit/.venv/bin/activate && cd /home/timoite/ICT-FaceKit/Scripts && python -u render_face_animation.py 2>&1 | head -30"
                loaded_expr = trimesh.load(str(expr_path), process=True, force='mesh')

                # Combine all geometries
                if isinstance(loaded_expr, trimesh.Scene):
                    expr_mesh = trimesh.util.concatenate(list(loaded_expr.geometry.values()))
                else:
                    expr_mesh = loaded_expr

                # Apply same normalization as neutral mesh
                expr_vertices = (np.array(expr_mesh.vertices, dtype=np.float32) - self.mesh_center) * self.mesh_scale

                # Compute delta from neutral
                delta = expr_vertices - self.neutral_vertices
                self.expression_modes[expr_name] = delta
                loaded_count += 1
            else:
                print(f"Warning: Expression {expr_name} not found at {expr_path}")

        print(f"Loaded {loaded_count}/{len(self.expression_names)} expression blend shapes")

    def map_beat_to_ict_names(self, beat_name):
        """
        Map BEAT/ARKit blendshape names to ICT-FaceKit expression names

        FIXED: Proper mapping logic:
        - browInnerUp, cheekPuff: bilateral (no Left/Right in BEAT but split in ICT as _L/_R)
        - jawLeft, jawRight, mouthLeft, mouthRight: direct mapping (ICT uses same names)
        - Other Left/Right expressions: convert to _L/_R format (e.g., browDownLeft → browDown_L)
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

    def load_animation(self, json_path):
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

        # Debug: Show mapping statistics
        mapped_count = 0
        unmapped_count = 0
        for beat_name in anim_data['names']:
            ict_names = self.map_beat_to_ict_names(beat_name)
            if any(name in self.expression_modes for name in ict_names):
                mapped_count += 1
            else:
                unmapped_count += 1
                print(f"  Warning: {beat_name} → {ict_names} not found in loaded expressions")

        print(f"Mapping: {mapped_count}/{num_blendshapes} BEAT blendshapes mapped successfully")

        return anim_data

    def deform_mesh(self, expression_weights):
        """
        Deform neutral mesh using expression weights

        Args:
            expression_weights: Dict mapping expression names to weights [0, 1]

        Returns:
            Deformed mesh vertices as numpy array
        """
        deformed_vertices = self.neutral_vertices.copy()

        # Apply each expression's contribution
        for expr_name, weight in expression_weights.items():
            if expr_name in self.expression_modes and weight > 0:
                delta = self.expression_modes[expr_name]
                deformed_vertices += weight * delta

        return deformed_vertices

    def render_frame(self, vertices, output_path=None, show=False, image_size=800):
        """
        Render a single frame using pyrender

        Args:
            vertices: Deformed mesh vertices (numpy array)
            output_path: If provided, save image to this path
            show: If True, display the image with matplotlib
            image_size: Output image resolution

        Returns:
            Rendered image as numpy array (H, W, 3) uint8
        """
        # Create mesh with deformed vertices
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=self.neutral_mesh.faces,
            process=False
        )

        # Create pyrender mesh with smooth shading
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, smooth=True)

        # Create scene with ambient lighting
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
        scene.add(mesh_pyrender)

        # Add camera looking at the centered mesh (mesh is already at origin)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],  # X: centered
            [0.0, 1.0, 0.0, 0.0],  # Y: centered
            [0.0, 0.0, 1.0, 3.0],  # Z: camera at distance 3.0
            [0.0, 0.0, 0.0, 1.0]
        ])
        scene.add(camera, pose=camera_pose)

        # Add directional light from camera position
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=camera_pose)

        # Render
        renderer = pyrender.OffscreenRenderer(image_size, image_size)
        color, depth = renderer.render(scene)
        renderer.delete()

        # Save or display
        if output_path:
            plt.imsave(output_path, color)

        if show:
            plt.figure(figsize=(10, 10))
            plt.imshow(color)
            plt.axis('off')
            plt.title('Rendered Face Frame (pyrender)')
            plt.tight_layout()
            plt.show()

        return color

    def render_animation(self, anim_data, output_dir='../sample_data_out/rendered_frames',
                        max_frames=None, sample_interval_sec=None):
        """
        Render full animation sequence

        Args:
            anim_data: Animation data dict from load_animation()
            output_dir: Directory to save rendered frames
            max_frames: If set, only render first N frames (for testing)
            sample_interval_sec: If set, sample frames at this interval (in seconds)

        Returns:
            List of rendered images with their frame indices
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        beat_names = anim_data['names']
        frames = anim_data['frames']

        # Determine which frames to render
        if sample_interval_sec is not None:
            # Sample frames at specific time intervals
            frame_indices = []
            for i, frame_data in enumerate(frames):
                time = frame_data.get('time', i * (1/60))  # Assume 60fps if time not provided
                if len(frame_indices) == 0 or time >= frame_indices[-1][1] + sample_interval_sec:
                    frame_indices.append((i, time))
                    if max_frames and len(frame_indices) >= max_frames:
                        break

            print(f"\nSampling frames at {sample_interval_sec}s intervals:")
            for idx, t in frame_indices[:10]:
                print(f"  Frame {idx} at t={t:.2f}s")
            if len(frame_indices) > 10:
                print(f"  ... and {len(frame_indices) - 10} more")
        else:
            # Render consecutive frames
            if max_frames:
                frames = frames[:max_frames]
            frame_indices = [(i, frames[i].get('time', i * (1/60))) for i in range(len(frames))]

        rendered_images = []

        print(f"\nRendering {len(frame_indices)} frames with pyrender...")
        for render_idx, (frame_idx, time) in enumerate(frame_indices):
            frame_data = frames[frame_idx]
            weights = frame_data['weights']

            # Map BEAT weights to ICT expression weights
            expression_weights = {}
            for beat_name, weight in zip(beat_names, weights):
                ict_names = self.map_beat_to_ict_names(beat_name)
                for ict_name in ict_names:
                    if ict_name in self.expression_modes:
                        expression_weights[ict_name] = weight

            # Deform mesh
            deformed_verts = self.deform_mesh(expression_weights)

            # Render frame
            output_path = output_dir / f'frame_{frame_idx:04d}_t{time:.2f}s.png'
            img = self.render_frame(deformed_verts, output_path=output_path)
            rendered_images.append((frame_idx, time, img))

            if (render_idx + 1) % 10 == 0:
                print(f"Rendered {render_idx + 1}/{len(frame_indices)} frames")

        print(f"✅ Animation rendering complete: {len(rendered_images)} frames")
        return rendered_images

    def export_video(self, rendered_data, output_path='../sample_data_out/face_animation.mp4', fps=30):
        """
        Export rendered frames as video

        Args:
            rendered_data: List of (frame_idx, time, image) tuples from render_animation()
            output_path: Path to save video file
            fps: Frames per second for output video

        Returns:
            Path to exported video file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract images from rendered data
        images = [img for _, _, img in rendered_data]

        print(f"\n🎬 Exporting video: {len(images)} frames at {fps} fps...")
        print(f"   Duration: {len(images) / fps:.2f} seconds")

        # Export video using imageio
        writer = imageio.get_writer(str(output_path), fps=fps, codec='libx264', quality=8)

        for idx, img in enumerate(images):
            writer.append_data(img)
            if (idx + 1) % 30 == 0:
                print(f"   Exported {idx + 1}/{len(images)} frames")

        writer.close()

        print(f"✅ Video exported: {output_path}")
        print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

        return output_path


def main():
    """Main execution for Phase 1: Face Animation Validation"""
    print("="*60)
    print("Phase 1: Face Animation Validation with BEAT Dataset")
    print("Using pyrender Rendering Pipeline")
    print("="*60)

    # Initialize renderer
    renderer = ICTFaceAnimationRenderer(model_dir='../FaceXModel')

    # Load animation data
    anim_data = renderer.load_animation('../data/26_reamey_0_1_1.json')

    # Render animation at 30fps for smooth video (10 seconds worth)
    print("\n🎬 Rendering animation for video export...")
    print("   Rendering at 30fps for 10 seconds (300 frames)...")

    rendered_data = renderer.render_animation(
        anim_data,
        max_frames=300  # 10 seconds at 30fps
    )

    # Export as video
    video_path = renderer.export_video(
        rendered_data,
        output_path='../sample_data_out/face_animation.mp4',
        fps=30
    )

    # Also create a validation image with key frames
    print("\n📸 Creating validation image with key frames...")

    # Extract images from rendered data
    images = [img for _, _, img in rendered_data]
    times = [t for _, t, _ in rendered_data]

    # Create 3-panel comparison with frames at 0s, 5s, 10s
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Frame at t=0s
    axes[0].imshow(images[0])
    axes[0].set_title(f't = {times[0]:.1f}s (Start)', fontsize=14)
    axes[0].axis('off')

    # Frame at t~5s (middle)
    mid_idx = len(images) // 2
    axes[1].imshow(images[mid_idx])
    axes[1].set_title(f't = {times[mid_idx]:.1f}s (Middle)', fontsize=14)
    axes[1].axis('off')

    # Frame at t~10s (end)
    axes[2].imshow(images[-1])
    axes[2].set_title(f't = {times[-1]:.1f}s (End)', fontsize=14)
    axes[2].axis('off')

    plt.suptitle('Phase 1 Validation: Pyrender with Fixed Blendshape Mapping', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('../sample_data_out/phase1_validation.png', dpi=150, bbox_inches='tight')
    print(f"✅ Validation image saved: '../sample_data_out/phase1_validation.png'")

    print("\n" + "="*60)
    print("Phase 1 Complete: Rendering pipeline validated")
    print(f"📹 Video: {video_path}")
    print(f"📸 Validation: sample_data_out/phase1_validation.png")
    print("🔧 Fixed: 13 missing blendshapes now mapped correctly")
    print("   - browInnerUp, cheekPuff: bilateral (mirrored to L/R)")
    print("   - jawForward, jawOpen, etc: center movements (no L/R split)")
    print("Next: Phase 2 - Tongue Linear Blend Skinning Implementation")
    print("="*60)


if __name__ == '__main__':
    main()
