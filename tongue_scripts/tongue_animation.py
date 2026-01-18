#!/usr/bin/env python3
"""Standalone tongue control-point animation and rigging preview."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import trimesh
from scipy.spatial.distance import cdist

# --- FILE PATHS -------------------------------------------------------------
BASE_DIR = Path(__file__).parent
PATH_TONGUE = BASE_DIR / 'seperated_tongue.obj'
PATH_DATA = BASE_DIR / '26/npy/26_reamey_0_1_1.npy'
PATH_STD = BASE_DIR / 'normalising_vectors/JW13_4points_std.npy'
OUTPUT_GIF = BASE_DIR / 'exports' / 'tongue_animation.gif'

# --- CONTROL POINTS & PARAMETERS -------------------------------------------
CONTROL_POINTS_REST = np.array([
    [0.0, -6.12435, -3.09399 - 0.1],  # T4 (Back)
    [0.0, -7.69406, -3.06964 - 0.1],  # T3
    [0.0, -8.78341, -3.29171 - 0.1],  # T2
    [0.0, -9.58909, -3.71498 - 0.1],  # T1 (Tip)
], dtype=np.float32)
CP_NAMES = ['T4', 'T3', 'T2', 'T1']

TONGUE_DIMS = 8  # Four points * (Y, Z)
STD_SCALE = 0.2
INFLUENCE_RADIUS = 4.0  # Wendland kernel falloff

# --- RIGGING UTILITIES ------------------------------------------------------

def load_tongue_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f'Expected Trimesh, received {type(mesh).__name__}')
    rot_matrix = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
    mesh.apply_transform(rot_matrix)
    return mesh


def compute_weights(mesh: trimesh.Trimesh, control_points: np.ndarray) -> np.ndarray:
    # Calculate distance from every vertex to every control point

    dists = cdist(mesh.vertices, control_points)

    # Inverse Distance Weighting (Power of 3 for smooth but local control)
    # Add epsilon to avoid divide-by-zero

    weights = 1.0 / (dists + 1e-6)**3

    weights /= weights.sum(axis=1, keepdims=True) # Normalize
    return weights


def apply_lbs(vertices: np.ndarray,
              rest_pts: np.ndarray,
              target_pts: np.ndarray,
              weights: np.ndarray) -> np.ndarray:
    deltas = target_pts - rest_pts
    displacement = weights @ deltas
    return vertices + displacement


# --- DATA LOADING -----------------------------------------------------------

def load_tongue_trajectory(path_data: Path, path_std: Path) -> np.ndarray:
    raw_data = np.load(path_data)
    if raw_data.shape[1] < TONGUE_DIMS:
        raise ValueError(f'Expected at least {TONGUE_DIMS} columns, found {raw_data.shape[1]}')

    std_vec = np.load(path_std).reshape(-1)[:TONGUE_DIMS] * STD_SCALE
    rest_mean = CONTROL_POINTS_REST[:, 1:3].reshape(-1)

    tongue_zscores = raw_data[:, :TONGUE_DIMS]
    tongue_real = (tongue_zscores * std_vec) + rest_mean
    trajectory = tongue_real.reshape(-1, 4, 2)
    return trajectory


def deform_mesh_sequence(mesh: trimesh.Trimesh,
                         trajectory: np.ndarray,
                         weights: np.ndarray) -> np.ndarray:
    rigged = []
    rest = CONTROL_POINTS_REST.copy()
    for frame in trajectory:
        target = rest.copy()
        target[:, 1:3] = frame
        rigged.append(apply_lbs(mesh.vertices, rest, target, weights))
    return np.array(rigged)


# --- VISUALIZATION ---------------------------------------------------------

def animate_tongue(
    trajectory: np.ndarray,
    rigged_vertices: np.ndarray,
    interval_ms: int = 33,
    tail_length: int = 10,
    save_path: Optional[Path] = None,
):
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh_scatter = ax.scatter([], [], s=4, color='gray', alpha=0.35, label='Tongue Mesh (YZ)')
    line, = ax.plot([], [], 'b-o', lw=2, label='Tongue CPs (T4->T1)')
    scatter = ax.scatter([], [], color='#1f77b4', zorder=3)
    tail_line, = ax.plot([], [], 'r-', alpha=0.3, label='Tip Trail')

    y_coords = trajectory[:, :, 0]
    z_coords = trajectory[:, :, 1]
    mesh_y = rigged_vertices[:, :, 1]
    mesh_z = rigged_vertices[:, :, 2]
    margin = 3
    ax.set_xlim(min(np.min(y_coords), np.min(mesh_y)) - margin,
                max(np.max(y_coords), np.max(mesh_y)) + margin)
    ax.set_ylim(min(np.min(z_coords), np.min(mesh_z)) - margin,
                max(np.max(z_coords), np.max(mesh_z)) + margin)
    ax.set_xlabel('Y (Posterior -> Anterior)')
    ax.set_ylabel('Z (Inferior -> Superior)')
    ax.set_title('Tongue Rig (YZ Sagittal View)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_aspect('equal', adjustable='box')

    def init():
        mesh_scatter.set_offsets(np.zeros((1, 2)))
        line.set_data([], [])
        scatter.set_offsets(np.zeros((4, 2)))
        tail_line.set_data([], [])
        return mesh_scatter, line, scatter, tail_line

    def update(frame_idx):
        verts = rigged_vertices[frame_idx]
        mesh_scatter.set_offsets(verts[:, [1, 2]])

        pts = trajectory[frame_idx]
        line.set_data(pts[:, 0], pts[:, 1])
        scatter.set_offsets(pts)

        start = max(0, frame_idx - tail_length)
        tip_trail = trajectory[start:frame_idx + 1, -1]
        tail_line.set_data(tip_trail[:, 0], tip_trail[:, 1])
        return mesh_scatter, line, scatter, tail_line

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=trajectory.shape[0],
        init_func=init,
        interval=interval_ms,
        blit=True,
        repeat=True,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fps = max(1, int(round(1000 / interval_ms)))
        print(f'Saving GIF to {save_path} ({fps} fps)')
        ani.save(save_path, writer=animation.PillowWriter(fps=fps))

    plt.show()
    return ani


# --- ENTRYPOINT ------------------------------------------------------------

def main():
    tongue_mesh = load_tongue_mesh(PATH_TONGUE)
    weights = compute_weights(tongue_mesh, CONTROL_POINTS_REST)
    trajectory = load_tongue_trajectory(PATH_DATA, PATH_STD)
    rigged_vertices = deform_mesh_sequence(tongue_mesh, trajectory, weights)

    print(f'Tongue trajectory frames: {trajectory.shape[0]}')
    print(f'Rigged vertex cache shape: {rigged_vertices.shape}')

    animate_tongue(trajectory, rigged_vertices, save_path=OUTPUT_GIF)


if __name__ == '__main__':
    main()
