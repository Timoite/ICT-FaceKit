#!/usr/bin/env python3
"""
Visualize and verify the upper/lower lip vertex pair on the ICT face mesh.

Default pair corresponds to Multi-PIE 68 landmarks (0-based):
- landmark 62 -> vertex 5533 (upper inner lip mid)
- landmark 66 -> vertex 5517 (lower inner lip mid)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TONGUE_SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TONGUE_SCRIPTS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tongue_scripts.tongue_animation.face_model_io_trimesh import load_face_model_trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a lip vertex pair on the neutral ICT face mesh."
    )
    parser.add_argument(
        "--face-model-dir",
        default=str(PROJECT_ROOT / "FaceXModel"),
        help="Path to FaceXModel directory",
    )
    parser.add_argument(
        "--upper-vertex",
        type=int,
        default=5533,
        help="Upper lip vertex index",
    )
    parser.add_argument(
        "--lower-vertex",
        type=int,
        default=5517,
        help="Lower lip vertex index",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=15,
        help="Plot every Nth vertex for context cloud",
    )
    parser.add_argument(
        "--output-path",
        default=str(TONGUE_SCRIPTS_DIR / "vis_output" / "lip_vertex_pair_check.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive window",
    )
    return parser.parse_args()


def _set_equal_3d(ax: plt.Axes, pts: np.ndarray) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) * 0.5
    span = (maxs - mins).max() * 0.5
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)


def main() -> None:
    args = parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    face_model = load_face_model_trimesh(str(args.face_model_dir))
    verts = np.asarray(face_model.neutral_verts, dtype=np.float32)

    n_verts = len(verts)
    u = int(args.upper_vertex)
    l = int(args.lower_vertex)
    if not (0 <= u < n_verts and 0 <= l < n_verts):
        raise ValueError(
            f"Vertex index out of range: upper={u}, lower={l}, n_verts={n_verts}"
        )

    upper_pt = verts[u]
    lower_pt = verts[l]
    pair_dist = float(np.linalg.norm(upper_pt - lower_pt))

    step = max(1, int(args.subsample))
    context = verts[::step]

    # Local mouth crop around pair midpoint
    mid = 0.5 * (upper_pt + lower_pt)
    r = 3.0
    local_mask = (
        (np.abs(verts[:, 0] - mid[0]) < r)
        & (np.abs(verts[:, 1] - mid[1]) < r)
        & (np.abs(verts[:, 2] - mid[2]) < r)
    )
    local = verts[local_mask]

    fig = plt.figure(figsize=(14, 6))

    # Panel 1: global face context
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(context[:, 0], context[:, 1], context[:, 2], s=1, c="lightgray", alpha=0.35)
    ax1.scatter(*upper_pt, s=120, c="red", label=f"Upper ({u})")
    ax1.scatter(*lower_pt, s=120, c="dodgerblue", label=f"Lower ({l})")
    ax1.plot(
        [upper_pt[0], lower_pt[0]],
        [upper_pt[1], lower_pt[1]],
        [upper_pt[2], lower_pt[2]],
        c="gold",
        linewidth=2.0,
        label=f"Distance = {pair_dist:.4f}",
    )
    ax1.set_title("Global mesh context")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    _set_equal_3d(ax1, context)
    ax1.legend(loc="upper right")

    # Panel 2: local mouth zoom
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    if len(local) > 0:
        ax2.scatter(local[:, 0], local[:, 1], local[:, 2], s=3, c="gray", alpha=0.4)
    ax2.scatter(*upper_pt, s=140, c="red", label=f"Upper ({u})")
    ax2.scatter(*lower_pt, s=140, c="dodgerblue", label=f"Lower ({l})")
    ax2.plot(
        [upper_pt[0], lower_pt[0]],
        [upper_pt[1], lower_pt[1]],
        [upper_pt[2], lower_pt[2]],
        c="gold",
        linewidth=2.5,
        label=f"Distance = {pair_dist:.4f}",
    )
    ax2.set_title("Local mouth zoom")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    _set_equal_3d(ax2, np.vstack([local, upper_pt[None, :], lower_pt[None, :]]) if len(local) > 0 else np.vstack([upper_pt[None, :], lower_pt[None, :]]))
    ax2.legend(loc="upper right")

    fig.suptitle(
        "ICT Face Mesh Lip Vertex Pair Check\n"
        "Multi-PIE(62)->5533, Multi-PIE(66)->5517",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if args.show:
        plt.show()
    plt.close(fig)

    print(f"Saved visualization: {output_path}")
    print(f"Upper vertex {u}: {upper_pt.tolist()}")
    print(f"Lower vertex {l}: {lower_pt.tolist()}")
    print(f"Euclidean distance: {pair_dist:.6f}")


if __name__ == "__main__":
    main()
