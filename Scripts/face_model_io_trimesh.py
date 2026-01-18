"""
ICT FaceKit - Trimesh/NumPy Version
===================================
Robust Loader: Uses a strict, raw OBJ parser to guarantee vertex order
and count matches exactly (essential for blendshapes).
"""

import os
import json
import numpy as np
import trimesh
from pathlib import Path

def load_face_model_trimesh(model_directory, load_identities=True):
    loader = _TrimeshModelLoader(model_directory, load_identities)
    return loader.load_model()

class _TrimeshModelLoader:
    def __init__(self, model_path, load_identities=True):
        self._model_path = Path(model_path)
        self.load_identities = load_identities

    def load_model(self):
        print("Loading face model with Trimesh...")

        # 1. Read Config
        model_config = self._read_model_config()

        # 2. Read Neutral Mesh (Strict Mode)
        print("Reading generic neutral mesh...")
        neutral_mesh = self._load_obj_strict(self._model_path / 'generic_neutral_mesh.obj')

        neutral_verts = np.array(neutral_mesh.vertices, dtype=np.float32)
        faces = np.array(neutral_mesh.faces, dtype=np.int32)

        print(f"Neutral mesh: {neutral_verts.shape[0]} vertices, {faces.shape[0]} faces")

        # 3. Read Expressions
        print("Reading expression morph targets...")
        ex_names, ex_deltas = self._read_expression_morph_targets(
            model_config['expressions'],
            neutral_verts
        )

        # 4. Read Identities
        id_names = []
        if self.load_identities:
            print("Reading identity morph targets...")
            # (Identities logic can be added here if needed later)
            pass

        # 5. Pack Data
        expressions_dict = {name: delta for name, delta in zip(ex_names, ex_deltas)}

        return {
            'generic_neutral_mesh': neutral_mesh,
            'neutral_verts': neutral_verts,
            'faces': faces,
            'expression_names': ex_names,
            'expressions': expressions_dict,
            'materials_idx': None,
            'material_colors': None
        }

    def _read_model_config(self):
        with open(self._model_path / 'vertex_indices.json') as file:
            return json.load(file)

    def _load_obj_strict(self, file_path):
        """
        Parses OBJ V/F data manually to guarantee 1:1 vertex correspondence.
        Ignores groups, materials, and normals to prevent Trimesh from splitting vertices.
        """
        vertices = []
        faces = []

        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    # Parse vertex: v x y z
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    # Parse face: f v1/vt1/vn1 v2/...
                    parts = line.split()
                    # OBJ is 1-indexed, convert to 0-indexed
                    # We only care about the vertex index (first number before any /)
                    face_idxs = [int(p.split('/')[0]) - 1 for p in parts[1:]]

                    # Handle quads/ngons by triangulation (fan method) if necessary
                    # But typically facekit is triangles.
                    if len(face_idxs) == 3:
                        faces.append(face_idxs)
                    elif len(face_idxs) == 4:
                        # Split quad into two triangles
                        faces.append([face_idxs[0], face_idxs[1], face_idxs[2]])
                        faces.append([face_idxs[0], face_idxs[2], face_idxs[3]])

        # Create Trimesh object directly
        # process=False is CRITICAL to prevent merging/reordering
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        return mesh

    def _read_expression_morph_targets(self, expression_names, neutral_verts):
        ex_names = []
        ex_deltas = []

        for ex_name in expression_names:
            if ex_name.startswith('identity'): continue

            file_path = self._model_path / f'{ex_name}.obj'
            if not file_path.exists(): continue

            # Use same strict loader
            target_mesh = self._load_obj_strict(file_path)
            target_verts = np.array(target_mesh.vertices, dtype=np.float32)

            # Validation
            if target_verts.shape != neutral_verts.shape:
                # If mismatch, it might be that the morph target is partial?
                # Usually ICT targets are full meshes.
                # If the count is slightly different, we can't subtract.
                # However, with strict loading, they should match perfectly.
                print(f"  SKIP {ex_name}: Count mismatch ({target_verts.shape[0]} vs {neutral_verts.shape[0]})")
                continue

            # Compute Delta
            delta = target_verts - neutral_verts

            ex_names.append(ex_name)
            ex_deltas.append(delta)

        return ex_names, ex_deltas

    def _read_identity_morph_targets(self, neutral_verts):
        return [], []
