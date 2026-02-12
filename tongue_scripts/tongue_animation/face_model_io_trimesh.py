"""
ICT FaceKit - Trimesh/NumPy Version (Class-Based)
=================================================
Robust Loader: Uses a strict, raw OBJ parser to guarantee vertex order
and count matches exactly (essential for blendshapes).
Returns a TrimeshFaceModel object compatible with animation scripts.
"""

import os
import json
import numpy as np
import trimesh
from pathlib import Path

# =========================================================================
# 1. THE FACE MODEL CLASS
# =========================================================================
class TrimeshFaceModel:
    def __init__(self, neutral_verts, faces, expression_names, expression_deltas, identity_names=None, identity_deltas=None):
        """
        A Numpy-based container for the ICT FaceKit.
        Stores neutral geometry and deltas, provides a deform() method.
        """
        # Base Geometry
        self.neutral_verts = neutral_verts.astype(np.float32)
        self.faces = faces
        
        # Expression Data
        self.expression_names = expression_names
        self.expression_deltas = expression_deltas.astype(np.float32) # Shape: (Num_Expr, Num_Verts, 3)
        
        # Fast Lookup Map: Name -> Index
        self.expr_map = {name: i for i, name in enumerate(expression_names)}
        
        # Identity Data (Optional placeholders)
        self.identity_names = identity_names if identity_names is not None else []
        self.identity_deltas = identity_deltas if identity_deltas is not None else np.array([])
        
        # Pre-allocate output buffer to avoid recreating arrays every frame
        self._current_verts = np.zeros_like(self.neutral_verts)

    def deform(self, weights_dict):
        """
        Computes the deformed mesh based on expression weights.
        
        Args:
            weights_dict: Dictionary mapping expression names to float weights (0.0 - 1.0).
            
        Returns:
            (N, 3) numpy array of deformed vertices.
        """
        # 1. Start with Neutral Vertices
        np.copyto(self._current_verts, self.neutral_verts)
        
        # 2. Accumulate Deltas
        # We iterate over the input dictionary (active shapes) for efficiency
        for name, weight in weights_dict.items():
            if weight == 0: 
                continue
            
            # Look up index
            idx = self.expr_map.get(name)
            if idx is not None:
                # V_current += weight * Delta
                self._current_verts += weight * self.expression_deltas[idx]
            # else:
            #     print(f"Warning: Expression '{name}' not found in model.")
        
        return self._current_verts

# =========================================================================
# 2. THE LOADER
# =========================================================================
def load_face_model_trimesh(model_directory, load_identities=True):
    """
    Factory function to initialize and load the model.
    """
    loader = _TrimeshModelLoader(model_directory, load_identities)
    return loader.load_model()

class _TrimeshModelLoader:
    def __init__(self, model_path, load_identities=True):
        self._model_path = Path(model_path)
        self.load_identities = load_identities

    def load_model(self):
        print("Loading face model with Trimesh...")

        # 1. Read Config (vertex_indices.json) to get strict order
        model_config = self._read_model_config()

        # 2. Read Neutral Mesh (Strict Mode)
        print("Reading generic neutral mesh...")
        neutral_path = self._model_path / 'generic_neutral_mesh.obj'
        if not neutral_path.exists():
            raise FileNotFoundError(f"Neutral mesh not found at {neutral_path}")

        # Use custom raw parser to ensure exact vertex order
        neutral_mesh = self._load_obj_strict(neutral_path)
        
        neutral_verts = np.array(neutral_mesh.vertices, dtype=np.float32)
        faces = np.array(neutral_mesh.faces, dtype=np.int32)

        print(f"Neutral mesh: {neutral_verts.shape[0]} vertices, {faces.shape[0]} faces")

        # 3. Determine Expression List
        if model_config and 'expressions' in model_config:
            # Use order from JSON
            target_names = model_config['expressions']
        else:
            # Fallback: scan directory
            print("Warning: No config found. Scanning directory for OBJ files.")
            target_names = self._scan_directory_for_objs()

        # 4. Read Expression Morph Targets
        print("Reading expression morph targets...")
        ex_names, ex_deltas = self._read_expression_morph_targets(target_names, neutral_verts)
        
        # 5. Read Identities (Skipped for now to keep it simple, but structure is here)
        id_names, id_deltas = [], None

        # 6. Construct and Return Model
        return TrimeshFaceModel(
            neutral_verts=neutral_verts,
            faces=faces,
            expression_names=ex_names,
            expression_deltas=np.array(ex_deltas),
            identity_names=id_names,
            identity_deltas=id_deltas
        )

    def _read_model_config(self):
        config_path = self._model_path / 'vertex_indices.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return None

    def _scan_directory_for_objs(self):
        # Exclude known non-expression files
        excludes = ["generic_neutral_mesh", "neutral", "base"]
        return [f.stem for f in self._model_path.glob("*.obj") if f.stem not in excludes]

    def _load_obj_strict(self, filename):
        """
        Manually parses an OBJ file to guarantee that vertices are loaded 
        in the exact order they appear in the file. 
        Standard loaders often reorder or merge vertices, breaking blendshapes.
       
        """
        vertices = []
        faces = []

        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    # Parse Vertex: v x y z
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    # Parse Face: f v1/vt1/vn1 v2/... v3/...
                    parts = line.split()
                    # OBJ is 1-indexed, convert to 0-indexed
                    face_idxs = []
                    for p in parts[1:]:
                        # Take the first number (vertex index) before any slash
                        v_idx = int(p.split('/')[0]) - 1
                        face_idxs.append(v_idx)
                    
                    # Handle Triangles and Quads (convert quads to 2 triangles)
                    if len(face_idxs) == 3:
                        faces.append(face_idxs)
                    elif len(face_idxs) == 4:
                        faces.append([face_idxs[0], face_idxs[1], face_idxs[2]])
                        faces.append([face_idxs[0], face_idxs[2], face_idxs[3]])

        # Create Trimesh object directly without processing
        # process=False is CRITICAL to prevent merging/reordering
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        return mesh

    def _read_expression_morph_targets(self, expression_names, neutral_verts):
        ex_names = []
        ex_deltas = []

        for ex_name in expression_names:
            if ex_name.startswith('identity'): continue

            file_path = self._model_path / f'{ex_name}.obj'
            if not file_path.exists(): 
                # print(f"  [Warning] Blendshape {ex_name} not found. Skipping/Zeroing.")
                # We append a zero delta to maintain index alignment with the JSON list
                # This ensures coeff[i] always corresponds to expression_names[i]
                delta = np.zeros_like(neutral_verts)
                ex_names.append(ex_name)
                ex_deltas.append(delta)
                continue

            # Strict Load
            target_mesh = self._load_obj_strict(file_path)
            target_verts = np.array(target_mesh.vertices, dtype=np.float32)

            # Validation
            if target_verts.shape != neutral_verts.shape:
                print(f"  [Error] Topology mismatch: {ex_name} ({target_verts.shape[0]} vs {neutral_verts.shape[0]})")
                # Append zero delta to maintain alignment
                delta = np.zeros_like(neutral_verts)
            else:
                # Compute Delta
                delta = target_verts - neutral_verts

            ex_names.append(ex_name)
            ex_deltas.append(delta)

        return ex_names, ex_deltas
