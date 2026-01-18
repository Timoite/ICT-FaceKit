##########################################################################################
#                                                                                        #
# ICT FaceKit - PyTorch3D Version                                                        #
#                                                                                        #
# Copyright (c) 2020 USC Institute for Creative Technologies                             #
#                                                                                        #
# Permission is hereby granted, free of charge, to any person obtaining a copy           #
# of this software and associated documentation files (the "Software"), to deal          #
# in the Software without restriction, including without limitation the rights           #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell              #
# copies of the Software, and to permit persons to whom the Software is                  #
# furnished to do so, subject to the following conditions:                               #
#                                                                                        #
# The above copyright notice and this permission notice shall be included in all         #
# copies or substantial portions of the Software.                                        #
#                                                                                        #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR             #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,               #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE            #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER                 #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,          #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE          #
# SOFTWARE.                                                                              #
##########################################################################################

"""PyTorch3D-based functionality to read ICT face models.

This module provides functionality to load the ICT Face Model using PyTorch tensors
for GPU acceleration and compatibility with PyTorch3D rendering pipeline.
"""

import os
import json
import torch
from pathlib import Path
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes


def read_coefficients(file_path):
    """Reads coefficients representing identity and expression

    Args:
        file_path: The file path of the .json file we are reading the
            identity and expression coefficients from.

    Returns:
        identity and expression shape coefficients as torch tensors
    """
    with open(file_path) as file:
        face_model_json = json.load(file)
        id_coeffs = torch.tensor(face_model_json['identity_coefficients'], dtype=torch.float32)
        ex_coeffs = torch.tensor(face_model_json['expression_coefficients'], dtype=torch.float32)
        return id_coeffs, ex_coeffs


def load_face_model_pytorch3d(model_directory, device='cpu', load_identities=True):
    """Loads the ICT Face Model using PyTorch3D.

    Args:
        model_directory: Path to the FaceXModel directory
        device: Device to load tensors on ('cpu', 'cuda', or torch.device)
        load_identities: Whether to load identity morph targets (default: True)

    Returns:
        Dictionary containing:
            - 'model_config': Model configuration dict
            - 'generic_neutral_mesh': Meshes object for neutral mesh
            - 'neutral_verts': Tensor of neutral vertices (V, 3)
            - 'faces': Tensor of face indices (F, 3)
            - 'verts_uvs': Tensor of UV coordinates (optional)
            - 'faces_uvs': Tensor of UV face indices (optional)
            - 'expression_names': List of expression names
            - 'identity_names': List of identity names
            - 'expression_shape_modes': Tensor of expression deltas (N_ex, V, 3)
            - 'identity_shape_modes': Tensor of identity deltas (N_id, V, 3)
            - 'device': Device tensors are on
    """
    loader = _PyTorch3DModelLoader(model_directory, device, load_identities)
    return loader.load_model()


class _PyTorch3DModelLoader:
    """Internal loader class for PyTorch3D-based ICT Face Model"""

    def __init__(self, model_path, device='cpu', load_identities=True):
        self._model_path = Path(model_path)
        self.load_identities = load_identities
        if isinstance(device, str):
            self._device = torch.device(device)
        else:
            self._device = device

    def load_model(self):
        """Loads the ICT Face Model with PyTorch3D.

        Returns:
            Dictionary containing all model components as PyTorch tensors
        """
        print("Loading face model with PyTorch3D...")
        print(f"Device: {self._device}")

        # Read model config
        model_config = self._read_model_config()

        # Read generic neutral mesh
        print("Reading generic neutral mesh...")
        neutral_data = self._read_mesh_pytorch3d(self._model_path / 'generic_neutral_mesh.obj')
        neutral_verts = neutral_data['verts']
        faces = neutral_data['faces']
        verts_uvs = neutral_data.get('verts_uvs', None)
        faces_uvs = neutral_data.get('faces_uvs', None)
        materials_idx = neutral_data.get('materials_idx', None)
        material_colors = neutral_data.get('material_colors', None)

        print(f"Neutral mesh: {neutral_verts.shape[0]} vertices, {faces.shape[0]} faces")
        if materials_idx is not None:
            print(f"Loaded material indices for {len(materials_idx)} faces")

        # Read expression and identity morph targets
        print("Reading expression morph targets...")
        ex_names, ex_meshes_verts = self._read_expression_morph_targets(model_config['expressions'])

        if self.load_identities:
            print("Reading identity morph targets...")
            id_names, id_meshes_verts = self._read_identity_morph_targets()
        else:
            print("Skipping identity morph targets...")
            id_names, id_meshes_verts = [], []

        num_expression_shapes = len(ex_names)
        num_identity_shapes = len(id_names)

        print(f"Loaded {num_expression_shapes} expressions, {num_identity_shapes} identities")

        # Compute shape mode deltas
        print("Computing expression shape modes...")
        expression_shape_modes = self._compute_shape_mode_deltas(neutral_verts, ex_meshes_verts)

        print("Computing identity shape modes...")
        identity_shape_modes = self._compute_shape_mode_deltas(neutral_verts, id_meshes_verts)

        # Create Meshes object for neutral mesh
        if verts_uvs is not None and faces_uvs is not None:
            textures = None  # Can add texture support later
            generic_neutral_mesh = Meshes(
                verts=[neutral_verts],
                faces=[faces],
                textures=textures
            )
        else:
            generic_neutral_mesh = Meshes(
                verts=[neutral_verts],
                faces=[faces]
            )

        print("Face model loaded successfully with PyTorch3D")

        return {
            'model_config': model_config,
            'generic_neutral_mesh': generic_neutral_mesh,
            'neutral_verts': neutral_verts,
            'faces': faces,
            'verts_uvs': verts_uvs,
            'faces_uvs': faces_uvs,
            'materials_idx': materials_idx,
            'material_colors': material_colors,
            'expression_names': ex_names,
            'identity_names': id_names,
            'expression_shape_modes': expression_shape_modes,
            'identity_shape_modes': identity_shape_modes,
            'num_expression_shapes': num_expression_shapes,
            'num_identity_shapes': num_identity_shapes,
            'device': self._device
        }

    def _read_model_config(self):
        """Reads and returns the face model config json file.

        Returns:
            A dictionary representation of the model config json file.
        """
        file_path = self._model_path / 'vertex_indices.json'
        with open(file_path) as file:
            model_config = json.load(file)
            return model_config

    def _read_mesh_pytorch3d(self, file_path):
        """Reads a mesh file using PyTorch3D.

        Args:
            file_path: Path to .obj file

        Returns:
            Dictionary with 'verts', 'faces', and optionally 'verts_uvs', 'faces_uvs'
        """
        # Load using pytorch3d
        verts, faces, aux = load_obj(str(file_path), device=self._device)

        # pytorch3d load_obj returns faces as Faces object with verts_idx attribute
        faces_idx = faces.verts_idx

        result = {
            'verts': verts,
            'faces': faces_idx
        }

        # Add texture coordinates if available
        if aux.verts_uvs is not None and len(aux.verts_uvs) > 0:
            result['verts_uvs'] = aux.verts_uvs
            if hasattr(faces, 'textures_idx') and faces.textures_idx is not None:
                result['faces_uvs'] = faces.textures_idx

        # Add material indices if available
        if hasattr(faces, 'materials_idx') and faces.materials_idx is not None:
            result['materials_idx'] = faces.materials_idx

        # Add material colors if available
        if hasattr(aux, 'material_colors') and aux.material_colors is not None:
            result['material_colors'] = aux.material_colors

        return result

    def _read_expression_morph_targets(self, expression_names):
        """Reads and returns the expressions in the face model.

        Args:
            expression_names: List of expression names from config

        Returns:
            Tuple of (expression_names, list of vertex tensors)
        """
        ex_names = []
        ex_meshes_verts = []

        for ex_name in expression_names:
            # Skip identity shapes if they are listed in expressions
            if ex_name.startswith('identity'):
                continue

            print(f"  Reading expression: {ex_name}")
            file_path = self._model_path / f'{ex_name}.obj'

            if not file_path.exists():
                print(f"    Warning: {file_path} not found, skipping")
                continue

            mesh_data = self._read_mesh_pytorch3d(file_path)
            ex_names.append(ex_name)
            ex_meshes_verts.append(mesh_data['verts'])

        return ex_names, ex_meshes_verts

    def _read_identity_morph_targets(self):
        """Reads and returns the identities in the face model.

        Returns:
            Tuple of (identity_names, list of vertex tensors)
        """
        id_names = []
        id_meshes_verts = []

        identity_num = 0
        while True:
            id_name = f'identity{identity_num:03d}'
            file_path = self._model_path / f'{id_name}.obj'

            if not file_path.exists():
                if identity_num == 0:
                    print(f"    Warning: No identity files found")
                break

            print(f"  Reading identity: {id_name}")
            mesh_data = self._read_mesh_pytorch3d(file_path)
            id_names.append(id_name)
            id_meshes_verts.append(mesh_data['verts'])
            identity_num += 1

        return id_names, id_meshes_verts

    def _compute_shape_mode_deltas(self, neutral_verts, morph_target_verts_list):
        """Computes shape mode deltas from morph targets.

        Args:
            neutral_verts: Neutral mesh vertices tensor (V, 3)
            morph_target_verts_list: List of morph target vertex tensors

        Returns:
            Tensor of shape (K, V, 3) where K is number of morph targets
        """
        if len(morph_target_verts_list) == 0:
            return torch.zeros((0, neutral_verts.shape[0], 3),
                             dtype=torch.float32, device=self._device)

        # Stack all morph targets into (K, V, 3) tensor
        morph_targets = torch.stack(morph_target_verts_list, dim=0)

        # Compute deltas: (K, V, 3) - (V, 3) -> (K, V, 3)
        shape_modes = morph_targets - neutral_verts.unsqueeze(0)

        return shape_modes


def write_mesh_pytorch3d(file_path, verts, faces, verts_uvs=None, faces_uvs=None):
    """Writes a mesh to OBJ file.

    Args:
        file_path: Output file path
        verts: Vertex tensor (V, 3)
        faces: Face indices tensor (F, 3)
        verts_uvs: Optional UV coordinates (V_uv, 2)
        faces_uvs: Optional UV face indices (F, 3)
    """
    from pytorch3d.io import save_obj

    print(f"Writing mesh to: {file_path}")

    # Ensure tensors are on CPU for saving
    verts_cpu = verts.cpu() if verts.is_cuda else verts
    faces_cpu = faces.cpu() if faces.is_cuda else faces

    if verts_uvs is not None and faces_uvs is not None:
        verts_uvs_cpu = verts_uvs.cpu() if verts_uvs.is_cuda else verts_uvs
        faces_uvs_cpu = faces_uvs.cpu() if faces_uvs.is_cuda else faces_uvs
        save_obj(file_path, verts_cpu, faces_cpu,
                verts_uvs=verts_uvs_cpu, faces_uvs=faces_uvs_cpu)
    else:
        save_obj(file_path, verts_cpu, faces_cpu)


# Convenience function for backward compatibility
def load_face_model(model_directory, device='cpu'):
    """Alias for load_face_model_pytorch3d for convenience.

    Args:
        model_directory: Path to the FaceXModel directory
        device: Device to load tensors on ('cpu', 'cuda', or torch.device)

    Returns:
        Dictionary containing model components
    """
    return load_face_model_pytorch3d(model_directory, device)
