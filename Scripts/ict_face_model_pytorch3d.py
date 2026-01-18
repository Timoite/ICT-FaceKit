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

"""PyTorch3D-based face model class for ICT Face Model.

This module defines the FaceModelPyTorch3D class which allows working with faces
parameterized by the ICT morphable face model using PyTorch tensors and PyTorch3D
for GPU acceleration and batch processing.
"""

import torch
from pytorch3d.structures import Meshes


class FaceModelPyTorch3D:
    """A class that parameterizes faces with the ICT face model using PyTorch3D.

    This class represents faces parameterized by the ICT face model using PyTorch
    tensors. Each FaceModelPyTorch3D object uses identity and expression weights
    to compute deformed meshes efficiently on GPU.

    Attributes:
        device: torch.device where tensors are stored
        neutral_verts: Neutral mesh vertices (V, 3)
        faces: Face indices (F, 3)
        expression_names: List of expression names
        identity_names: List of identity names
        expression_shape_modes: Expression deltas (N_ex, V, 3)
        identity_shape_modes: Identity deltas (N_id, V, 3)
        expression_weights: Current expression weights (N_ex,)
        identity_weights: Current identity weights (N_id,)
    """

    def __init__(self, model_data, device=None):
        """Creates a new FaceModelPyTorch3D object.

        Args:
            model_data: Dictionary returned by load_face_model_pytorch3d()
            device: Device to use (defaults to device in model_data)
        """
        # Set device
        if device is None:
            self.device = model_data['device']
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        # Store model components
        self.neutral_verts = model_data['neutral_verts'].to(self.device)
        self.faces = model_data['faces'].to(self.device)
        self.verts_uvs = model_data.get('verts_uvs')
        self.faces_uvs = model_data.get('faces_uvs')
        self.materials_idx = model_data.get('materials_idx')
        self.material_colors = model_data.get('material_colors')

        if self.verts_uvs is not None:
            self.verts_uvs = self.verts_uvs.to(self.device)
        if self.faces_uvs is not None:
            self.faces_uvs = self.faces_uvs.to(self.device)
        if self.materials_idx is not None:
            self.materials_idx = self.materials_idx.to(self.device)

        # Store expression and identity data
        self.expression_names = model_data['expression_names']
        self.identity_names = model_data['identity_names']
        self.expression_shape_modes = model_data['expression_shape_modes'].to(self.device)
        self.identity_shape_modes = model_data['identity_shape_modes'].to(self.device)

        # Initialize weights to zero
        self.num_expression_shapes = len(self.expression_names)
        self.num_identity_shapes = len(self.identity_names)
        self.expression_weights = torch.zeros(self.num_expression_shapes,
                                             dtype=torch.float32, device=self.device)
        self.identity_weights = torch.zeros(self.num_identity_shapes,
                                           dtype=torch.float32, device=self.device)

        # Create expression name to index mapping for fast lookup
        self._expression_name_to_idx = {name: idx for idx, name in enumerate(self.expression_names)}
        self._identity_name_to_idx = {name: idx for idx, name in enumerate(self.identity_names)}

    def set_identity(self, identity_weights):
        """Sets the identity weights.

        Args:
            identity_weights: Tensor, list, or dict of identity weights
                - If tensor/list: should match number of identity shapes
                - If dict: maps identity names to weights
        """
        if isinstance(identity_weights, dict):
            # Dictionary mapping names to weights
            self.identity_weights.zero_()
            for name, weight in identity_weights.items():
                if name in self._identity_name_to_idx:
                    idx = self._identity_name_to_idx[name]
                    self.identity_weights[idx] = float(weight)
        else:
            # Tensor or list
            if not isinstance(identity_weights, torch.Tensor):
                identity_weights = torch.tensor(identity_weights, dtype=torch.float32, device=self.device)
            else:
                identity_weights = identity_weights.to(self.device)

            # Copy up to available shapes
            min_num = min(self.num_identity_shapes, len(identity_weights))
            self.identity_weights[:min_num] = identity_weights[:min_num]

    def set_expression(self, expression_weights):
        """Sets the expression weights.

        Args:
            expression_weights: Tensor, list, or dict of expression weights
                - If tensor/list: should match number of expression shapes
                - If dict: maps expression names to weights
        """
        if isinstance(expression_weights, dict):
            # Dictionary mapping names to weights
            self.expression_weights.zero_()
            for name, weight in expression_weights.items():
                if name in self._expression_name_to_idx:
                    idx = self._expression_name_to_idx[name]
                    self.expression_weights[idx] = float(weight)
        else:
            # Tensor or list
            if not isinstance(expression_weights, torch.Tensor):
                expression_weights = torch.tensor(expression_weights, dtype=torch.float32, device=self.device)
            else:
                expression_weights = expression_weights.to(self.device)

            # Copy up to available shapes
            min_num = min(self.num_expression_shapes, len(expression_weights))
            self.expression_weights[:min_num] = expression_weights[:min_num]

    def reset_weights(self):
        """Resets all expression and identity weights to zero."""
        self.expression_weights.zero_()
        self.identity_weights.zero_()

    def deform_mesh(self):
        """Computes deformed vertices using current weights.

        Returns:
            Deformed vertices tensor (V, 3)
        """
        # Start with neutral vertices
        deformed_verts = self.neutral_verts.clone()

        # Add identity contribution: (V, 3) + (N_id,) @ (N_id, V, 3)
        if self.num_identity_shapes > 0 and self.identity_weights.abs().sum() > 0:
            identity_delta = torch.einsum('k,kvd->vd', self.identity_weights, self.identity_shape_modes)
            deformed_verts = deformed_verts + identity_delta

        # Add expression contribution: (V, 3) + (N_ex,) @ (N_ex, V, 3)
        if self.num_expression_shapes > 0 and self.expression_weights.abs().sum() > 0:
            expression_delta = torch.einsum('k,kvd->vd', self.expression_weights, self.expression_shape_modes)
            deformed_verts = deformed_verts + expression_delta

        return deformed_verts

    def deform_batch(self, batch_weights, identity_weights=None):
        """Deforms multiple frames in batch for efficiency.

        Args:
            batch_weights: List of dicts or tensor (B, N_ex) of expression weights per frame
            identity_weights: Optional identity weights (shared across batch)
                - If None, uses current identity_weights
                - If provided, can be dict or tensor

        Returns:
            Deformed vertices tensor (B, V, 3)
        """
        # Handle identity weights
        if identity_weights is not None:
            # Save current weights to restore later
            saved_id_weights = self.identity_weights.clone()
            self.set_identity(identity_weights)

        # Compute identity contribution once (shared across batch)
        base_verts = self.neutral_verts.clone()  # (V, 3)

        if self.num_identity_shapes > 0 and self.identity_weights.abs().sum() > 0:
            identity_delta = torch.einsum('k,kvd->vd', self.identity_weights, self.identity_shape_modes)
            base_verts = base_verts + identity_delta

        # Process expression weights
        if isinstance(batch_weights, list) and len(batch_weights) > 0 and isinstance(batch_weights[0], dict):
            # Convert list of dicts to tensor
            batch_size = len(batch_weights)
            ex_weights_tensor = torch.zeros((batch_size, self.num_expression_shapes),
                                           dtype=torch.float32, device=self.device)

            for i, weights_dict in enumerate(batch_weights):
                for name, weight in weights_dict.items():
                    if name in self._expression_name_to_idx:
                        idx = self._expression_name_to_idx[name]
                        ex_weights_tensor[i, idx] = float(weight)
        else:
            # Already a tensor
            if not isinstance(batch_weights, torch.Tensor):
                ex_weights_tensor = torch.tensor(batch_weights, dtype=torch.float32, device=self.device)
            else:
                ex_weights_tensor = batch_weights.to(self.device)

        batch_size = ex_weights_tensor.shape[0]

        # Broadcast base vertices to batch: (V, 3) -> (B, V, 3)
        deformed_batch = base_verts.unsqueeze(0).expand(batch_size, -1, -1).clone()

        # Add expression contribution: (B, V, 3) + (B, N_ex) @ (N_ex, V, 3)
        if self.num_expression_shapes > 0:
            expression_deltas = torch.einsum('bk,kvd->bvd', ex_weights_tensor, self.expression_shape_modes)
            deformed_batch = deformed_batch + expression_deltas

        # Restore identity weights if we changed them
        if identity_weights is not None:
            self.identity_weights = saved_id_weights

        return deformed_batch

    def get_meshes(self, deformed_verts=None):
        """Creates PyTorch3D Meshes object from deformed vertices.

        Args:
            deformed_verts: Deformed vertices (V, 3) or (B, V, 3)
                If None, deforms using current weights

        Returns:
            pytorch3d.structures.Meshes object
        """
        if deformed_verts is None:
            deformed_verts = self.deform_mesh()

        # Handle single mesh or batch
        if deformed_verts.dim() == 2:
            # Single mesh (V, 3)
            verts_list = [deformed_verts]
            faces_list = [self.faces]
        else:
            # Batch (B, V, 3)
            batch_size = deformed_verts.shape[0]
            verts_list = [deformed_verts[i] for i in range(batch_size)]
            faces_list = [self.faces for _ in range(batch_size)]

        meshes = Meshes(verts=verts_list, faces=faces_list)
        return meshes

    def get_meshes_batch(self, batch_weights, identity_weights=None):
        """Convenience method to deform and create Meshes in one call.

        Args:
            batch_weights: Batch expression weights (list of dicts or tensor)
            identity_weights: Optional identity weights

        Returns:
            pytorch3d.structures.Meshes object with batch_size meshes
        """
        deformed_batch = self.deform_batch(batch_weights, identity_weights)
        return self.get_meshes(deformed_batch)

    def to(self, device):
        """Moves all tensors to specified device.

        Args:
            device: Target device ('cpu', 'cuda', or torch.device)

        Returns:
            self for chaining
        """
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.neutral_verts = self.neutral_verts.to(device)
        self.faces = self.faces.to(device)
        self.expression_shape_modes = self.expression_shape_modes.to(device)
        self.identity_shape_modes = self.identity_shape_modes.to(device)
        self.expression_weights = self.expression_weights.to(device)
        self.identity_weights = self.identity_weights.to(device)

        if self.verts_uvs is not None:
            self.verts_uvs = self.verts_uvs.to(device)
        if self.faces_uvs is not None:
            self.faces_uvs = self.faces_uvs.to(device)

        return self

    def __repr__(self):
        return (f"FaceModelPyTorch3D(\n"
                f"  vertices={self.neutral_verts.shape[0]}, "
                f"  faces={self.faces.shape[0]},\n"
                f"  expressions={self.num_expression_shapes}, "
                f"  identities={self.num_identity_shapes},\n"
                f"  device={self.device}\n"
                f")")
