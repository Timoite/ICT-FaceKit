import torch
import numpy as np
import os
import torch.nn.functional as F

class ICTTongueModel(torch.nn.Module):
    """
    PyTorch module for Tongue Linear Blend Skinning (LBS).
    Driven by 4 control points (Root, Back, Mid, Tip) which drive 3 bones.
    """
    def __init__(self,
                 rest_verts_path,
                 weights_path,
                 rest_matrices_path,
                 rest_tails_path,
                 device='cpu'):
        """
        Args:
            rest_verts_path: Path to .npy file (V, 3)
            weights_path: Path to .npy file (V, B)
            rest_matrices_path: Path to .npy file (B, 4, 4) - Bone Rest Matrices (Bind Matrices)
            rest_tails_path: Path to .npy file (B, 3) - Bone Rest Tail positions (to compute length)
            device: torch device
        """
        super().__init__()
        self.device = device

        # Load Data
        print(f"Loading Tongue Model data...")
        self.rest_verts = torch.tensor(np.load(rest_verts_path), dtype=torch.float32).to(device)
        self.weights = torch.tensor(np.load(weights_path), dtype=torch.float32).to(device) # (V, B)
        self.rest_matrices = torch.tensor(np.load(rest_matrices_path), dtype=torch.float32).to(device) # (B, 4, 4)
        self.rest_tails = torch.tensor(np.load(rest_tails_path), dtype=torch.float32).to(device) # (B, 3)

        # Validate shapes
        self.num_bones = self.rest_matrices.shape[0]
        self.num_verts = self.rest_verts.shape[0]
        print(f"  Vertices: {self.num_verts}")
        print(f"  Bones: {self.num_bones}")

        # Extract Rest Heads from matrices (Translation component)
        self.rest_heads = self.rest_matrices[:, :3, 3]

        # Compute Rest Vectors and Lengths
        self.rest_vectors = self.rest_tails - self.rest_heads # (B, 3)
        self.rest_lengths = torch.norm(self.rest_vectors, dim=1, keepdim=True) # (B, 1)
        self.rest_dirs = self.rest_vectors / (self.rest_lengths + 1e-6) # (B, 3)

        # Compute Inverse Bind Matrices (M_rest^-1)
        self.inverse_bind_matrices = torch.inverse(self.rest_matrices)

        # Extract Rest Rotations (3x3)
        self.rest_rotations = self.rest_matrices[:, :3, :3]

    def compute_rotation_from_to(self, u, v):
        """
        Compute rotation matrix that rotates vector u to vector v (shortest arc).
        u, v: (B, 3) normalized vectors
        Returns: (B, 3, 3) rotation matrices
        """
        # Cross product gives the axis of rotation
        w = torch.cross(u, v, dim=1)

        # Dot product gives cosine of angle
        dot = torch.sum(u * v, dim=1, keepdim=True)

        # Quaternion construction
        # q = [1 + dot, w_x, w_y, w_z]
        # Note: This formula is for u, v unit vectors.
        # If u == -v (180 deg), this is singular. We assume tongue doesn't flip 180 deg.

        q_xyz = w
        q_w = 1.0 + dot

        q = torch.cat([q_w, q_xyz], dim=1)
        q = F.normalize(q, dim=1)

        return self.quat2mat(q)

    def quat2mat(self, quat):
        """Convert quaternion to rotation matrix."""
        # quat: (B, 4) [w, x, y, z]
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        B = quat.shape[0]
        rot_mat = torch.zeros((B, 3, 3), device=self.device)

        rot_mat[:, 0, 0] = 1 - 2*y*y - 2*z*z
        rot_mat[:, 0, 1] = 2*x*y - 2*z*w
        rot_mat[:, 0, 2] = 2*x*z + 2*y*w

        rot_mat[:, 1, 0] = 2*x*y + 2*z*w
        rot_mat[:, 1, 1] = 1 - 2*x*x - 2*z*z
        rot_mat[:, 1, 2] = 2*y*z - 2*x*w

        rot_mat[:, 2, 0] = 2*x*z - 2*y*w
        rot_mat[:, 2, 1] = 2*y*z + 2*x*w
        rot_mat[:, 2, 2] = 1 - 2*x*x - 2*y*y

        return rot_mat

    def forward(self, control_points):
        """
        Deform tongue mesh based on control points.

        Args:
            control_points: (B+1, 3) [P0, P1, P2, P3] corresponding to [Root, Back, Mid, Tip]
                           or (Batch, B+1, 3) if batching is needed.
        Returns:
            deformed_verts: (V, 3) or (Batch, V, 3)
        """
        # Ensure input is tensor
        if not isinstance(control_points, torch.Tensor):
            control_points = torch.tensor(control_points, dtype=torch.float32, device=self.device)

        # Check if batched
        is_batched = control_points.dim() == 3
        if not is_batched:
            # Add batch dimension: (1, B+1, 3)
            control_points = control_points.unsqueeze(0)

        batch_size = control_points.shape[0]
        # print(f"DEBUG: forward batch_size={batch_size}, input_shape={control_points.shape}")

        # Construct Bone Segments from Control Points
        # Bone 0: P0 -> P1
        # Bone 1: P1 -> P2
        # Bone 2: P2 -> P3

        starts = control_points[:, :-1, :] # (Batch, B, 3)
        ends = control_points[:, 1:, :]    # (Batch, B, 3)

        current_vectors = ends - starts # (Batch, B, 3)
        current_lengths = torch.norm(current_vectors, dim=2, keepdim=True) # (Batch, B, 1)
        current_dirs = current_vectors / (current_lengths + 1e-6) # (Batch, B, 3)

        # Expand rest properties to batch
        # rest_dirs: (B, 3) -> (Batch, B, 3)
        rest_dirs_batch = self.rest_dirs.unsqueeze(0).expand(batch_size, -1, -1)
        rest_rotations_batch = self.rest_rotations.unsqueeze(0).expand(batch_size, -1, -1, -1)
        rest_lengths_batch = self.rest_lengths.unsqueeze(0).expand(batch_size, -1, -1)

        # 1. Compute Rotation (Align Rest Dir to Current Dir)
        # R_diff rotates Rest_Dir to Current_Dir
        # We need to flatten batch and bone dims for compute_rotation_from_to
        # (Batch*B, 3)
        R_diff_flat = self.compute_rotation_from_to(
            rest_dirs_batch.reshape(-1, 3),
            current_dirs.reshape(-1, 3)
        )
        R_diff = R_diff_flat.reshape(batch_size, self.num_bones, 3, 3)

        # R_current = R_diff @ R_rest
        # (Batch, B, 3, 3) @ (Batch, B, 3, 3) -> (Batch, B, 3, 3)
        R_current = torch.matmul(R_diff, rest_rotations_batch)

        # 2. Compute Scale
        # Scale factor s = L_curr / L_rest
        scales = current_lengths / (rest_lengths_batch + 1e-6) # (Batch, B, 1)

        # Construct Scale Matrix (Scaling along Y axis of the bone)
        # In Blender bone space, Y is the axis along the bone.
        S_local = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_bones, -1, -1).clone()
        # S_local[:, :, 1, 1] = scales.squeeze(-1) # This might fail if squeeze removes too much
        S_local[:, :, 1, 1] = scales[..., 0]

        # 3. Construct Full Transform M_current
        # M_current = T_current @ R_current @ S_local
        # Note: S_local is applied first (in local frame), then Rotation, then Translation.

        # Combine R and S: RS = R_current @ S_local
        RS = torch.matmul(R_current, S_local)

        # Construct 4x4 Matrix
        M_current = torch.eye(4, device=self.device).unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_bones, -1, -1).clone()
        M_current[:, :, :3, :3] = RS
        M_current[:, :, :3, 3] = starts

        # 4. Compute Skinning Matrix
        # M_skin = M_current @ M_inv_bind
        # M_inv_bind: (B, 4, 4) -> (Batch, B, 4, 4)
        inv_bind_batch = self.inverse_bind_matrices.unsqueeze(0).expand(batch_size, -1, -1, -1)
        M_skin = torch.matmul(M_current, inv_bind_batch) # (Batch, B, 4, 4)

        # 5. Linear Blend Skinning
        # v' = sum( w_i * (M_skin_i @ v) )

        # Prepare vertices (V, 4)
        V = self.rest_verts.shape[0]
        ones = torch.ones((V, 1), device=self.device)
        verts_homo = torch.cat([self.rest_verts, ones], dim=1) # (V, 4)

        # Compute Weighted Transformation Matrix per Vertex
        # weights: (V, B)
        # M_skin: (Batch, B, 4, 4)
        # M_weighted: (Batch, V, 4, 4) = sum_b (weight_vb * M_skin_b)

        # einsum: vb, kbij -> kvij
        M_weighted = torch.einsum('vb,kbij->kvij', self.weights, M_skin)
        # print(f"DEBUG: M_weighted shape: {M_weighted.shape}")

        # Apply to vertices
        # (Batch, V, 4, 4) @ (V, 4, 1) -> (Batch, V, 4, 1)
        # We need to expand verts_homo to batch: (Batch, V, 4, 1)
        verts_homo_batch = verts_homo.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, -1)

        deformed_homo = torch.matmul(M_weighted, verts_homo_batch).squeeze(-1) # (Batch, V, 4)

        deformed_verts = deformed_homo[:, :, :3]

        if not is_batched:
            return deformed_verts.squeeze(0)

        return deformed_verts

def load_tongue_animation_data(npy_path, mu_path, std_path, rest_control_points, animation_scale=0.09):
    """
    Load tongue animation data and convert to control points sequence.

    Args:
        npy_path: Path to .npy file with animation data (Frames, C)
        mu_path: Path to mean .npy
        std_path: Path to std .npy
        rest_control_points: Tensor (4, 3) of rest control points [P0, P1, P2, P3]
        animation_scale: Scale factor for animation (default 0.09 from Blender script)

    Returns:
        control_points: Tensor (Frames, 4, 3)
    """
    # Load data
    data = np.load(npy_path) # (Frames, C)
    mu = np.load(mu_path)
    std = np.load(std_path)

    # Constants
    TONGUE_CHANNELS = 8 # 4 points * 2 dims (y, z)

    # Extract tongue channels (first 8)
    tongue_zscores = data[:, :TONGUE_CHANNELS]
    mu_tongue = mu.flatten()[:TONGUE_CHANNELS]
    std_tongue = std.flatten()[:TONGUE_CHANNELS]

    # Un-normalize
    # coords = zscore * std + mu
    tongue_coords = (tongue_zscores * std_tongue) + mu_tongue

    # Reshape to (Frames, 4, 2)
    # Data order in Blender script: T4_x, T4_y, T3_x, T3_y ...
    # Blender script says:
    # Data 'x' -> Blender 'Y' (forward/backward)
    # Data 'y' -> Blender 'Z' (up/down)

    coords_reshaped = tongue_coords.reshape(-1, 4, 2)
    mu_reshaped = mu_tongue.reshape(4, 2)

    # Prepare output tensor
    num_frames = coords_reshaped.shape[0]
    control_points = torch.zeros((num_frames, 4, 3), dtype=torch.float32)

    # Rest points (4, 3)
    if isinstance(rest_control_points, torch.Tensor):
        rest_cp_np = rest_control_points.cpu().numpy()
    else:
        rest_cp_np = rest_control_points

    # Rest YZ (4, 2) -> [Y, Z]
    # Note: Blender script uses [p.y, p.z]
    rest_yz = rest_cp_np[:, 1:3]

    # Compute animated YZ
    # Data X (Front-Back) -> Blender Y (Forward-Backward)
    # Data Y (Up-Down)    -> Blender Z (Up-Down)

    # Note:
    # Blender Bone Y is aligned along -Y (Decreasing Y goes Front).
    # Data X is aligned along +X (Increasing X goes Front).
    # So we must NEGATE the X->Y mapping.

    data_x = coords_reshaped[:, :, 0]
    data_y = coords_reshaped[:, :, 1]

    mu_x = mu_reshaped[:, 0]
    mu_y = mu_reshaped[:, 1]

    rest_y = rest_yz[:, 0]
    rest_z = rest_yz[:, 1]

    # Blender Y = Rest Y - scale * (Data X - Mu X)
    blended_y = rest_y[np.newaxis, :] - animation_scale * (data_x - mu_x[np.newaxis, :])

    # Blender Z = Rest Z + scale * (Data Y - Mu Y)
    blended_z = rest_z[np.newaxis, :] + animation_scale * (data_y - mu_y[np.newaxis, :])

    # Construct full 3D points
    # X is constant (rest X)
    # Y, Z are animated

    rest_x = rest_cp_np[:, 0] # (4,)

    # Fill X
    control_points[:, :, 0] = torch.tensor(rest_x).unsqueeze(0).expand(num_frames, -1)

    # Fill Y, Z
    control_points[:, :, 1] = torch.tensor(blended_y)
    control_points[:, :, 2] = torch.tensor(blended_z)

    return control_points
