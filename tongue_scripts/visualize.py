import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ==========================================
# 2. LOAD & VERIFY DATA
# ==========================================

# --- CONFIGURATION: REPLACE WITH YOUR PATHS ---
path_data = 'tongue_scripts/26/npy/26_reamey_0_1_1.npy'
path_mean = 'tongue_scripts/normalising_vectors/JW13_4points_mu.npy'
path_std  = 'tongue_scripts/normalising_vectors/JW13_4points_std.npy'

print(f"\n--- Loading {path_data} ---")
raw_data = np.load(path_data)
data_mean = np.load(path_mean).reshape(-1)
data_std = np.load(path_std).reshape(-1)

# 1. Check Dimensions
print(f"Data Shape: {raw_data.shape}")
print(f"Mean Shape: {data_mean.shape}")
print(f"Std Shape:  {data_std.shape}")

expected_dim = 16
if raw_data.shape[1] != expected_dim:
    print(f"WARNING: Expected {expected_dim} columns based on docs, found {raw_data.shape[1]}")
else:
    print(f"Dimension check passed: {expected_dim} columns found.")

TONGUE_DIMS = 8  # 4 control points (T4-T1), 2 coordinates each

# ==========================================
# 3. UN-Z-SCORE (UN-NORMALIZE) FOR TONGUE CONTROL POINTS
# Formula: Real = (Normalized * Std) + Mean
# ==========================================
if data_mean.shape[0] < TONGUE_DIMS or data_std.shape[0] < TONGUE_DIMS:
    raise ValueError(
        f"Mean/Std only provide {data_mean.shape[0]} dims, but {TONGUE_DIMS} required for tongue points."
    )

tongue_zscores = raw_data[:, :TONGUE_DIMS]
mu_tongue = data_mean[:TONGUE_DIMS]
std_tongue = data_std[:TONGUE_DIMS]
real_world_tongue = (tongue_zscores * std_tongue) + mu_tongue

print("\n--- Un-normalization Sample (Frame 0) ---")
print(f"Raw (Z-score): {tongue_zscores[0, :4]}")
print(f"Real World:    {real_world_tongue[0, :4]}")

# ==========================================
# 4. EXTRACT TONGUE POINTS (T4 - T1)
# Docs: T4_x, T4_y, T3_x, T3_y, T2_x, T2_y, T1_x, T1_y
# Indices: 0, 1,    2, 3,    4, 5,    6, 7
# ==========================================

# We already sliced to the first 8 columns above
# Optionally mirror across the Y-axis if you need the flipped orientation.
MIRROR_TONGUE = True
tongue_trajectory = real_world_tongue.reshape(-1, 4, 2)
if MIRROR_TONGUE:
    tongue_trajectory[:, :, 0] *= -1

print(f"\nExtracted Tongue Data Shape: {tongue_trajectory.shape}")
print("(Frames, Points, xy)")

# ==========================================
# 5. VISUALIZATION CHECK
# ==========================================
# Let's plot the MEAN position of the tongue to see if the shape makes sense.
# T4 is usually Back, T1 is Tip.

avg_positions = np.mean(tongue_trajectory, axis=0)

plt.figure(figsize=(8, 6))

# Plot the chain (T4 -> T3 -> T2 -> T1)
plt.plot(avg_positions[:, 0], avg_positions[:, 1], 'b-o', label='Mean Tongue Shape')

# Annotate points
labels = ['T4 (Back)', 'T3', 'T2', 'T1 (Tip)']
for i, txt in enumerate(labels):
    plt.text(avg_positions[i, 0], avg_positions[i, 1] + 0.5, txt, fontsize=9)

plt.title("Verification: Mean Tongue Shape from .npy")
plt.xlabel("X (Check Documentation for orientation)")
plt.ylabel("Y")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

print("\nVerification Complete. If the plot looks like a tongue curve, the mapping is correct.")

# ==========================================
# 6. MATPLOTLIB ANIMATION
# ==========================================

def animate_tongue(tongue_points, interval_ms=33, tail_length=10):
    """Create a simple 2D animation of tongue control points over time."""
    num_frames = tongue_points.shape[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    line, = ax.plot([], [], 'b-o', lw=2, label='Tongue Shape')
    scatter = ax.scatter([], [], color='#1f77b4', zorder=3)
    tail_line, = ax.plot([], [], 'r-', alpha=0.3, label='Tip Trail')

    y_coords = tongue_points[:, :, 0]
    z_coords = tongue_points[:, :, 1]
    margin = 5
    ax.set_xlim(np.min(y_coords) - margin, np.max(y_coords) + margin)
    ax.set_ylim(np.min(z_coords) - margin, np.max(z_coords) + margin)
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_title('Tongue Control-Point Animation (T4 -> T1)')
    ax.grid(True)
    ax.legend(loc='upper right')
    ax.set_aspect('equal', adjustable='box')

    def init():
        line.set_data([], [])
        scatter.set_offsets(np.zeros((4, 2)))
        tail_line.set_data([], [])
        return line, scatter, tail_line

    def update(frame_idx):
        pts = tongue_points[frame_idx]
        line.set_data(pts[:, 0], pts[:, 1])
        scatter.set_offsets(pts)

        if tail_length > 0:
            start = max(0, frame_idx - tail_length)
            tip_trail = tongue_points[start:frame_idx + 1, -1]  # T1 tip history
            tail_line.set_data(tip_trail[:, 0], tip_trail[:, 1])
        else:
            tail_line.set_data([], [])

        return line, scatter, tail_line

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        init_func=init,
        interval=interval_ms,
        blit=True,
        repeat=True,
    )

    plt.show()
    return ani


if __name__ == "__main__":
    animate_tongue(tongue_trajectory)
