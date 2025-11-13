import bpy
import numpy as np
from pathlib import Path

# --- 1. USER SETTINGS: POINT THESE TO YOUR DATA ---

DATA_DIR = Path("/home/timoite/Documents/optimization_project/data")
NPY_FILE_PATH = DATA_DIR / "26_reamey_0_1_1.npy"
MU_FILE_PATH = DATA_DIR / "JW13_4points_mu.npy"
STD_FILE_PATH = DATA_DIR / "JW13_4points_std.npy"

ANIMATION_SCALE = 0.09

# The name of your Armature object in Blender
ARMATURE_NAME = "Armature"

# The names of your 3 tongue bones, from ROOT to TIP
BONE_ROOT_NAME = "RootBack"
BONE_MID_NAME = "BackMid"
BONE_TIP_NAME = "MidTip"


# --- 2. SCRIPT LOGIC ---

TONGUE_CHANNELS = [
    "T4_x",
    "T4_y",
    "T3_x",
    "T3_y",
    "T2_x",
    "T2_y",
    "T1_x",
    "T1_y",
]
TONGUE_DIMS = len(TONGUE_CHANNELS)


def load_tongue_animation(data_path, mu_path=None, std_path=None):
    """Load and un-normalise the tongue channels, discarding other articulators."""
    data_path = Path(data_path)
    data = np.load(data_path)

    if data.ndim != 2 or data.shape[1] < TONGUE_DIMS:
        raise ValueError(
            f"Expected (frames, >= {TONGUE_DIMS}) array, got {data.shape}"
        )

    tongue_zscores = data[:, :TONGUE_DIMS]

    if mu_path and std_path:
        mu = np.squeeze(np.load(Path(mu_path)))
        std = np.squeeze(np.load(Path(std_path)))

        if mu.size < TONGUE_DIMS or std.size < TONGUE_DIMS:
            raise ValueError(
                "Mean/std arrays must provide at least the tongue channels; "
                f"mu size={mu.size}, std size={std.size}"
            )

        mu_tongue = mu.flat[:TONGUE_DIMS]
        std_tongue = std.flat[:TONGUE_DIMS]
    else:
        mu_tongue = np.zeros(TONGUE_DIMS)
        std_tongue = np.ones(TONGUE_DIMS)

    tongue_coords = (tongue_zscores * std_tongue) + mu_tongue
    coords_reshaped = tongue_coords.reshape(-1, 4, 2)
    mu_reshaped = mu_tongue.reshape(4, 2)
    std_reshaped = std_tongue.reshape(4, 2)

    return coords_reshaped, mu_reshaped, std_reshaped


def get_rest_pose_info(armature_name, b1_name, b2_name, b3_name):
    """Read tongue rest pose coordinates from the armature."""
    try:
        armature_obj = bpy.data.objects[armature_name]
    except KeyError:
        print(f"ERROR: Armature object named '{armature_name}' not found.")
        print("Check your ARMATURE_NAME setting.")
        return None, None

    mat_world = armature_obj.matrix_world
    
    try:
        b1 = armature_obj.data.bones[b1_name]
        b2 = armature_obj.data.bones[b2_name]
        b3 = armature_obj.data.bones[b3_name]
    except KeyError as e:
        print(f"ERROR: Bone not found: {e}. Check your BONE names.")
        return None, None

    # Get world-space coordinates of the 4 points
    p4_rest = mat_world @ b1.head_local
    p3_rest = mat_world @ b1.tail_local
    p2_rest = mat_world @ b2.tail_local
    p1_rest = mat_world @ b3.tail_local
    
    rest_points = [p4_rest, p3_rest, p2_rest, p1_rest]
    
    # Create the 8-column MEAN array for the Y-Z plane
    # Data 'x' -> Blender 'Y' (forward/backward)
    # Data 'y' -> Blender 'Z' (up/down)
    means = np.array([
        p4_rest.y, p4_rest.z,
        p3_rest.y, p3_rest.z,
        p2_rest.y, p2_rest.z,
        p1_rest.y, p1_rest.z
    ])
    
    print(f"Successfully read rest pose MEANS (Y, Z):\n{means}")
    return means, rest_points

def create_and_animate_targets(tongue_positions, mu_tongue, rest_points, animation_scale):
    """Animate tongue targets in the Y-Z plane using only tongue endpoints."""
    num_frames = tongue_positions.shape[0]
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = num_frames - 1
    
    target_names = []
    targets = []
    
    # Create 4 Empties
    for i in range(4, 0, -1):  # T4, T3, T2, T1
        name = f"T{i}_Target"
        target_names.append(name)
        if name not in bpy.data.objects:
            bpy.ops.object.empty_add(type='CUBE', location=(0, 0, 0))
            bpy.context.object.name = name
            bpy.context.object.empty_display_size = 0.02 # Make them small
        targets.append(bpy.data.objects[name])
        
    print(f"Animating {num_frames} frames for 4 targets...")

    rest_yz = np.array([[p.y, p.z] for p in rest_points])
    mu_yz = mu_tongue

    # Loop through every frame of data
    for frame_idx in range(num_frames):
        frame_positions = tongue_positions[frame_idx]
        blended_yz = rest_yz + animation_scale * (frame_positions - mu_yz)

        # Apply to targets
        for point_idx in range(4):  # 0=T4, 1=T3, 2=T2, 3=T1
            target = targets[point_idx]

            coord_x = rest_points[point_idx].x
            coord_y, coord_z = blended_yz[point_idx]

            target.location = (coord_x, coord_y, coord_z)
            target.keyframe_insert(data_path="location", frame=frame_idx)
            
        if frame_idx % 500 == 0:
            print(f"Processed frame {frame_idx} / {num_frames}")
            
    print("Animation complete! Empties are keyframed.")
    return target_names

def connect_bones_to_targets(armature_name, b1_name, b2_name, b3_name, target_names):
    """
    --- THIS IS THE AUTO-CONNECT FIX ---
    Applies constraints to the bones to be driven by the targets.
    """
    try:
        arm_obj = bpy.data.objects[armature_name]
    except KeyError:
        print(f"ERROR: Could not find armature '{armature_name}' to connect.")
        return

    print("Connecting bones to animated targets...")
    
    # Switch to Pose Mode
    if bpy.context.active_object != arm_obj:
        bpy.ops.object.select_all(action='DESELECT')
        arm_obj.select_set(True)
        bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    # Get the Pose Bones
    try:
        p_b1 = arm_obj.pose.bones[b1_name]
        p_b2 = arm_obj.pose.bones[b2_name]
        p_b3 = arm_obj.pose.bones[b3_name]
    except KeyError as e:
        print(f"ERROR: Could not find pose bone {e}. Aborting connection.")
        bpy.ops.object.mode_set(mode='OBJECT')
        return

    # Clear all existing constraints
    for bone in [p_b1, p_b2, p_b3]:
        for c in bone.constraints:
            bone.constraints.remove(c)
    
    # Get target objects
    t4_target = bpy.data.objects[target_names[0]] # T4_Target
    t3_target = bpy.data.objects[target_names[1]] # T3_Target
    t2_target = bpy.data.objects[target_names[2]] # T2_Target
    t1_target = bpy.data.objects[target_names[3]] # T1_Target
    
    # --- Apply new constraints ---
    
    # 1. Bone 1 (Root): Lock its head to T4, stretch its tail to T3
    loc_c = p_b1.constraints.new(type='COPY_LOCATION')
    loc_c.target = t4_target
    
    stretch_c = p_b1.constraints.new(type='STRETCH_TO')
    stretch_c.target = t3_target
    
    # 2. Bone 2 (Mid): Connects to tail of Bone 1, stretches to T2
    stretch_c_2 = p_b2.constraints.new(type='STRETCH_TO')
    stretch_c_2.target = t2_target
    
    # 3. Bone 3 (Tip): Connects to tail of Bone 2, stretches to T1
    stretch_c_3 = p_b3.constraints.new(type='STRETCH_TO')
    stretch_c_3.target = t1_target
    
    # Switch back to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')
    print("SUCCESS: Armature is now connected to animated targets.")


# --- 3. MAIN SCRIPT EXECUTION ---
def main():
    # Add a clear print statement at the very beginning
    print("--- Running Tongue Animation Script ---")
    
    # Ensure we are in Object Mode
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # 1. Get the MEANS and Rest Coords from the armature's rest pose
    means, rest_points = get_rest_pose_info(ARMATURE_NAME, BONE_ROOT_NAME, BONE_MID_NAME, BONE_TIP_NAME)
    
    if means is None:
        print("Script failed: Could not get rest pose.")
        return

    # 2. Load the .npy data
    try:
        tongue_positions, mu_tongue, std_tongue = load_tongue_animation(
            NPY_FILE_PATH,
            MU_FILE_PATH,
            STD_FILE_PATH,
        )
        print(f"Loaded {tongue_positions.shape[0]} frames of tongue animation data")
        print("Tongue std (dataset frame)\n", std_tongue)
    except Exception as e:
        print(f"ERROR: Could not load .npy file at {NPY_FILE_PATH}")
        print(e)
        return

    # 3. Create and animate the targets
    target_names = create_and_animate_targets(
        tongue_positions,
        mu_tongue,
        rest_points,
        ANIMATION_SCALE,
    )

    # 4. Connect the bones to the new targets
    connect_bones_to_targets(ARMATURE_NAME, BONE_ROOT_NAME, BONE_MID_NAME, BONE_TIP_NAME, target_names)
    
    print("--- Script Finished ---")

# Run the main function
main()