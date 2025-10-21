import bpy
import json
import os

json_filepath = "/home/timoite/Downloads/26_reamey_0_15_15.json"
object_name = "ICTFaceModel" 



def map_json_name_to_shape_key_names(json_name):
    """
    Converts a JSON blendshape name to the corresponding
    Blender shape key name(s).
    Handles special cases where one JSON value drives two shape keys.
    """
    # Special case for 'browInnerUp' which controls both left and right sides
    if json_name == "browInnerUp":
        return ["browInnerUp_L", "browInnerUp_R"]
        
    # Standard name conversion
    if json_name.endswith("Left"):
        return [json_name.replace("Left", "_L")]
    elif json_name.endswith("Right"):
        return [json_name.replace("Right", "_R")]
    
    # If no side is specified, the name is likely the same
    return [json_name]


def apply_facial_animation(context, filepath, obj_name):
    """
    Reads animation data from a JSON file, maps the names correctly, and applies 
    it as shape key animations to the specified object in Blender.
    """
    obj = context.scene.objects.get(obj_name)
    if not obj:
        print(f"Error: Object named '{obj_name}' not found.")
        return
        
    if not obj.data.shape_keys:
        print(f"Error: Object '{obj_name}' has no shape keys.")
        return
        
    shape_keys = obj.data.shape_keys.key_blocks
    print(f"Found object '{obj_name}' with {len(shape_keys)} shape keys.")

    if not os.path.exists(filepath):
        print(f"Error: JSON file not found at '{filepath}'")
        return
        
    with open(filepath, 'r') as f:
        anim_data = json.load(f)
        
    json_blendshape_names = anim_data.get("names", [])
    frames_data = anim_data.get("frames", [])

    if not json_blendshape_names or not frames_data:
        print("Error: JSON is missing 'names' or 'frames' data.")
        return

    print(f"Loaded animation data with {len(json_blendshape_names)} blendshapes and {len(frames_data)} frames.")

    scene = context.scene

    scene.frame_end = len(frames_data)

    for i, frame in enumerate(frames_data):
        frame_num = i + 1  # Blender frames are 1-indexed
        scene.frame_set(frame_num)
        
        weights = frame.get("weights", [])
    

        # Iterate through the blendshapes from the JSON
        for json_name, weight_value in zip(json_blendshape_names, weights):
            
            # Get the mapped shape key name(s) for the current JSON name
            target_shape_key_names = map_json_name_to_shape_key_names(json_name)

            for shape_key_name in target_shape_key_names:
                # Find the matching shape key in the Blender object
                if shape_key_name in shape_keys:
                    shape_key = shape_keys[shape_key_name]
                    
                    # Set the value (influence) of the shape key
                    shape_key.value = weight_value
                    
                    # Insert a keyframe for this value at the current frame
                    shape_key.keyframe_insert(data_path="value", frame=frame_num)

    print("Animation complete! Keyframes have been inserted.")
    print(f"Timeline set from frame 1 to {scene.frame_end}.")

if __name__ == "__main__":
    apply_facial_animation(bpy.context, json_filepath, object_name)