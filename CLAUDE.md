# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ICT-FaceKit is USC Institute for Creative Technologies' morphable face model toolkit. It provides a parametric 3D face model with:
- 26,719 vertices organized into 17 geometry components (face, head/neck, mouth socket, eyes, teeth, gums/tongue, etc.)
- 100 PCA identity shape modes (identity000.obj - identity099.obj)
- 53 expression blend shapes (ARKit-compatible naming convention with _L/_R suffixes)
- Python SDK for face model manipulation
- Blender integration scripts for animation workflow

Python version: `>=3.9,<3.12` (strict requirement due to openmesh compatibility)

## Current Project Task

**Goal**: Implement and optimize tongue animation using linear blend skinning with PyTorch rendering.

### Phase 1: Face Animation Validation
- Animate face using ICT-FaceKit code with BEAT dataset blendshapes
- Render using PyTorch renderer to verify camera and lighting code works correctly
- Establish baseline rendering pipeline for facial animation

### Phase 2: Tongue Linear Blend Skinning Implementation
- Implement linear blend skinning (LBS) for tongue deformation from armature positions
- **Critical requirement**: Armature positions must be in vector format that's easy to adjust during optimization
- Storage format: Should enable direct manipulation of bone transforms (position/rotation/scale) for optimization
- Reference: `npybone_*_weights.npy` files contain bone weight data, `npybone_rest_positions.npy` contains rest pose

### Phase 3: Manual Testing and Rendering Verification
- Manually adjust armature vectors to test tongue deformation
- Render tongue with full face model to verify integration
- Render tongue independently to verify LBS implementation in isolation
- Validate that rendering pipeline and blend skinning work correctly for both cases

### Phase 4: Tongue Position Optimization
- Optimize tongue position based on established criteria (to be defined)
- Compare optimized results against baseline:
  - Visual comparison of rendered outputs
  - Quantitative comparison of tongue vertex positions
  - Validation against metrics established in Phase 3

**Current Status**: Project starting phase - establishing rendering and animation pipeline

## Development Commands

### Environment Setup
```bash
# Using uv (recommended - already configured via pyproject.toml)
uv sync

# or use uv with pip
uv pip install
```

### Guide to tools
- use "context7" whenever you need to search the documentation

### Running Scripts
**IMPORTANT**: Scripts must be run from `/Scripts` directory due to relative path dependencies.

```bash
cd Scripts

# Generate 10 random face identities
python sample_random.py

# Read identity coefficients from JSON and export mesh
python read_identity.py
```

Output files are written to `/sample_data_out/` directory.

### Blender Workflow
Blender scripts in `/Blender/Scripts/`:
- `ICTFaceKit.py` - Load ICT Face Model into Blender as shape keys
- `load_json.py` - Apply facial animation from JSON to Blender shape keys
- `load_tongue_animation.py` - Advanced tongue/armature animation loader

JSON animation format expects:
```json
{
  "names": ["browInnerUp", "eyeBlinkLeft", ...],
  "frames": [{"weights": [0.5, 0.3, ...]}, ...]
}
```

Note: JSON blendshape names must be mapped to Blender shape keys:
- `browInnerUp` → controls both `browInnerUp_L` and `browInnerUp_R`
- `eyeBlinkLeft` → `eyeBlink_L`
- `eyeBlinkRight` → `eyeBlink_R`

## Architecture

### Core Components

**FaceModel Class** (`Scripts/ict_face_model.py`)
- Parameterized face representation using identity and expression weights
- Deforms generic neutral mesh using linear shape modes
- Main workflow: `load_face_model()` → `from_coefficients()` or `randomize_identity()` → `deform_mesh()` → `write_deformed_mesh()`

**Face Model IO** (`Scripts/face_model_io.py`)
- Model loading/saving utilities
- Coefficient serialization (JSON format)
- OBJ mesh export using openmesh

**Data Directory Structure**
```
/FaceXModel/               # Core model data (405MB)
  generic_neutral_mesh.obj # Base topology (26,719 verts)
  identity000-099.obj      # 100 PCA identity modes
  browDown_L.obj           # Expression blend shapes (53 total)
  vertex_indices.json      # Vertex group definitions + expression list

/data/                     # Sample animation data
  *.json                   # BEAT dataset animation frames
  *.npy                    # NumPy binary data

/sample_data_out/          # Script output directory
```

### Face Model Coordinate System

The model uses **vertex indices** extensively for:
- Facial landmarks (Multi-PIE 68 points): indices like `1225, 1888, 1052...`
- Geometry regions: Face [0:9408], Teeth [17039:21450], Eyes [21451:24590], etc.
- Morphable vs. rigid vertex separation (see README tables)

**Key vertex ranges** (see README.md for complete table):
- Full face area: vertices 0-9408 (9,409 verts)
- Narrow face area: vertices 0-6705 (6,706 verts) - used for fitting
- Individual teeth: precisely indexed (e.g., Central incisor upper left: 18067-18142)

### Linear Morphable Model Math

Face deformation follows: `V_deformed = V_neutral + Σ(w_i × M_i)`

Where:
- `V_neutral` = generic_neutral_mesh vertices (N×3 array)
- `w_i` = identity or expression weight coefficients
- `M_i` = shape mode vectors (pre-computed PCA modes or artist-sculpted expressions)

Implementation in `FaceModel._compute_shape_modes()`:
```python
shape_mode = mesh_vertices - neutral_vertices  # Delta vectors
```

Then in `FaceModel.deform_mesh()`:
```python
deformed = neutral + Σ(identity_weights × identity_modes) + Σ(expression_weights × expression_modes)
```

### Expression System Details

**ARKit Compatibility**: Expression naming follows Apple ARKit with explicit L/R suffixes
- 53 expressions total (vs. ARKit's combined left/right)
- Separated: `browInnerUp_L` + `browInnerUp_R` instead of single `browInnerUp`
- Separated: `cheekPuff_L` + `cheekPuff_R` instead of single `cheekPuff`

**FACS Mapping**: See README.md for full correspondence between Action Units and expression shapes

**Incompatibilities**: Check `vertex_indices.json → incompatibilities` array for mutually exclusive expressions

### Blender Integration Pipeline

**Shape Key Creation** (`ICTFaceKit.py:loadICTFaceModel()`):
1. Load generic_neutral_mesh.obj as base object
2. Import all identity and expression OBJ files
3. Use `bpy.ops.object.join_shapes()` to create Blender shape keys
4. Delete temporary imported objects
5. Result: Single "ICTFaceModel" object with ~153 shape keys (100 identity + 53 expression)

**Animation Application** (`load_json.py:apply_facial_animation()`):
1. Read JSON with `names` and `frames` arrays
2. Map JSON blendshape names to Blender shape key names
3. Set `shape_key.value` and insert keyframes for each frame
4. Timeline automatically set to match animation length

### Data File Formats

**NPY Files** (NumPy binary):
- Armature rest positions, bone weights, vertex data
- Prefixed with `npy` in root directory (excluded from git via .gitignore)
- Load with `np.load(filepath)`

**JSON Configuration**:
- `vertex_indices.json` - Master model configuration
- Animation files - Frame-based blendshape weights

**OBJ Meshes**:
- All 26,719 vertices per file (same topology)
- Only vertex positions vary between identity/expression meshes

## Working with Face Model

### Typical Workflow
1. **Load Model**: `face_model = face_model_io.load_face_model('FaceXModel')`
2. **Set Parameters**:
   - Random: `face_model.randomize_identity()`
   - From coefficients: `face_model.from_coefficients(id_weights, ex_weights)`
3. **Deform**: `face_model.deform_mesh()`
4. **Export**: `face_model_io.write_deformed_mesh('output.obj', face_model)`

### Coordinate with Blender
- Ensure same expression weights are used in both Python SDK and Blender
- JSON export from Python can drive Blender animation via `load_json.py`
- Blender object name must be `ICTFaceModel` for animation scripts to work

## Common Pitfalls

1. **Path Dependencies**: Scripts expect to be run from `/Scripts/` directory - relative paths like `../FaceXModel` will fail otherwise
2. **Python Version**: openmesh==1.2.1 requires Python <3.12 - project strictly enforces `>=3.9,<3.12`
3. **Expression Name Mapping**: JSON animation data may use ARKit names without _L/_R suffixes - requires mapping layer (see `load_json.py:map_json_name_to_shape_key_names()`)
4. **Vertex Index Offsets**: All indices are 0-based, but Blender UI displays 1-based - account for off-by-one when debugging
5. **Large Files Excluded**: Blender .blend files are gitignored - model must be loaded fresh from FaceXModel OBJs each time

## Model Configuration Reference

**Expression List** (from vertex_indices.json):
- Brow: browDown_L/R, browInnerUp_L/R, browOuterUp_L/R
- Eyes: eyeBlink_L/R, eyeLookDown/In/Out/Up_L/R, eyeSquint_L/R, eyeWide_L/R
- Jaw: jawForward, jawLeft, jawOpen, jawRight
- Mouth: 28 mouth expressions (mouthSmile_L/R, mouthFunnel, etc.)
- Nose: noseSneer_L/R
- Special: cheekPuff_L/R, cheekSquint_L/R (separated from ARKit standard)

**Geometry Ordinals** (17 total):
| # | Name | Vertex Range | Count |
|---|------|--------------|-------|
| 0 | Face | 0:9408 | 9,409 |
| 1 | Head and Neck | 9409:11247 | 1,839 |
| 5 | Gums and tongue | 14062:17038 | 2,977 |
| 6 | Teeth | 17039:21450 | 4,412 |
| 7-8 | Eyeballs (L/R) | 21451:24590 | 3,140 |

(See README.md for complete table)
