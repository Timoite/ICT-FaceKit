# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Vision

**Final Goal**: Establish a new standard for ICT-FaceKit that adds **morphable blendshapes to the tongue** for improved speech animation quality. The current ICT-FaceKit only has facial blendshapes (51 ARKit-style); this project extends it with tongue articulation capabilities.

**Development Approach**:
- Pure Python workflow (moved away from Blender for easier scripting and optimization)
- Package management via **uv**
- Mesh handling via **trimesh** (openmesh deprecated due to difficult installation)
- Tongue mesh already separated and exported from Blender with armature data

**Evaluation**: Compare passive-tongue vs active-tongue rendered videos using Word Error Rate (WER) from a vision lip-reading model.

## Commands

```bash
# Package management (use uv)
uv sync                              # Install dependencies
uv run python Scripts/sample_random.py   # Run scripts

# Face model scripts
cd Scripts
python sample_random.py              # Generate random faces → sample_data_out/
python read_identity.py              # Load and deform from coefficients

# Optimization (in optimization_project/)
cd optimization_project
python src/optimize.py --config config/textgrid_targets.json --output output/optimized.npy
python src/proper_animation.py       # Render face + tongue animation
```

## Directory Structure

```
ICT-FaceKit/
├── Scripts/                    # Legacy face model code (uses openmesh)
│   ├── ict_face_model.py       # FaceModel class (PCA morphable model)
│   ├── face_model_io.py        # Load model, read coefficients, write OBJ
│   └── sample_random.py        # Example usage
│
├── FaceXModel/                 # Face model data (26,719 vertices)
│   ├── generic_neutral_mesh.obj
│   ├── identity000-099.obj     # 100 PCA identity modes
│   ├── *.obj                   # 51 expression blendshapes
│   └── vertex_indices.json     # Geometry config
│
├── tongue/                     # Separated tongue armature data
│   ├── npyarmature_metadata.json   # Bone names, structure
│   ├── npybone_*_weights.npy       # Vertex skinning weights (489 verts × 4 bones)
│   ├── npybone_rest_positions.npy  # Control point rest positions
│   └── npytongue_rest_vertices.npy # Tongue mesh rest pose
│
├── data/                       # Animation and speech data
│   ├── *.npy                   # Tongue params (z-scored)
│   ├── *.json                  # Facial blendshape weights
│   ├── *.TextGrid              # PRAAT phoneme alignment
│   └── JW13_4points_mu/std.npy # Normalization params
│
├── optimization_project/       # Tongue optimization pipeline (submodule)
│   └── (see below)
│
├── Blender/Scripts/            # Legacy Blender scripts (reference only)
├── sample_data/                # Example coefficients
└── sample_data_out/            # Generated mesh outputs
```

## Architecture

### Face Model (PCA-based)
- **Deformation**: `V = V_neutral + Σ(id_weight × id_mode) + Σ(ex_weight × ex_mode)`
- **Core**: `FaceModel` class in `Scripts/ict_face_model.py`
- 100 identity PCA modes + 51 expression blendshapes

### Tongue System (4-point Armature)
3-bone chain with linear blend skinning:

| Point | Bone | Description |
|-------|------|-------------|
| T4 (Root) | head of RootBack | Base of tongue |
| T3 (Back) | tail of RootBack | Mid-back |
| T2 (Middle) | tail of BackMid | Mid-front |
| T1 (Tip) | tail of MidTip | Tongue tip |

Animation data: 16D z-scored parameters → 4 bones × 4 values → blend skinning → 500 deformed vertices

### Geometry Reference
- **Full face**: 26,719 vertices
- **Gums and tongue region**: vertices [14062:17038], 2,977 vertices

---

## optimization_project/ - Tongue Optimization Pipeline

Complete L-BFGS-B optimization system using phoneme alignment to improve tongue articulation.

### Key Modules
| File | Purpose |
|------|---------|
| `src/optimize.py` | Main optimizer using scipy L-BFGS-B |
| `src/armature_logic.py` | 4-point armature with blend skinning |
| `src/objective_function.py` | Distance-to-palate loss computation |
| `src/textgrid_parser.py` | PRAAT TextGrid phoneme extraction |
| `src/target_config.py` | Phoneme → tongue region → palate target mapping |
| `src/proper_animation.py` | Face + tongue rendering to OBJ |
| `src/data_utils.py` | Z-score normalization/denormalization |

### Optimization Workflow
```
TextGrid (phoneme timing)
    ↓
Extract stop consonants (T, D, K, G, N, NG) → 156 targets
    ↓
Map phoneme → tongue region → palate target
  /t/, /d/, /n/ → tongue tip → alveolar ridge
  /k/, /g/, /ng/ → tongue dorsum → velum
    ↓
For each target frame:
    L-BFGS-B minimize: distance(tongue_region, palate_target) + λ×regularization
    ↓
Output: optimized 16D parameters with improved articulation
```

### Data in optimization_project/
```
data/
├── 26_reamey_0_1_1.npy          # Tongue: (4299, 16) z-scored
├── 26_reamey_0_1_1.json         # Face: (5159, 51) blendshapes
├── 26_reamey_0_1_1_50fps.json   # Face converted to 50fps
├── 26_reamey_0_1_1.TextGrid     # PRAAT phoneme timing
├── JW13_4points_mu.npy          # Mean (1, 14)
├── JW13_4points_std.npy         # Std (1, 14)
├── model/
│   ├── face_model.obj           # 4,314 vertices
│   └── tongue_model.obj         # 500 vertices
└── blender_export/
    ├── tongue_rest_vertices.npy # (500, 3)
    ├── bone_rest_positions.npy  # (4, 3)
    └── bone_*_weights.npy       # (500,) × 4 bones
```

### Configuration
- `config/textgrid_targets.json` - 156 phoneme optimization targets
- `config/test_targets.json` - Single target for testing

### Key Parameters
- **Animation FPS**: 50
- **Dimension**: 16D z-scored → 14D for normalization (2 locked)
- **Bounds**: ±5σ from mean
- **Regularization**: λ = 0.01 (prevents extreme deviations)

### Coordinate Systems

**OBJ Mesh Space** (face_model.obj, tongue_model.obj):
- X: Left-Right (0 = midline)
- Y: Front-Back (positive = nose direction, negative = throat)
- Z: Up-Down (positive = top of head, toward palate)

**Sagittal Visualization** (90° CCW rotation of YZ plane):
- Plot X = -OBJ_Z (left = high Z = toward palate)
- Plot Y = OBJ_Y (up = toward nose/front)

### Palate Target Positions (Verified from face_model.obj)

| Region | OBJ Coords (Y, Z) | Vertex ID | Phonemes | Description |
|--------|-------------------|-----------|----------|-------------|
| Alveolar ridge | (-2.45, 9.79) | 41950 | /t/, /d/, /n/ | Behind upper front teeth |
| Hard palate | (-1.27, 6.21) | 44369 | reference | Middle roof of mouth |
| Velum (soft palate) | (-2.40, 2.75) | 44723 | /k/, /g/, /ng/ | Back roof of mouth |

### Tongue Control Points (Verified from tongue_model.obj)

| Point | OBJ Coords (Y, Z) | Description |
|-------|-------------------|-------------|
| T1 Tip | (-3.80, 9.50) | Front of tongue, reaches alveolar ridge |
| T2 Middle | (-3.20, 7.80) | Mid section (dorsum) |
| T3 Back | (-3.40, 5.50) | Back-mid section |
| T4 Root | (-4.50, 3.50) | Base of tongue, near velum |

### Optimization Targets

For /t/, /d/, /n/ (alveolar): T1 (Tip) should reach alveolar ridge
- T1 rest: Z=9.50 → Target: Z=9.79 (need +0.29 upward movement)
- T1 rest: Y=-3.80 → Target: Y=-2.45 (need +1.35 forward movement)

For /k/, /g/, /ng/ (velar): T3/T4 should reach velum
- T4 rest: Z=3.50 → Target: Z=2.75 (need -0.75 movement)
- T4 rest: Y=-4.50 → Target: Y=-2.40 (need +2.10 forward movement)

### Example Commands
```bash
cd optimization_project

# Parse TextGrid and view phoneme summary
python -c "from src.textgrid_parser import ArticulationExtractor; e = ArticulationExtractor('data/26_reamey_0_1_1.TextGrid', fps=50.0); e.print_summary()"

# Run optimization
python src/optimize.py --config config/textgrid_targets.json --output output/optimized.npy --max-iter 100

# Render animation frames
python src/proper_animation.py
```

## Current Phase

Optimization pipeline is implemented. Next steps:
1. Optimize mu/std values using L-BFGS-B with phoneme contact constraints
2. Integrate optimized tongue back into full ICT-FaceKit model
3. Render comparison videos (passive vs active tongue)
4. Evaluate WER improvement with lip-reading model
5. Define morphable tongue blendshapes as new ICT-FaceKit standard
