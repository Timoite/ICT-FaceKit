# Tongue Movement Realism Improvement Plan

## Problem Statement

The current tongue animation exhibits a **translation artifact**: the root of the tongue (T4) has comparable movement freedom to other control points, causing the entire tongue to shift position rather than stretch/contract like real muscle tissue.

**Observed behavior**: Tongue appears to translate as a whole
**Desired behavior**: Tongue stretches/contracts from a stable base, with tip having maximum articulation freedom

### Current Movement Distribution

| Point | Avg Std (mobility) | Vertex Influence | Issue |
|-------|-------------------|------------------|-------|
| T4 Root | 3.71 (82.8% of avg) | 73.4% of vertices | Too mobile for an anchor |
| T3 Back | 4.15 (92.6%) | 89.6% | - |
| T2 Middle | 4.87 (108.5%) | 94.7% | - |
| T1 Tip | 5.21 (116.1%) | 77.3% | Appropriate |

---

## Proposed Solutions

### Solution 1: Hierarchical/Relative Movement Constraints (Architectural)

**Concept**: Transform from absolute world-space movement to relative parent-child movement.

**Current approach**:
```python
offset = frame_position - rest_position  # Each point moves independently
```

**Proposed approach**:
```python
# T4 moves minimally in world space (anchored)
# T3 moves relative to T4
# T2 moves relative to T3
# T1 moves relative to T2
offset_T4 = small_factor * (frame_T4 - rest_T4)
offset_T3 = offset_T4 + (frame_T3 - rest_T3)
offset_T2 = offset_T3 + (frame_T2 - rest_T2)
offset_T1 = offset_T2 + (frame_T1 - rest_T1)
```

**Benefits**:
- Natural "stretch from base" behavior
- Root stability is built into the model
- Cumulative offsets create realistic deformation

**Complexity**: High - requires modifying core animation formula

---

### Solution 2: Root Anchoring Penalty in Optimization

**Concept**: Add explicit penalty term that discourages T4 movement.

**Implementation** (in `optimize_mu_std.py`):
```python
# Add to objective function
lambda_root = 5.0  # Strong penalty
root_movement = np.linalg.norm(T4_current - T4_rest)
root_penalty = lambda_root * root_movement**2

total_loss = distance_error + std_reg + min_std_penalty + root_penalty
```

**Benefits**:
- Quick to implement
- Works within existing optimization framework
- Tunable via lambda parameter

**Complexity**: Low

---

### Solution 3: Differentiated Std Bounds by Control Point

**Concept**: Apply graduated movement bounds that restrict root more than tip.

**Current bounds** (same for all):
```python
std_bounds = [(0.5, 20) for _ in range(8)]
```

**Proposed bounds**:
```python
std_bounds = [
    (0.5, 2.0),   # T4 Root Y - nearly fixed
    (0.5, 2.0),   # T4 Root Z
    (0.5, 5.0),   # T3 Back Y - limited
    (0.5, 5.0),   # T3 Back Z
    (0.5, 10.0),  # T2 Middle Y - moderate
    (0.5, 10.0),  # T2 Middle Z
    (0.5, 20.0),  # T1 Tip Y - full range
    (0.5, 20.0),  # T1 Tip Z
]
```

**Benefits**:
- Very easy to implement
- Hard constraint (cannot be violated)
- Directly addresses the mobility imbalance

**Complexity**: Very low

---

### Solution 4: Length Preservation Constraint

**Concept**: Penalize changes in distance between adjacent control points to simulate muscle incompressibility.

**Implementation**:
```python
# Rest distances between adjacent points
rest_lengths = [
    np.linalg.norm(rest_T3 - rest_T4),  # T4-T3
    np.linalg.norm(rest_T2 - rest_T3),  # T3-T2
    np.linalg.norm(rest_T1 - rest_T2),  # T2-T1
]

# Current distances
current_lengths = [
    np.linalg.norm(current_T3 - current_T4),
    np.linalg.norm(current_T2 - current_T3),
    np.linalg.norm(current_T1 - current_T2),
]

# Penalty for length changes
lambda_length = 2.0
length_penalty = lambda_length * sum(
    (curr - rest)**2 for curr, rest in zip(current_lengths, rest_lengths)
)
```

**Benefits**:
- Physically motivated (constant volume approximation)
- Allows bending without stretching
- Natural-looking deformation

**Complexity**: Medium

---

### Solution 5: Weighted Displacement Scaling by Bone Depth

**Concept**: Apply position-dependent scaling factors instead of uniform `animation_scale`.

**Current**:
```python
animation_scale = 0.25  # Same for all
```

**Proposed**:
```python
scale_factors = {
    'T4_root': 0.05,   # 5x less movement than current
    'T3_back': 0.12,
    'T2_middle': 0.20,
    'T1_tip': 0.30,    # Slightly more than current
}
```

**Implementation location**: `proper_animation.py` TongueArmature class

**Benefits**:
- Simple parameter change
- Immediate visual impact
- Easy to tune

**Complexity**: Very low

---

## Implementation Priority

| Priority | Solution | Complexity | Impact | Status |
|----------|----------|------------|--------|--------|
| 1 | Solution 3 (Differentiated bounds) | Very low | High | ✅ IMPLEMENTED |
| 2 | Solution 5 (Weighted scaling) | Very low | High | ✅ IMPLEMENTED |
| 3 | Solution 2 (Root penalty) | Low | Medium | ✅ IMPLEMENTED |
| 4 | Solution 4 (Length preservation) | Medium | High | Pending |
| 5 | Solution 1 (Hierarchical) | High | Very high | Future |

### Implementation Details (2026-01-17)

**Solution 2 (Root Anchoring Penalty)** - `optimize_mu_std.py`:
- Added `ROOT_ANCHOR_WEIGHT = 3.0` constant
- Added root anchoring penalty to objective function that penalizes T4 movement
- Formula: `root_anchor_penalty = root_anchor_weight * ||T4_mu - T4_mu_initial||²`

**Solution 3 (Differentiated Bounds)** - `optimize_mu_std.py`:
- Added `STD_BOUNDS_BY_POINT` and `MU_BOUNDS_BY_POINT` dictionaries
- T4 Root: std=(0.5, 3.0), mu=(-20, 10) - nearly fixed
- T3 Back: std=(0.5, 6.0), mu=(-50, 30) - limited
- T2 Middle: std=(0.5, 12.0), mu=(-80, 40) - moderate
- T1 Tip: std=(0.5, 20.0), mu=(-100, 50) - full range

**Solution 5 (Weighted Displacement Scaling)** - `proper_animation.py`:
- Added `DEFAULT_PER_BONE_SCALES = [0.08, 0.15, 0.22, 0.32]` (T4→T1)
- Modified `TongueArmature` to accept `per_bone_scales` parameter
- Updated `deform()` to use per-bone scaling instead of uniform scale

---

## Features Integrated from `hugging` Branch

### Inverse Distance Weighting (IDW)
```python
def compute_idw_weights(vertices, control_points, power=3.0):
    """Compute weights dynamically from mesh geometry."""
    dists = cdist(vertices, control_points)
    weights = 1.0 / (dists + 1e-6) ** power
    weights /= weights.sum(axis=1, keepdims=True)
    return [weights[:, i] for i in range(4)]
```

**Usage**: `TongueArmature(..., use_idw_weights=True, idw_power=3.0)`

**Benefits**:
- No Blender dependency for weight export
- Weights computed from geometry, not pre-baked
- `power` parameter controls locality (higher = more local influence)

### Simple Denormalization Mode
```python
# Standard mode (our default):
real = (z * std * scale) + mu

# Simple mode (from hugging, optional):
real = (z * std * scale) + rest_pos
```

**Usage**: `TongueArmature(..., use_simple_denorm=True)`

**Benefits**:
- Uses rest control point positions as mean
- Removes need for separate mu calibration
- Simpler data pipeline

### Other hugging Features (Not Yet Integrated)
- **WavLM + LoRA audio inversion** - Predicts tongue from audio
- **8Hz low-pass filtering** - Smooths predictions
- **pyrender + FFmpeg** - Direct video rendering (we use OBJ export)

---

## Branch Considerations: `hugging` vs `master`

### Current State

**master branch**:
- 1 commit ahead: `280117a position calibrate`
- Contains current optimization_project with L-BFGS-B pipeline

**hugging branch** (20 commits ahead):
- More developed rendering pipeline with PyTorch3D
- FFmpeg direct video output
- Batch processing for BEAT dataset
- Different file organization (`tongue/` → `data/`)
- New scripts in `tongue_scripts/`
- Uses inverse-distance weighting instead of Blender bone weights

### Key Differences in Tongue Animation

| Aspect | master | hugging |
|--------|--------|---------|
| Vertex weights | Blender-exported bone weights | Computed IDW (power of 3) |
| STD_SCALE | 0.25 | 0.2 |
| Control point positions | From Blender armature | Hardcoded in script |
| Rendering | OBJ export | PyTorch3D + FFmpeg |
| Data location | `tongue/`, `data/` | `data/`, `tongue_scripts/` |

### Merge Strategy Options

**Option A: Implement on master, merge later**
- Pros: Avoid conflicts now, focused development
- Cons: May need to re-implement on hugging

**Option B: Switch to hugging, implement there**
- Pros: Better rendering pipeline, batch processing ready
- Cons: Need to adapt optimization code to IDW weights

**Option C: Merge hugging into master first, then implement**
- Pros: Unified codebase
- Cons: Potential conflicts in `data/`, `tongue/` paths, and weight systems

### Recommended Approach

1. **Create a feature branch** from master: `feature/tongue-realism`
2. **Implement solutions 3 and 5** (quick wins) on feature branch
3. **Test and validate** the improvements
4. **Then evaluate** whether to port to hugging or merge hugging into master

This isolates the realism work from branch complexity.

---

## Files to Modify

### For Solutions 2, 3, 4 (Optimization changes):
- `optimization_project/src/optimize_mu_std.py` - bounds and objective function

### For Solution 5 (Scaling):
- `optimization_project/src/proper_animation.py` - TongueArmature class

### For Solution 1 (Hierarchical - future):
- `optimization_project/src/armature_logic.py` - core deformation logic
- `optimization_project/src/proper_animation.py` - animation formula

---

## Validation Approach

1. Render comparison videos: before vs after each solution
2. Visual inspection for:
   - Root stability (T4 should stay relatively fixed)
   - Natural stretch/contract behavior
   - Articulation quality at phoneme targets
3. Quantitative metrics:
   - Root displacement variance (should decrease)
   - Tip-to-target distance (should remain good)
   - Inter-point length variance (should decrease with solution 4)

---

## Notes

- Document created: 2026-01-17
- Related to: ICT-FaceKit tongue morphable blendshape standardization
- Context: optimization_project submodule
