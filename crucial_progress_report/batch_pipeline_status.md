# Technical Progress Report: Speaker 1 Batch Pipeline & Grid Search

## 1. Batch Pipeline Implementation
**Script:** `tongue_scripts/run_batch_pipeline_speaker1.py`

*   **Objective:** Automated processing of the full Speaker 1 dataset (~130 files).
*   **Architecture:**
    *   **Audio Loading:** Uses `scipy.io.wavfile` to bypass `soundfile` dependency issues in the current venv. Includes normalization for `int16` inputs.
    *   **Inference:** Uses `WavLMWrapper` with LoRA checkpoint (`lora_multispeaker_consistency...`) to predict EMA tongue motion coordinates.
    *   **Duration:** output truncated to **7.5s** per video for optimization.
    *   **Execution:** Runs via `uv run` with PEP 723 metadata handling dependencies (`torch`, `trimesh`, `pyrender`, etc.).

## 2. Rendering Module
**Module:** `tongue_scripts/batch_render_corrected.py`

*   **Refactoring:** Converted from a standalone script to a callable module (`render_bright_video`) accepting dynamic input/output paths.
*   **Visual Style:**
    *   **Skin Material:** "Humane" preset (Base: `[0.82, 0.65, 0.55]`, Roughness: 0.65, Metallic: 0.0).
    *   **Camera:** Dynamic orbital camera ("Moving Camera") for 3D depth visualization.
*   **Lip Correction:**
    *   Implemented systematic **Jaw Correction**: Shifts the `jawOpen` blendshape curve so the sequence minimum aligns with 0.0.
    *   Replaces previous manual bias approach, ensuring mouth closure without hardcoded offsets.

## 3. Grid Search & Validation Tool
**Script:** `tongue_scripts/test_tongue_grid_search_25fps.py`

*   **Purpose:** Parameter tuning (`rotation`, `thickness`, `std_scalar`) and visual verification.
*   **Multi-View Generation:** Produces three simultaneous renders per permutation:
    1.  **`vsr_input.mp4`**: Static Frontal View (Optimized for VSR inference).
    2.  **`visualization.mp4`**: Gentle Circular Orbit (Frontal-focused, aesthetic validation).
    3.  **`low_angle.mp4`**: Static Low Angle (`[0, -5, 32]`) (Depth/Volume validation).
*   **Metrics:** Runs Pretrained VSR inference on **all three** video outputs to generate comparative transcripts.
*   **Consistency:** Includes the same `jawOpen` correction logic as the main pipeline to ensure comparable results.
