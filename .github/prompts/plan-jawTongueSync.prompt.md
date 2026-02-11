## Plan: Jaw–Tongue Sync PoC (DRAFT)

Goal is to quantify jaw–tongue temporal alignment on a short clip using the WavLM tongue `.npy` + BEAT blendshape JSONs, run a global lead/lag shift test, and validate whether WER improves. This plan leans on existing render/metadata conventions in the face and tongue pipelines, and uses TextGrid phoneme intervals to anchor analysis. If the PoC shows improvement, we can extend to piecewise keyframe anchors and later time-warping. The intent is to produce a single analysis script plus a minimal alignment test in the render/AVSR pipeline.

**Steps**
1. Locate canonical sources and conventions for jaw and tongue signals in [tongue_scripts/run_batch_pipeline_speaker1.py](tongue_scripts/run_batch_pipeline_speaker1.py), and tongue_scripts/test.py and their referenced scripts; document the exact blendshape names and tongue axis conventions in [AGENT.md](AGENT.md).
2. Add a dedicated analysis script (e.g., [tongue_scripts/](tongue_scripts/)) that loads: BEAT blendshape JSONs (extract `jawOpen` + lip shapes), tongue `.npy` (use tongue Y/vertical tip or blade channel), and TextGrid intervals (phoneme selection per AGENT). Plot jaw vs tongue, highlight phoneme spans, compute Pearson correlation and lead/lag sweep, and export a small report.
3. Define the PoC alignment method: compute best global time shift from correlation peak, then apply shift to tongue (or jaw) in the render or evaluation path; keep everything else constant for the short 5–10s segment.
4. Run AutoAVSR on the shifted vs unshifted render to compare WER using the existing evaluation paths in /home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py and baseline notes in [crucial_progress_report/SESSION_2026_02_04_VSR_MOUTH_CORRECTED_7.5S.md](crucial_progress_report/SESSION_2026_02_04_VSR_MOUTH_CORRECTED_7.5S.md).
5. Record findings and next steps in [AGENT.md](AGENT.md): correlation values, best lag, WER delta, and whether to proceed to anchor-based keyframes or warp markers.

**Verification**
- Run the analysis script on one short segment and confirm: plots generated, correlation + lag values logged, and phoneme highlighting is visually correct.
- Render two videos (baseline vs shifted) and run AutoAVSR to produce WERs; compare against the baseline report.

**Decisions**
- Tongue motion source: WavLM inversion `.npy` from batch pipeline
- Face motion source: BEAT blendshape JSONs
- PoC scope: single speaker, short segment
- PoC sync method: global time shift only

If you want tweaks to the plan scope or the analysis metrics before implementation, tell me what to change.
