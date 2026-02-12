# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "matplotlib>=3.8",
#     "scipy>=1.11",
#     "trimesh",
# ]
# ///
"""
Compare actual tongue EMA motion against manually-defined ground truth.

For every GT keyframe, computes per-anchor error between the GT (Y,Z)
position and the actual EMA-driven position.  Then sweeps a global time
shift to find the lag that minimises total error.

Outputs
-------
- JSON report with per-phoneme-class errors, best global shift, error
  reduction.
- Optional PNG plot of error-vs-shift curve and a few example overlays.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
TONGUE_SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TONGUE_SCRIPTS_DIR.parent
TONGUE_ANIMATION_DIR = TONGUE_SCRIPTS_DIR / "tongue_animation"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TONGUE_ANIMATION_DIR) not in sys.path:
    sys.path.insert(0, str(TONGUE_ANIMATION_DIR))

from face_model_io_trimesh import load_face_model_trimesh
from generate_tongue_animation import (
    load_ema_motion,
    FaceKitTongueRig,
    TONGUE_CONFIG,
)

TONGUE_SLICE = slice(16611, 17039)
ANCHOR_INDICES = [16661, 16696, 16755, 16758]
BONE_INDICES = [16661, 16757]
ANCHOR_NAMES = ["T4 (Back)", "T3 (Dorsum)", "T2 (Blade)", "T1 (Tip)"]
FPS = 50


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _anchor_error(gt_yz: List[List[float]], ema_yz: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean Euclidean distance across 4 anchors in (Y,Z) space.

    Returns (mean_dist, per_anchor_dists).
    """
    gt = np.asarray(gt_yz, dtype=np.float64)  # (4, 2)
    d = np.sqrt(np.sum((gt - ema_yz) ** 2, axis=1))  # (4,)
    return float(np.mean(d)), d


def _shifted_error(
    targets: list,
    ema_seq: np.ndarray,
    shift: int,
) -> Tuple[float, int]:
    """Total mean-anchor error when the entire EMA sequence is shifted
    by *shift* frames (positive = tongue delayed)."""
    total = 0.0
    n = 0
    for tgt in targets:
        for kf in tgt["keyframes"]:
            f = kf["frame_idx"] + shift
            if 0 <= f < len(ema_seq):
                ema_yz = ema_seq[f, :, [1, 2]]
                err, _ = _anchor_error(kf["anchors_yz"], ema_yz)
                total += err
                n += 1
    return total / n if n > 0 else float("inf"), n


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare tongue EMA vs manual GT")
    parser.add_argument(
        "--gt-json",
        default=str(TONGUE_SCRIPTS_DIR / "jaw_tongue_sync" / "1_wayne_0_112_112_tongue_gt.json"),
        help="Path to GT JSON produced by tongue_gt_editor.py",
    )
    parser.add_argument("--tongue-npy-dir", default=str(TONGUE_SCRIPTS_DIR / "outputs"))
    parser.add_argument(
        "--std-path",
        default=str(TONGUE_SCRIPTS_DIR / "normalising_vectors" / "JW13_4points_std.npy"),
    )
    parser.add_argument("--face-model-dir", default=str(PROJECT_ROOT / "FaceXModel"))
    parser.add_argument("--output-dir", default=str(TONGUE_SCRIPTS_DIR / "jaw_tongue_sync"))
    parser.add_argument("--max-shift-s", type=float, default=0.5, help="Max shift in seconds")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    gt_path = Path(args.gt_json)
    if not gt_path.exists():
        raise FileNotFoundError(f"GT JSON not found: {gt_path}")
    gt = json.loads(gt_path.read_text(encoding="utf-8"))

    dataset_id = gt["dataset_id"]
    npy_path = Path(args.tongue_npy_dir) / f"{dataset_id}.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Tongue .npy not found: {npy_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load EMA
    print("Loading face model + EMA …")
    face_model = load_face_model_trimesh(args.face_model_dir)
    tongue_rig = FaceKitTongueRig(
        face_model.neutral_verts,
        face_model.faces,
        TONGUE_SLICE,
        ANCHOR_INDICES,
        BONE_INDICES,
        TONGUE_CONFIG,
    )
    ema_seq = load_ema_motion(str(npy_path), args.std_path, tongue_rig.anchors, TONGUE_CONFIG["std_scalar"])
    targets = gt.get("targets", [])
    if not targets:
        print("No GT targets found in JSON.")
        return

    n_kf = sum(len(t["keyframes"]) for t in targets)
    print(f"Loaded {n_kf} GT keyframes across {len(targets)} phone intervals.")

    # ------ per-class error at zero shift ------
    class_errors: Dict[str, List[float]] = {}
    per_anchor_errors: Dict[str, List[np.ndarray]] = {}

    for tgt in targets:
        phone = tgt["phone_normalized"]
        for kf in tgt["keyframes"]:
            f = kf["frame_idx"]
            if 0 <= f < len(ema_seq):
                ema_yz = ema_seq[f, :, [1, 2]]
                mean_err, per_anc = _anchor_error(kf["anchors_yz"], ema_yz)
                class_errors.setdefault(phone, []).append(mean_err)
                per_anchor_errors.setdefault(phone, []).append(per_anc)

    print("\n═══ Per-class error at zero shift ═══")
    for phone, errs in sorted(class_errors.items()):
        m = np.mean(errs)
        print(f"  /{phone}/:  mean={m:.3f}  (n={len(errs)} keyframes)")

    zero_err, _ = _shifted_error(targets, ema_seq, 0)
    print(f"\n  Overall mean error (shift=0): {zero_err:.4f}")

    # ------ lag sweep ------
    max_shift = int(args.max_shift_s * FPS)
    shifts = list(range(-max_shift, max_shift + 1))
    errors = []
    for s in shifts:
        e, _ = _shifted_error(targets, ema_seq, s)
        errors.append(e)

    errors_arr = np.array(errors)
    best_idx = int(np.argmin(errors_arr))
    best_shift = shifts[best_idx]
    best_err = float(errors_arr[best_idx])
    reduction = zero_err - best_err

    print(f"\n═══ Lag sweep ({-max_shift} … +{max_shift} frames) ═══")
    print(f"  Best shift: {best_shift} frames ({best_shift / FPS:.3f}s)")
    print(f"  Error at best shift: {best_err:.4f}")
    print(f"  Error reduction: {reduction:.4f} ({reduction / zero_err * 100:.1f}%)" if zero_err > 0 else "")

    # ------ per-class best shift (diagnostic) ------
    class_best_shifts: Dict[str, Tuple[int, float]] = {}
    for phone in class_errors:
        phone_targets = [t for t in targets if t["phone_normalized"] == phone]
        best_s, best_e = 0, float("inf")
        for s in shifts:
            e, _ = _shifted_error(phone_targets, ema_seq, s)
            if e < best_e:
                best_e = e
                best_s = s
        class_best_shifts[phone] = (best_s, best_e)

    print("\n═══ Per-class best shift (diagnostic) ═══")
    for phone, (bs, be) in sorted(class_best_shifts.items()):
        print(f"  /{phone}/:  best_shift={bs} ({bs / FPS:.3f}s)  error={be:.3f}")

    # ------ write report ------
    report = {
        "dataset_id": dataset_id,
        "n_targets": len(targets),
        "n_keyframes": n_kf,
        "fps": FPS,
        "max_shift_s": args.max_shift_s,
        "zero_shift_error": float(zero_err),
        "best_global_shift_frames": best_shift,
        "best_global_shift_s": float(best_shift / FPS),
        "best_shift_error": best_err,
        "error_reduction": float(reduction),
        "error_reduction_pct": float(reduction / zero_err * 100) if zero_err > 0 else 0.0,
        "per_class_zero_shift": {
            ph: {"mean_error": float(np.mean(errs)), "n_keyframes": len(errs)}
            for ph, errs in class_errors.items()
        },
        "per_class_best_shift": {
            ph: {"shift_frames": bs, "shift_s": float(bs / FPS), "error": float(be)}
            for ph, (bs, be) in class_best_shifts.items()
        },
    }
    report_path = output_dir / f"{dataset_id}_gt_compare.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport → {report_path}")

    # ------ plot ------
    if not args.no_plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # (a) Error vs shift
        ax = axes[0]
        ax.plot(np.array(shifts) / FPS, errors_arr, "k-", lw=1.5)
        ax.axvline(0, color="#888", ls="--", lw=1)
        ax.axvline(best_shift / FPS, color="red", ls="--", lw=1.5, label=f"best={best_shift / FPS:.3f}s")
        ax.set_xlabel("Shift (s)  [+] tongue delayed")
        ax.set_ylabel("Mean anchor error")
        ax.set_title("Error vs global tongue shift")
        ax.legend()

        # (b) Per-class bar chart (zero-shift error)
        ax = axes[1]
        phones = sorted(class_errors.keys())
        means = [np.mean(class_errors[p]) for p in phones]
        x = np.arange(len(phones))
        ax.bar(x, means, color="#5599dd")
        ax.set_xticks(x)
        ax.set_xticklabels([f"/{p}/" for p in phones], rotation=45, ha="right")
        ax.set_ylabel("Mean anchor error")
        ax.set_title("Per-class error (zero shift)")

        fig.tight_layout()
        plot_path = output_dir / f"{dataset_id}_gt_compare.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Plot → {plot_path}")


if __name__ == "__main__":
    main()
