#!/usr/bin/env python3
"""Render and evaluate a small active-tongue parameter grid for one VOCASets clip."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_VOCASET_LINK_DIR = PROJECT_ROOT / "tests" / "vocaset_outputs" / "grid_search" / "videos"

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

from tongue_scripts.evaluation.evaluate_vsr_ver import evaluate_videos  # noqa: E402
from tongue_scripts.rendering.render_dual_tongue_comparison import (  # noqa: E402
    ANCHOR_INDICES,
    BONE_INDICES,
    STD_PATH,
    TONGUE_CONFIG,
    TONGUE_SLICE,
    apply_jawopen_offset_correction,
    merge_audio,
    render_video_with_dynamic_tongue,
    shift_sequence,
)
from tongue_scripts.tongue_animation.face_model_io_trimesh import (  # noqa: E402
    load_face_model_trimesh,
)
from tongue_scripts.tongue_animation.generate_tongue_animation import (  # noqa: E402
    FaceKitTongueRig,
    load_blendshape_json_sequence,
    load_ema_motion,
)


@dataclass(frozen=True)
class TongueGridParams:
    std_scalar: float
    shift_z: float
    rotation_deg: float = 5.0
    thickness: float = 1.2
    shift_y: float = 0.0


def build_grid(
    std_scalars: tuple[float, ...] = (0.15, 0.20, 0.25),
    shift_z_values: tuple[float, ...] = (-0.5, 0.0, 0.5),
    rotation_deg_values: tuple[float, ...] = (5.0,),
    thickness_values: tuple[float, ...] = (1.2,),
    shift_y_values: tuple[float, ...] = (0.0,),
) -> list[TongueGridParams]:
    return [
        TongueGridParams(
            std_scalar=std_scalar,
            shift_z=shift_z,
            rotation_deg=rotation_deg,
            thickness=thickness,
            shift_y=shift_y,
        )
        for std_scalar in std_scalars
        for shift_z in shift_z_values
        for rotation_deg in rotation_deg_values
        for thickness in thickness_values
        for shift_y in shift_y_values
    ]


def build_default_grid() -> list[TongueGridParams]:
    return build_grid()


def _format_float(value: float) -> str:
    sign = "m" if value < 0 else ""
    return sign + f"{abs(value):.2f}".replace(".", "p")


def param_slug(params: TongueGridParams) -> str:
    slug = f"std{_format_float(params.std_scalar)}_z{_format_float(params.shift_z)}"
    if params.rotation_deg != 5.0:
        slug += f"_rot{_format_float(params.rotation_deg)}"
    if params.thickness != 1.2:
        slug += f"_th{_format_float(params.thickness)}"
    if params.shift_y != 0.0:
        slug += f"_y{_format_float(params.shift_y)}"
    return slug


def replace_symlink(link: Path, target: Path) -> None:
    tmp_link = link.with_name(f".{link.name}.{os.getpid()}.tmp")
    tmp_link.unlink(missing_ok=True)
    tmp_link.symlink_to(target)
    os.replace(tmp_link, link)


def resample_ema_to_face_frames(ema_seq: np.ndarray, face_frames: int) -> np.ndarray:
    if len(ema_seq) == face_frames:
        return ema_seq

    from scipy.interpolate import interp1d

    duration = len(ema_seq) / 50.0
    x_source = np.linspace(0, duration, len(ema_seq))
    x_target = np.linspace(0, duration, face_frames)
    ema_flat = ema_seq.reshape(len(ema_seq), -1)
    ema_resampled = np.zeros((face_frames, ema_flat.shape[1]), dtype=np.float32)
    interp_kind = "cubic" if len(ema_seq) >= 4 else "linear"
    for i in range(ema_flat.shape[1]):
        ema_resampled[:, i] = interp1d(x_source, ema_flat[:, i], kind=interp_kind)(
            x_target
        )
    return ema_resampled.reshape(face_frames, 4, 3)


def render_grid(args: argparse.Namespace) -> list[tuple[TongueGridParams, Path]]:
    json_path = Path(args.json_path)
    audio_path = Path(args.audio_path)
    motion_path = Path(args.motion_path)
    out_root = Path(args.output_root)
    link_dir = Path(args.link_dir)
    clip_id = args.clip_id

    out_root.mkdir(parents=True, exist_ok=True)
    link_dir.mkdir(parents=True, exist_ok=True)

    face_model = load_face_model_trimesh(PROJECT_ROOT / "FaceXModel")
    face_seq = load_blendshape_json_sequence(
        json_path,
        face_model,
        source_fps=args.source_fps,
        target_fps=args.fps,
    )
    face_seq = apply_jawopen_offset_correction(face_seq, face_model)

    rendered: list[tuple[TongueGridParams, Path]] = []
    for params in build_grid(
        std_scalars=tuple(args.std_scalars),
        shift_z_values=tuple(args.shift_z_values),
        rotation_deg_values=tuple(args.rotation_deg_values),
        thickness_values=tuple(args.thickness_values),
        shift_y_values=tuple(args.shift_y_values),
    ):
        slug = param_slug(params)
        out_dir = out_root / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_video = out_dir / f"{clip_id}_{slug}_active_tongue.mp4"
        audio_video = out_dir / f"{clip_id}_{slug}_active_tongue_with_audio.mp4"

        if not audio_video.is_file() or args.force:
            config = dict(TONGUE_CONFIG)
            config.update(
                {
                    "std_scalar": params.std_scalar,
                    "shift_z": params.shift_z,
                    "rotation_deg": params.rotation_deg,
                    "thickness": params.thickness,
                    "shift_y": params.shift_y,
                }
            )
            tongue_rig = FaceKitTongueRig(
                face_model.neutral_verts,
                face_model.faces,
                TONGUE_SLICE,
                ANCHOR_INDICES,
                BONE_INDICES,
                config,
            )
            ema_seq = load_ema_motion(
                motion_path,
                STD_PATH,
                tongue_rig.anchors,
                config["std_scalar"],
            )
            ema_seq = resample_ema_to_face_frames(ema_seq, len(face_seq))
            if args.tongue_shift_seconds:
                shift_frames = int(round(args.tongue_shift_seconds * args.fps))
                ema_seq = shift_sequence(ema_seq, shift_frames)

            print(f"[RENDER] {slug} -> {audio_video}", flush=True)
            render_video_with_dynamic_tongue(
                face_model,
                face_seq,
                tongue_rig,
                ema_seq,
                str(raw_video),
                fps=args.fps,
            )
            merge_audio(str(raw_video), str(audio_path), str(audio_video))
        else:
            print(f"[SKIP] existing {audio_video}", flush=True)

        link = link_dir / f"vocaset_{clip_id}_{slug}_active_tongue_with_audio.mp4"
        replace_symlink(link, audio_video)
        rendered.append((params, audio_video))

    return rendered


def write_grid_summary(
    csv_path: Path,
    md_path: Path,
    plot_path: Path,
    rows: list[dict],
    params_by_video: dict[str, TongueGridParams],
    baseline_passive: Path | None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "video",
        "std_scalar",
        "shift_z",
        "rotation_deg",
        "thickness",
        "shift_y",
        "ver",
        "wer_norm",
        "wer_raw",
        "composite_index",
        "hypothesis",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            params = params_by_video.get(row["video"])
            writer.writerow(
                {
                    "rank": rank,
                    "video": row["video"],
                    "std_scalar": "" if params is None else params.std_scalar,
                    "shift_z": "" if params is None else params.shift_z,
                    "rotation_deg": "" if params is None else params.rotation_deg,
                    "thickness": "" if params is None else params.thickness,
                    "shift_y": "" if params is None else params.shift_y,
                    "ver": f"{row['ver']:.4f}",
                    "wer_norm": f"{row['wer_norm']:.4f}",
                    "wer_raw": f"{row['wer_raw']:.4f}",
                    "composite_index": f"{row['composite_index']:.4f}",
                    "hypothesis": row["hypothesis"],
                }
            )
    write_ver_plot(plot_path, rows, params_by_video)

    lines = [
        "# VOCASets Active Tongue Grid Search",
        "",
        f"- passive baseline: `{baseline_passive}`" if baseline_passive else "- passive baseline: not included",
        f"- csv: `{csv_path}`",
        f"- VER plot: `{plot_path}`",
        "",
        "| Rank | Video | std_scalar | shift_z | VER | WER(norm) | Composite |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        params = params_by_video.get(row["video"])
        lines.append(
            "| "
            f"{rank} | {Path(row['video']).name} | "
            f"{'' if params is None else params.std_scalar} | "
            f"{'' if params is None else params.shift_z} | "
            f"{row['ver']:.4f} | {row['wer_norm']:.4f} | {row['composite_index']:.4f} |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def write_ver_plot(
    plot_path: Path,
    rows: list[dict],
    params_by_video: dict[str, TongueGridParams],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    active_rows = [row for row in rows if row["video"] in params_by_video]
    passive_rows = [row for row in rows if row["video"] not in params_by_video]
    if not active_rows:
        return

    by_shift_z: dict[float, list[tuple[float, float]]] = {}
    for row in active_rows:
        params = params_by_video[row["video"]]
        by_shift_z.setdefault(params.shift_z, []).append((params.std_scalar, row["ver"]))

    fig, ax = plt.subplots(figsize=(9, 5))
    for shift_z, points in sorted(by_shift_z.items()):
        points = sorted(points)
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            linewidth=1.8,
            label=f"shift_z={shift_z:g}",
        )

    if passive_rows:
        passive_best = min(row["ver"] for row in passive_rows)
        ax.axhline(
            passive_best,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"passive VER={passive_best:.4f}",
        )

    ax.set_title("VOCASets Active Tongue Grid Search: VER vs std_scalar")
    ax.set_xlabel("std_scalar")
    ax.set_ylabel("VER (lower is better)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def metric_path_for(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}_metrics.json")


def coerce_metric_row(row: dict) -> dict:
    coerced = dict(row)
    for key in ("ver", "wer_raw", "wer_norm", "composite_index", "viseme_accuracy", "word_accuracy"):
        if key in coerced and isinstance(coerced[key], str) and coerced[key] != "":
            coerced[key] = float(coerced[key])
    if "hyp_words" in coerced and isinstance(coerced["hyp_words"], str) and coerced["hyp_words"] != "":
        coerced["hyp_words"] = int(coerced["hyp_words"])
    return coerced


def load_metric(metric_path: Path) -> dict | None:
    if not metric_path.is_file():
        return None
    return coerce_metric_row(json.loads(metric_path.read_text(encoding="utf-8")))


def save_metric(metric_path: Path, row: dict, params: TongueGridParams | None) -> None:
    payload = dict(row)
    if params is not None:
        payload["params"] = {
            "std_scalar": params.std_scalar,
            "shift_z": params.shift_z,
            "rotation_deg": params.rotation_deg,
            "thickness": params.thickness,
            "shift_y": params.shift_y,
        }
    else:
        payload["params"] = None
    metric_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def evaluate_with_durable_metrics(
    args: argparse.Namespace,
    rendered: list[tuple[TongueGridParams, Path]],
    passive_baseline: Path | None,
) -> list[dict]:
    rows: list[dict] = []
    videos_with_params: list[tuple[Path, TongueGridParams | None]] = [
        (path, params) for params, path in rendered
    ]
    if passive_baseline and passive_baseline.is_file():
        videos_with_params.append((passive_baseline, None))

    for video_path, params in videos_with_params:
        metric_path = metric_path_for(video_path)
        if metric_path.is_file() and not args.force_eval:
            row = load_metric(metric_path)
            if row is not None:
                rows.append(row)
                print(f"[SKIP] existing metrics {metric_path}", flush=True)
                continue

        eval_args = argparse.Namespace(
            videos=[str(video_path)],
            ground_truth=args.ground_truth,
            infer_script=str(
                PROJECT_ROOT
                / "ADFA_EVALUATION"
                / "Visual_Speech_Recognition_for_Multiple_Languages"
                / "infer.py"
            ),
            infer_pipeline_script=str(
                PROJECT_ROOT
                / "ADFA_EVALUATION"
                / "Visual_Speech_Recognition_for_Multiple_Languages"
                / "infer_pipeline.py"
            ),
            infer_mode=args.infer_mode,
            textgrid_path=args.textgrid_path,
            seg_target_seconds=8.0,
            seg_min_seconds=4.0,
            seg_max_seconds=12.0,
            seg_min_silence=0.0,
            config_filename=args.config_filename,
            vowel_mode="grouped",
            detector=args.detector,
            python_bin=args.python_bin,
            experiment_name=f"{args.clip_id}_active_tongue_grid",
            report_path=str(Path(args.output_root) / "adfa_grid_search_report.md"),
            report_mode="append",
            dataset_id=args.clip_id,
            speaker_id=None,
            hypothesis="active tongue parameter grid can outperform passive baseline",
        )
        _, one_rows = evaluate_videos(eval_args)
        row = one_rows[0]
        save_metric(metric_path, row, params)
        rows.append(row)

        params_by_video = {str(path): p for p, path in rendered}
        write_grid_summary(
            csv_path=Path(args.output_root) / "grid_search_results.csv",
            md_path=Path(args.output_root) / "grid_search_summary.md",
            plot_path=Path(args.output_root) / "grid_search_ver_plot.png",
            rows=sorted(rows, key=lambda r: r["composite_index"]),
            params_by_video=params_by_video,
            baseline_passive=passive_baseline,
        )

    return sorted(rows, key=lambda r: r["composite_index"])


def parse_args() -> argparse.Namespace:
    clip_id = "FaceTalk_170725_00137_TA_sentence01"
    out_root = Path(
        "/research/milsrg1/user_workspace/ht467/smirk_task/outputs/vocasets_grid_search"
    )
    parser = argparse.ArgumentParser(
        description="Render/evaluate a small active-tongue parameter grid for a VOCASets clip."
    )
    parser.add_argument("--clip-id", default=clip_id)
    parser.add_argument(
        "--json-path",
        default=str(
            PROJECT_ROOT
            / "tests"
            / "vocasets"
            / "blendshape_json"
            / "FaceTalk_170725_00137_TA"
            / "sentence01.json"
        ),
    )
    parser.add_argument(
        "--audio-path",
        default=str(PROJECT_ROOT / "tests" / "vocasets" / "wav" / f"{clip_id}.wav"),
    )
    parser.add_argument(
        "--motion-path",
        default=str(
            Path("/research/milsrg1/user_workspace/ht467/smirk_task/outputs/vocasets")
            / clip_id
            / "tongue_motion"
            / "tongue_motion.npy"
        ),
    )
    parser.add_argument(
        "--ground-truth",
        default=str(
            Path("/research/milsrg1/user_workspace/ht467/smirk_task/outputs/vocasets")
            / clip_id
            / "ground_truth_sentence01.txt"
        ),
    )
    parser.add_argument("--output-root", default=str(out_root / clip_id))
    parser.add_argument("--link-dir", default=str(DEFAULT_VOCASET_LINK_DIR))
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--source-fps", type=float, default=60.0)
    parser.add_argument("--tongue-shift-seconds", type=float, default=0.0)
    parser.add_argument("--std-scalars", nargs="+", type=float, default=[0.15, 0.20, 0.25])
    parser.add_argument("--shift-z-values", nargs="+", type=float, default=[-0.5, 0.0, 0.5])
    parser.add_argument("--rotation-deg-values", nargs="+", type=float, default=[5.0])
    parser.add_argument("--thickness-values", nargs="+", type=float, default=[1.2])
    parser.add_argument("--shift-y-values", nargs="+", type=float, default=[0.0])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument(
        "--passive-baseline",
        default=str(
            Path(
                "/research/milsrg1/user_workspace/ht467/smirk_task/outputs/vocasets_passive"
            )
            / "FaceTalk_170725_00137_TA"
            / "sentence01"
            / f"{clip_id}_passive_tongue_with_audio.mp4"
        ),
    )
    parser.add_argument(
        "--python-bin",
        default="/research/milsrg1/user_workspace/ht467/tools/uv/adfa-vsr/bin/python",
    )
    parser.add_argument("--infer-mode", choices=["full", "segmented"], default="full")
    parser.add_argument("--textgrid-path", default=None)
    parser.add_argument("--config-filename", default="configs/LRS3_V_WER19.1.ini")
    parser.add_argument("--detector", default="mediapipe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rendered = render_grid(args)
    if args.render_only:
        return

    passive_baseline = Path(args.passive_baseline) if args.passive_baseline else None
    rows_sorted = evaluate_with_durable_metrics(args, rendered, passive_baseline)

    params_by_video = {str(path): params for params, path in rendered}
    write_grid_summary(
        csv_path=Path(args.output_root) / "grid_search_results.csv",
        md_path=Path(args.output_root) / "grid_search_summary.md",
        plot_path=Path(args.output_root) / "grid_search_ver_plot.png",
        rows=rows_sorted,
        params_by_video=params_by_video,
        baseline_passive=passive_baseline,
    )


if __name__ == "__main__":
    main()
