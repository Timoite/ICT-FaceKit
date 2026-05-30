#!/usr/bin/env python3
"""Render and evaluate one VOCASets tongue color experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

LIGHT_RED_TONGUE_COLOR = (1.0, 0.6, 0.6, 1.0)
MID_RED_TONGUE_COLOR = (0.86, 0.34, 0.34, 1.0)
DARK_RED_TONGUE_COLOR = (0.72, 0.08, 0.08, 1.0)
COLOR_VARIANTS = {
    "light_red": {
        "slug": "lightred",
        "experiment": "light_red_std0p27_rot5",
        "color": LIGHT_RED_TONGUE_COLOR,
        "hypothesis": "light-red active tongue material may improve VSR visibility",
        "title": "VOCASets Light-Red Tongue Color Experiment",
    },
    "mid_red": {
        "slug": "midred",
        "experiment": "mid_red_std0p27_rot5",
        "color": MID_RED_TONGUE_COLOR,
        "hypothesis": "mid-red active tongue material may improve VSR visibility",
        "title": "VOCASets Mid-Red Tongue Color Experiment",
    },
    "dark_red": {
        "slug": "darkred",
        "experiment": "darker_red_std0p27_rot5",
        "color": DARK_RED_TONGUE_COLOR,
        "hypothesis": "dark-red active tongue material may improve VSR visibility",
        "title": "VOCASets Dark-Red Tongue Color Experiment",
    },
}
EXPERIMENT_NAME = COLOR_VARIANTS["dark_red"]["experiment"]
CLIP_SPEAKER = "FaceTalk_170725_00137_TA"
CLIP_SENTENCE = "sentence01"
CLIP_ID = f"{CLIP_SPEAKER}_{CLIP_SENTENCE}"

DEFAULT_OUTPUT_ROOT = (
    Path("/research/milsrg1/user_workspace/ht467/smirk_task/outputs")
    / "vocasets_color_search"
)
DEFAULT_LINK_ROOT = PROJECT_ROOT / "tests" / "vocaset_outputs" / "color_search"
DEFAULT_PASSIVE_VIDEO = (
    PROJECT_ROOT
    / "tests"
    / "vocaset_outputs"
    / "passive"
    / f"vocaset_{CLIP_ID}_passive_tongue_with_audio.mp4"
)


@dataclass(frozen=True)
class ColorExperimentParams:
    std_scalar: float = 0.27
    rotation_deg: float = 5.0
    thickness: float = 1.2
    shift_z: float = 0.0
    shift_y: float = 0.0


@dataclass(frozen=True)
class ColorExperimentPaths:
    out_dir: Path
    link_dir: Path
    motion_path: Path
    raw_video: Path
    audio_video: Path
    audio_link: Path
    metric_active: Path
    metric_passive: Path
    report_csv: Path
    report_md: Path


def color_experiment_paths(
    output_root: Path,
    link_root: Path,
    color_name: str = "dark_red",
    params: ColorExperimentParams = ColorExperimentParams(),
    speaker: str = CLIP_SPEAKER,
    sentence: str = CLIP_SENTENCE,
) -> ColorExperimentPaths:
    experiment = experiment_name_for(color_name, params)
    slug = COLOR_VARIANTS[color_name]["slug"]
    if params.thickness != 1.2:
        slug = f"{slug}_std0p27_rot5_th{str(f'{params.thickness:.2f}').replace('.', 'p')}"
    else:
        slug = f"{slug}_std0p27_rot5"
    clip_id = f"{speaker}_{sentence}"
    out_dir = output_root / experiment / speaker / sentence
    link_dir = link_root / experiment
    stem = f"{clip_id}_{slug}_active_tongue"
    return ColorExperimentPaths(
        out_dir=out_dir,
        link_dir=link_dir,
        motion_path=out_dir / "tongue_motion" / "tongue_motion.npy",
        raw_video=out_dir / f"{stem}.mp4",
        audio_video=out_dir / f"{stem}_with_audio.mp4",
        audio_link=link_dir / f"vocaset_{stem}_with_audio.mp4",
        metric_active=out_dir / f"{clip_id}_{slug}_active_metrics.json",
        metric_passive=out_dir / f"{clip_id}_passive_metrics.json",
        report_csv=out_dir / f"{clip_id}_{slug}_vsr_comparison.csv",
        report_md=out_dir / f"{clip_id}_{slug}_vsr_report.md",
    )


def tongue_color_for(color_name: str) -> tuple[float, float, float, float]:
    return COLOR_VARIANTS[color_name]["color"]


def experiment_name_for(
    color_name: str,
    params: ColorExperimentParams = ColorExperimentParams(),
) -> str:
    base = COLOR_VARIANTS[color_name]["experiment"]
    if params.thickness == 1.2:
        return base
    return f"{base}_th{str(f'{params.thickness:.2f}').replace('.', 'p')}"


def replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    tmp_link = link.with_name(f".{link.name}.{os.getpid()}.tmp")
    tmp_link.unlink(missing_ok=True)
    tmp_link.symlink_to(target)
    os.replace(tmp_link, link)


def sentence_ground_truth(transcript_path: Path, sentence: str) -> str:
    index = int(sentence.replace("sentence", ""))
    lines = [
        line.strip()
        for line in transcript_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]
    return lines[index - 1]


def render_color_active(args: argparse.Namespace, paths: ColorExperimentPaths) -> None:
    import numpy as np

    from tongue_scripts.inversion.invert import infer_ema, load_model
    from tongue_scripts.pipelines.grid_search_vocaset_active_tongue import (
        resample_ema_to_face_frames,
    )
    from tongue_scripts.rendering.render_dual_tongue_comparison import (
        ANCHOR_INDICES,
        BONE_INDICES,
        STD_PATH,
        TONGUE_CONFIG,
        TONGUE_SLICE,
        apply_jawopen_offset_correction,
        merge_audio,
        render_video_with_dynamic_tongue,
    )
    from tongue_scripts.tongue_animation.face_model_io_trimesh import load_face_model_trimesh
    from tongue_scripts.tongue_animation.generate_tongue_animation import (
        FaceKitTongueRig,
        load_blendshape_json_sequence,
        load_ema_motion,
    )

    if paths.audio_video.is_file() and not args.force:
        replace_symlink(paths.audio_link, paths.audio_video)
        return

    paths.out_dir.mkdir(parents=True, exist_ok=True)
    paths.motion_path.parent.mkdir(parents=True, exist_ok=True)

    wav_path = Path(args.wav_path)
    if not paths.motion_path.is_file() or args.force_motion:
        model = load_model(Path(args.checkpoint))
        ema = infer_ema(wav_path, model=model, max_seconds=args.max_seconds)
        np.save(str(paths.motion_path), ema)

    face_model = load_face_model_trimesh(PROJECT_ROOT / "FaceXModel")
    config = dict(TONGUE_CONFIG)
    config.update(
        {
            "std_scalar": args.std_scalar,
            "shift_z": 0.0,
            "rotation_deg": args.rotation_deg,
            "thickness": args.thickness,
            "shift_y": 0.0,
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
    face_seq = load_blendshape_json_sequence(
        Path(args.json_path),
        face_model,
        source_fps=args.source_fps,
        target_fps=args.fps,
    )
    face_seq = apply_jawopen_offset_correction(face_seq, face_model)
    ema_seq = load_ema_motion(paths.motion_path, STD_PATH, tongue_rig.anchors, config["std_scalar"])
    ema_seq = resample_ema_to_face_frames(ema_seq, len(face_seq))
    render_video_with_dynamic_tongue(
        face_model,
        face_seq,
        tongue_rig,
        ema_seq,
        str(paths.raw_video),
        fps=args.fps,
        tongue_color=tongue_color_for(args.color_name),
    )
    merge_audio(str(paths.raw_video), str(wav_path), str(paths.audio_video))
    replace_symlink(paths.audio_link, paths.audio_video)


def evaluate_video(
    args: argparse.Namespace,
    video: Path,
    ground_truth: str,
    metric_path: Path,
    condition: str,
) -> dict:
    if metric_path.is_file() and not args.force_eval:
        return json.loads(metric_path.read_text(encoding="utf-8"))

    from tongue_scripts.evaluation.evaluate_vsr_ver import evaluate_videos

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
        f.write(ground_truth + "\n")
        gt_path = Path(f.name)
    try:
        eval_args = argparse.Namespace(
            videos=[str(video)],
            ground_truth=str(gt_path),
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
            textgrid_path=None,
            seg_target_seconds=8.0,
            seg_min_seconds=4.0,
            seg_max_seconds=12.0,
            seg_min_silence=0.0,
            config_filename=args.config_filename,
            vowel_mode=args.vowel_mode,
            detector=args.detector,
            python_bin=args.python_bin,
            experiment_name=f"{CLIP_ID}_{condition}_{experiment_name_for(args.color_name, args.params)}",
            report_path=str(
                Path(args.output_root)
                / experiment_name_for(args.color_name, args.params)
                / "adfa_color_search_report.md"
            ),
            report_mode="append",
            dataset_id=CLIP_ID,
            speaker_id=CLIP_SPEAKER,
            hypothesis=COLOR_VARIANTS[args.color_name]["hypothesis"],
        )
        _, rows = evaluate_videos(eval_args)
        payload = dict(rows[0])
        payload.update({"condition": condition, "ground_truth": ground_truth})
        metric_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload
    finally:
        gt_path.unlink(missing_ok=True)


def write_report(paths: ColorExperimentPaths, active: dict, passive: dict) -> None:
    paths.report_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "ver",
        "wer_norm",
        "wer_raw",
        "composite_index",
        "hypothesis",
        "ground_truth",
        "video",
    ]
    with paths.report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: active.get(key, "") for key in fieldnames})
        writer.writerow({key: passive.get(key, "") for key in fieldnames})

    lines = [
        f"# {COLOR_VARIANTS[active['condition'].replace('_active', '')]['title']}",
        "",
        f"- active VER: {active['ver']:.4f}",
        f"- passive VER: {passive['ver']:.4f}",
        f"- delta active-passive VER: {active['ver'] - passive['ver']:.4f}",
        f"- active hypothesis: {active['hypothesis']}",
        f"- passive hypothesis: {passive['hypothesis']}",
        f"- ground truth: {active['ground_truth']}",
        f"- active video: `{paths.audio_video}`",
        f"- passive video: `{DEFAULT_PASSIVE_VIDEO}`",
    ]
    paths.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    replace_symlink(paths.link_dir / paths.report_csv.name, paths.report_csv)
    replace_symlink(paths.link_dir / paths.report_md.name, paths.report_md)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color-name", choices=sorted(COLOR_VARIANTS), default="light_red")
    parser.add_argument("--std-scalar", type=float, default=0.27)
    parser.add_argument("--rotation-deg", type=float, default=5.0)
    parser.add_argument("--thickness", type=float, default=1.2)
    parser.add_argument(
        "--json-path",
        default=str(
            PROJECT_ROOT
            / "tests"
            / "vocasets"
            / "blendshape_json"
            / CLIP_SPEAKER
            / f"{CLIP_SENTENCE}.json"
        ),
    )
    parser.add_argument(
        "--wav-path",
        default=str(PROJECT_ROOT / "tests" / "vocasets" / "wav" / f"{CLIP_ID}.wav"),
    )
    parser.add_argument(
        "--transcript-path",
        default=str(PROJECT_ROOT / "tests" / "vocasets" / "sentencestext" / f"{CLIP_SPEAKER}.txt"),
    )
    parser.add_argument("--passive-video", default=str(DEFAULT_PASSIVE_VIDEO))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--link-root", default=str(DEFAULT_LINK_ROOT))
    parser.add_argument(
        "--checkpoint",
        default=str(
            PROJECT_ROOT
            / "tongue_scripts"
            / "inversion_checkpoints"
            / "lora_multispeaker_consistency_alpha_0.25_threshold_0_vctk_vvn_4tonguepoints"
        ),
    )
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--source-fps", type=float, default=60.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-motion", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument(
        "--python-bin",
        default="/research/milsrg1/user_workspace/ht467/tools/uv/adfa-vsr/bin/python",
    )
    parser.add_argument("--infer-mode", choices=["full", "segmented"], default="full")
    parser.add_argument("--config-filename", default="configs/LRS3_V_WER19.1.ini")
    parser.add_argument("--vowel-mode", choices=["grouped", "exact"], default="grouped")
    parser.add_argument("--detector", default="mediapipe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.params = ColorExperimentParams(
        std_scalar=args.std_scalar,
        rotation_deg=args.rotation_deg,
        thickness=args.thickness,
    )
    paths = color_experiment_paths(
        Path(args.output_root),
        Path(args.link_root),
        args.color_name,
        args.params,
    )
    render_color_active(args, paths)
    ground_truth = sentence_ground_truth(Path(args.transcript_path), CLIP_SENTENCE)
    active = evaluate_video(
        args,
        paths.audio_video,
        ground_truth,
        paths.metric_active,
        f"{args.color_name}_active",
    )
    passive = evaluate_video(
        args,
        Path(args.passive_video).resolve(),
        ground_truth,
        paths.metric_passive,
        "passive",
    )
    write_report(paths, active, passive)
    print(f"active_ver={active['ver']:.6f}")
    print(f"passive_ver={passive['ver']:.6f}")
    print(f"csv={paths.report_csv}")
    print(f"link_dir={paths.link_dir}")


if __name__ == "__main__":
    main()
