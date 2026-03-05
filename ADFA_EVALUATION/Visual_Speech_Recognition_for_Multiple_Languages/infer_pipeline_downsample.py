"""Run AVSR inference after forcing videos to match the model frame rate.

This variant mirrors infer_pipeline.py but always downsamples input videos to
25 fps (or a user-provided target) before running inference. Segmentation via
TextGrid is intentionally disabled so full-length clips are processed in a
single pass, and transcripts are written to transcript_new by default.
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from infer_pipeline import (
    InferencePipeline,
    clip_video_segment,
    enforce_max_duration,
    ensure_textgrid_file,
    extract_speaker_id,
    get_video_duration,
    maybe_clear_cuda_cache,
    parse_textgrid_words,
    word_intervals_to_segments,
)

MODEL_CONF = "data/LRS3_V_WER19.1/model.json"
MODEL_PATH = "data/LRS3_V_WER19.1/model.pth"
VIDEO_PATH = "data/26_reamey_0_112_112.mp4"
DEFAULT_OUTPUT_DIR = "transcript_new"
DEFAULT_TARGET_FPS = 25
TEXTGRID_CACHE_ROOT = "data/beat_textgrids"
MAX_INFERENCE_SECONDS = 60.0


def infer_entire_video(pipeline: InferencePipeline, video_path: str) -> str:
    """Run inference on the provided clip and flush CUDA cache afterwards."""
    try:
        return pipeline(video_path).strip()
    finally:
        maybe_clear_cuda_cache(pipeline.device)


def downsample_video(video_path: str, target_fps: int, working_dir: str) -> str:
    """Create a temporary clip encoded at the desired frame rate."""
    safe_name = f"{Path(video_path).stem}_{target_fps}fps.mp4"
    output_path = os.path.join(working_dir, safe_name)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"fps={target_fps}",
        "-r",
        str(target_fps),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-loglevel",
        "error",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - external tool
        raise RuntimeError(
            f"Failed to downsample {video_path} to {target_fps} fps"
        ) from exc
    return output_path


def build_segments(
    video_path: str,
    textgrid_root: Optional[str],
    silence_threshold: float,
    max_segment_seconds: float,
    auto_download_textgrid: bool,
) -> Tuple[List[Tuple[float, float]], float, Optional[str]]:
    speaker_id = extract_speaker_id(video_path)
    textgrid_path = ensure_textgrid_file(
        video_path, speaker_id, textgrid_root, auto_download_textgrid
    )
    segments: List[Tuple[float, float]] = []
    if textgrid_path:
        intervals = parse_textgrid_words(textgrid_path)
        segments = word_intervals_to_segments(
            intervals,
            silence_threshold=silence_threshold,
            max_segment_seconds=max_segment_seconds,
        )

    video_duration = get_video_duration(video_path)

    if segments:
        segments = enforce_max_duration(
            segments, min(MAX_INFERENCE_SECONDS, max_segment_seconds)
        )
    else:
        if not textgrid_path:
            print("TextGrid not found; falling back to duration-based segmentation.")
        segments = enforce_max_duration(
            [(0.0, video_duration)],
            min(MAX_INFERENCE_SECONDS, max_segment_seconds),
        )

    return segments, video_duration, textgrid_path


def infer_segments(
    pipeline: InferencePipeline,
    video_path: str,
    segments: List[Tuple[float, float]],
    work_dir: str,
) -> str:
    transcripts: List[str] = []
    segment_dir = os.path.join(work_dir, "segments")
    os.makedirs(segment_dir, exist_ok=True)

    for idx, (start, end) in enumerate(segments):
        if end <= start:
            continue
        segment_path = os.path.join(segment_dir, f"segment_{idx:03d}.mp4")
        clip_video_segment(video_path, segment_path, start, end)
        text = pipeline(segment_path).strip()
        transcripts.append(text)
        maybe_clear_cuda_cache(pipeline.device)

    return " ".join(t for t in transcripts if t).strip()


def transcribe_video(
    pipeline: InferencePipeline,
    video_path: str,
    output_path: str,
    target_fps: int,
    textgrid_root: Optional[str],
    auto_download_textgrid: bool,
    silence_threshold: float,
    max_segment_seconds: float,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Processing video: {video_path}")
    segments, video_duration, textgrid_path = build_segments(
        video_path,
        textgrid_root,
        silence_threshold,
        max_segment_seconds,
        auto_download_textgrid,
    )
    segment_lengths = [max(end - start, 0.0) for start, end in segments]
    if segment_lengths:
        length_info = ", ".join(f"{length:.2f}s" for length in segment_lengths)
        if textgrid_path:
            print(
                f"Segments derived from TextGrid ({len(segment_lengths)}): {length_info}"
            )
        else:
            print(
                f"Segments without TextGrid ({len(segment_lengths)}): {length_info}"
            )

        total = sum(segment_lengths)
        avg = total / len(segment_lengths)
        print(
            "Segment stats -> total: "
            f"{total:.2f}s, avg: {avg:.2f}s, min: {min(segment_lengths):.2f}s, max: {max(segment_lengths):.2f}s"
        )

    with tempfile.TemporaryDirectory(prefix="avsr_downsample_") as tmp_dir:
        processed_path = downsample_video(video_path, target_fps, tmp_dir)
        use_direct = (
            len(segments) == 1
            and segments[0][0] <= 1e-3
            and abs(segments[0][1] - video_duration) <= 1e-3
            and (segments[0][1] - segments[0][0]) <= MAX_INFERENCE_SECONDS + 1e-3
        )

        if use_direct:
            transcript = infer_entire_video(pipeline, processed_path)
        else:
            transcript = infer_segments(
                pipeline,
                processed_path,
                segments,
                tmp_dir,
            )
    with open(output_path, "w") as handle:
        handle.write(transcript)
    print(f"Saved transcript to {output_path}")


def process_directory(
    pipeline: InferencePipeline,
    source_dir: str,
    target_root: str,
    target_fps: int,
    textgrid_root: Optional[str],
    auto_download_textgrid: bool,
    silence_threshold: float,
    max_segment_seconds: float,
):
    source_dir = os.path.abspath(source_dir)
    target_root = os.path.abspath(target_root)
    print("")
    print(
        f"Processing directory tree {source_dir}\nSaving transcripts to {target_root}"
    )

    if not os.path.isdir(source_dir):
        raise ValueError(
            f"source_dir {source_dir} does not exist or is not a directory"
        )

    for root, _, files in os.walk(source_dir):
        video_files = [f for f in files if f.lower().endswith(".mp4")]
        if not video_files:
            continue

        rel_dir = os.path.relpath(root, source_dir)
        for video_file in sorted(video_files):
            video_path = os.path.join(root, video_file)
            rel_output_dir = os.path.join(target_root, rel_dir)
            transcript_name = f"{os.path.splitext(video_file)[0]}.txt"
            output_path = os.path.join(rel_output_dir, transcript_name)

            if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                print(f"Skipping existing transcript: {output_path}")
                continue

            print("")
            try:
                transcribe_video(
                    pipeline,
                    video_path,
                    output_path,
                    target_fps=target_fps,
                    textgrid_root=textgrid_root,
                    auto_download_textgrid=auto_download_textgrid,
                    silence_threshold=silence_threshold,
                    max_segment_seconds=max_segment_seconds,
                )
            except Exception as exc:  # pragma: no cover - runtime safeguard
                print(f"Skipping {video_path} due to error: {exc}")
                maybe_clear_cuda_cache(pipeline.device)
                continue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AVSR inference after enforcing a fixed frame rate"
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=VIDEO_PATH,
        help="Path to a single video for quick testing",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Root directory that contains speaker folders with videos",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where transcripts will be stored",
    )
    parser.add_argument(
        "--model-conf",
        type=str,
        default=MODEL_CONF,
        help="Path to model configuration JSON",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=DEFAULT_TARGET_FPS,
        help="Frame rate to enforce before inference (default: 25)",
    )
    parser.add_argument(
        "--textgrid-root",
        type=str,
        default=TEXTGRID_CACHE_ROOT,
        help="Directory used to cache/download TextGrid files from BEAT.",
    )
    parser.add_argument(
        "--auto-download-textgrid",
        dest="auto_download_textgrid",
        action="store_true",
        help="Automatically download missing TextGrid files (default).",
    )
    parser.add_argument(
        "--no-auto-download-textgrid",
        dest="auto_download_textgrid",
        action="store_false",
        help="Disable automatic TextGrid downloads; fall back to single-pass inference if missing.",
    )
    parser.set_defaults(auto_download_textgrid=True)
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.5,
        help="Minimum silence duration (s) that triggers a new segment when TextGrids are available.",
    )
    parser.add_argument(
        "--max-segment-seconds",
        type=float,
        default=15.0,
        help="Maximum allowed segment duration before enforcing a split.",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="mediapipe",
        help="Face detector to use for landmark extraction",
    )
    parser.add_argument(
        "--face-track",
        dest="face_track",
        action="store_true",
        help="Enable face tracking (landmark detection) for video modalities",
    )
    parser.add_argument(
        "--no-face-track",
        dest="face_track",
        action="store_false",
        help="Disable face tracking if precomputed landmarks are supplied",
    )
    parser.set_defaults(face_track=True)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device identifier, e.g., cuda:0 or cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device:
        device = args.device
        print(f"Using provided device: {device}")
    elif torch.cuda.is_available():
        device = "cuda:0"
        print(f"CUDA is available. Using device: {device}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA is not available. Using CPU.")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    modality = "video"
    model_conf = (
        args.model_conf
        if os.path.isabs(args.model_conf)
        else os.path.join(base_dir, args.model_conf)
    )
    model_path = (
        args.model_path
        if os.path.isabs(args.model_path)
        else os.path.join(base_dir, args.model_path)
    )

    pipeline = InferencePipeline(
        modality,
        model_path,
        model_conf,
        detector=args.detector,
        face_track=args.face_track,
        device=device,
    )

    textgrid_root = (
        args.textgrid_root
        if os.path.isabs(args.textgrid_root)
        else os.path.join(base_dir, args.textgrid_root)
    )
    output_root = (
        args.output_dir
        if os.path.isabs(args.output_dir)
        else os.path.join(base_dir, args.output_dir)
    )
    os.makedirs(output_root, exist_ok=True)

    if args.video_dir:
        source_dir = (
            args.video_dir
            if os.path.isabs(args.video_dir)
            else os.path.join(base_dir, args.video_dir)
        )
        process_directory(
            pipeline,
            source_dir,
            output_root,
            target_fps=args.target_fps,
            textgrid_root=textgrid_root,
            auto_download_textgrid=args.auto_download_textgrid,
            silence_threshold=args.silence_threshold,
            max_segment_seconds=args.max_segment_seconds,
        )
    else:
        video_path = (
            args.video_path
            if os.path.isabs(args.video_path)
            else os.path.join(base_dir, args.video_path)
        )

        if not os.path.exists(video_path):
            fallback = os.path.join(base_dir, "data/clip.mp4")
            print(f"Provided video not found. Falling back to {fallback}")
            video_path = fallback

        transcript_name = f"{os.path.splitext(os.path.basename(video_path))[0]}.txt"
        output_path = os.path.join(output_root, transcript_name)
        transcribe_video(
            pipeline,
            video_path,
            output_path,
            target_fps=args.target_fps,
            textgrid_root=textgrid_root,
            auto_download_textgrid=args.auto_download_textgrid,
            silence_threshold=args.silence_threshold,
            max_segment_seconds=args.max_segment_seconds,
        )


if __name__ == "__main__":
    main()
