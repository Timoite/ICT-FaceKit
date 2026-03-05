"""Unified AVSR inference pipeline.

Key behavior:
- Always enforce 25 FPS input (default, configurable).
- Segment long clips by TextGrid word/silence boundaries (no word is cut in half).
- Target ~8 second segments for stable VSR inference memory/quality tradeoff.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, cast

import torch

from pipelines.data.data_module import AVSRDataLoader
from pipelines.detectors.mediapipe.detector import LandmarksDetector
from pipelines.model import AVSR

MODEL_CONF = "data/LRS3_V_WER19.1/model.json"
MODEL_PATH = "data/LRS3_V_WER19.1/model.pth"
VIDEO_PATH = "data/26_reamey_0_112_112.mp4"
DEFAULT_OUTPUT_DIR = "transcripts"
DEFAULT_TARGET_FPS = 25

# Prefer local BEAT cache shipped at repo root, then project-local fallback.
DEFAULT_TEXTGRID_ROOTS = [
    Path(__file__).resolve().parents[2] / "data" / "beat_cache" / "beat_english_v0.2.1" / "beat_english_v0.2.1",
    Path(__file__).resolve().parent / "data" / "beat_cache" / "beat_english_v0.2.1" / "beat_english_v0.2.1",
    Path(__file__).resolve().parent / "data" / "beat_textgrids",
]

# Segment policy tuned from sample 1_wayne_0_75_75.TextGrid:
# silence median ~0.09s, p75 ~0.32s, p90 ~0.66s.
# General criterion used below:
#   cut candidates = long silences (>= max(0.18, min(0.45, p75)))
# with short-pause fallback when needed to stay close to ~8s segments.
DEFAULT_TARGET_SEGMENT_SECONDS = 8.0
DEFAULT_MIN_SEGMENT_SECONDS = 4.0
DEFAULT_MAX_SEGMENT_SECONDS = 12.0
DEFAULT_MIN_SILENCE_SECONDS = 0.0  # <= 0 means auto-derive from TextGrid silence stats
SHORT_PAUSE_FALLBACK_SECONDS = 0.08


@dataclass
class WordInterval:
    start: float
    end: float
    text: str


@dataclass
class SilenceInterval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def midpoint(self) -> float:
        return self.start + 0.5 * self.duration


class InferencePipeline(torch.nn.Module):
    def __init__(
        self,
        modality: str,
        model_path: str,
        model_conf: str,
        detector: str = "mediapipe",
        face_track: bool = False,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.device = device
        self.modality = modality
        self.dataloader = AVSRDataLoader(modality, detector=detector)
        self.model = AVSR(
            modality,
            model_path,
            model_conf,
            rnnlm=None,
            rnnlm_conf=None,
            penalty=0.0,
            ctc_weight=0.1,
            lm_weight=0.0,
            beam_size=40,
            device=device,
        )
        if face_track and self.modality in ["video", "audiovisual"]:
            self.landmarks_detector = LandmarksDetector()
        else:
            self.landmarks_detector = None

    def process_landmarks(self, data_filename: str, landmarks_filename: Optional[str]):
        if self.modality == "audio":
            return "data/clip.mp4"
        if self.modality in ["video", "audiovisual"]:
            if self.landmarks_detector is None:
                raise RuntimeError(
                    "Landmarks detector requested but not initialized. Enable face tracking."
                )
            return self.landmarks_detector(data_filename)
        return landmarks_filename

    def forward(self, data_filename: str, landmarks_filename: Optional[str] = None):
        assert os.path.isfile(data_filename), f"data_filename does not exist: {data_filename}"
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        return self.model.infer(data)

    def extract_features(
        self,
        data_filename: str,
        landmarks_filename: Optional[str] = None,
        extract_resnet_feats: bool = False,
    ):
        assert os.path.isfile(data_filename), f"data_filename does not exist: {data_filename}"
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        if data is None:
            raise ValueError(f"Dataloader returned no data for {data_filename}")

        model_core = cast(Any, self.model.model)
        with torch.no_grad():
            if isinstance(data, tuple):
                video_tensor, audio_tensor = data
                return model_core.encode(
                    video_tensor.to(self.device),
                    audio_tensor.to(self.device),
                    extract_resnet_feats=extract_resnet_feats,
                )
            return model_core.encode(
                data.to(self.device),
                extract_resnet_feats=extract_resnet_feats,
            )


def maybe_clear_cuda_cache(device: Optional[str]) -> None:
    if device and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_video_duration(filename: str) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            filename,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        text=True,
    )
    return float(result.stdout.strip())


def parse_textgrid_words(textgrid_path: str, tier_name: str = "words") -> List[WordInterval]:
    intervals: List[WordInterval] = []
    in_tier = False
    current: dict = {}

    with open(textgrid_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("item ["):
                in_tier = False
                continue
            if line.startswith('name = "'):
                tier = line.split("=", 1)[1].strip().strip('"')
                in_tier = tier == tier_name
                continue
            if not in_tier:
                continue
            if line.startswith("intervals ["):
                current = {}
                continue
            if line.startswith("xmin ="):
                current["start"] = float(line.split("=", 1)[1])
                continue
            if line.startswith("xmax ="):
                current["end"] = float(line.split("=", 1)[1])
                continue
            if line.startswith("text ="):
                text_value = line.split("=", 1)[1].strip()
                if text_value.startswith('"') and text_value.endswith('"'):
                    text_value = text_value[1:-1]
                current["text"] = text_value
                if {"start", "end", "text"} <= current.keys():
                    intervals.append(
                        WordInterval(
                            start=float(current["start"]),
                            end=float(current["end"]),
                            text=str(current["text"]),
                        )
                    )
    return intervals


def split_words_and_silences(
    intervals: Sequence[WordInterval],
) -> Tuple[List[WordInterval], List[SilenceInterval]]:
    words: List[WordInterval] = []
    silences: List[SilenceInterval] = []
    for itv in intervals:
        if itv.text.strip():
            words.append(itv)
        else:
            silences.append(SilenceInterval(start=itv.start, end=itv.end))
    return words, silences


def summarize_silences(silences: Sequence[SilenceInterval]) -> str:
    if not silences:
        return "no silence intervals"
    durations = sorted(s.duration for s in silences)

    def q(v: float) -> float:
        idx = round((len(durations) - 1) * v)
        return durations[idx]

    return (
        f"count={len(durations)}, min={min(durations):.3f}s, "
        f"p50={q(0.5):.3f}s, p75={q(0.75):.3f}s, "
        f"p90={q(0.9):.3f}s, max={max(durations):.3f}s"
    )


def derive_min_silence_for_cut(
    silences: Sequence[SilenceInterval],
    manual_threshold: float,
) -> float:
    """Return silence threshold for primary cut candidates.

    If `manual_threshold` is > 0, use it directly.
    Otherwise derive threshold from TextGrid silence distribution.
    """
    if manual_threshold > 0:
        return manual_threshold

    durations = sorted(s.duration for s in silences if s.duration > 0.0)
    if not durations:
        return 0.30

    p75_idx = round((len(durations) - 1) * 0.75)
    p75 = durations[p75_idx]
    return max(0.18, min(0.45, p75))


def choose_cut_time(
    segment_start: float,
    candidate_silences: Sequence[SilenceInterval],
    target_seconds: float,
    min_seconds: float,
    max_seconds: float,
) -> Optional[float]:
    best_time: Optional[float] = None
    best_score: Optional[float] = None

    for silence in candidate_silences:
        cut = silence.midpoint
        seg_len = cut - segment_start
        if seg_len < min_seconds or seg_len > max_seconds:
            continue
        score = abs(seg_len - target_seconds)
        if best_score is None or score < best_score:
            best_score = score
            best_time = cut

    return best_time


def build_segments_from_textgrid(
    words: Sequence[WordInterval],
    silences: Sequence[SilenceInterval],
    target_seconds: float,
    min_seconds: float,
    max_seconds: float,
    min_silence_for_cut: float,
) -> List[Tuple[float, float]]:
    if not words:
        return []

    primary_silences = [s for s in silences if s.duration >= min_silence_for_cut]
    fallback_silences = [s for s in silences if s.duration >= SHORT_PAUSE_FALLBACK_SECONDS]
    segments: List[Tuple[float, float]] = []

    seg_start = words[0].start
    i = 0
    while i < len(words):
        current_end = words[i].end
        seg_len = current_end - seg_start

        if seg_len >= target_seconds:
            candidate_pool = [
                s
                for s in primary_silences
                if s.start >= seg_start and s.end <= current_end + 1e-6
            ]
            cut = choose_cut_time(
                seg_start,
                candidate_pool,
                target_seconds,
                min_seconds,
                max_seconds,
            )

            # If no long-pause cut exists, allow shorter pauses to stay near target length.
            if cut is None:
                fallback_pool = [
                    s
                    for s in fallback_silences
                    if s.start >= seg_start and s.end <= current_end + 1e-6
                ]
                cut = choose_cut_time(
                    seg_start,
                    fallback_pool,
                    target_seconds,
                    min_seconds,
                    max_seconds,
                )

            # If no suitable silence exists, cut at current word boundary to avoid runaway length.
            if cut is None and seg_len >= max_seconds:
                cut = current_end

            if cut is not None and (cut - seg_start) >= min_seconds:
                segments.append((seg_start, cut))
                seg_start = cut

        i += 1

    final_end = words[-1].end
    if final_end > seg_start + 0.20:
        if final_end - seg_start < min_seconds and segments:
            prev_start, _ = segments[-1]
            segments[-1] = (prev_start, final_end)
        else:
            segments.append((seg_start, final_end))

    # Safety cleanup.
    normalized: List[Tuple[float, float]] = []
    for start, end in segments:
        if end - start > 0.15:
            normalized.append((round(start, 3), round(end, 3)))
    return normalized


def clip_video_segment(
    video_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    target_fps: int,
) -> None:
    duration = max(end_time - start_time, 0.1)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-i",
        video_path,
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-r",
        str(target_fps),
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
    subprocess.run(cmd, check=True)


def enforce_fps(video_path: str, target_fps: int, work_dir: str) -> str:
    output_path = os.path.join(work_dir, f"{Path(video_path).stem}_{target_fps}fps.mp4")
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
    subprocess.run(cmd, check=True)
    return output_path


def infer_entire_video(pipeline: InferencePipeline, video_path: str) -> str:
    try:
        return pipeline(video_path).strip()
    finally:
        maybe_clear_cuda_cache(pipeline.device)


def infer_segments(
    pipeline: InferencePipeline,
    video_path: str,
    segments: Sequence[Tuple[float, float]],
    work_dir: str,
    target_fps: int,
) -> str:
    out: List[str] = []
    seg_dir = os.path.join(work_dir, "segments")
    os.makedirs(seg_dir, exist_ok=True)

    for idx, (start, end) in enumerate(segments):
        seg_path = os.path.join(seg_dir, f"segment_{idx:03d}.mp4")
        clip_video_segment(video_path, seg_path, start, end, target_fps=target_fps)
        out.append(pipeline(seg_path).strip())
        maybe_clear_cuda_cache(pipeline.device)

    return " ".join(t for t in out if t).strip()


def extract_speaker_id(video_path: str) -> Optional[str]:
    stem = Path(video_path).stem
    token = stem.split("_", 1)[0]
    return token if token.isdigit() else None


def resolve_textgrid_root(textgrid_root: Optional[str]) -> Optional[Path]:
    if textgrid_root:
        p = Path(textgrid_root)
        return p if p.exists() else None
    for candidate in DEFAULT_TEXTGRID_ROOTS:
        if candidate.exists():
            return candidate
    return None


def ensure_textgrid_file(
    video_path: str,
    textgrid_root: Optional[str],
    explicit_textgrid_path: Optional[str],
) -> Optional[str]:
    if explicit_textgrid_path:
        p = Path(explicit_textgrid_path)
        return str(p) if p.is_file() else None

    root = resolve_textgrid_root(textgrid_root)
    if root is None:
        return None

    stem = Path(video_path).stem
    speaker_id = extract_speaker_id(video_path)

    candidates: List[Path] = [root / f"{stem}.TextGrid"]
    if speaker_id is not None:
        candidates.append(root / speaker_id / f"{stem}.TextGrid")

    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def build_segments(
    video_path: str,
    textgrid_root: Optional[str],
    explicit_textgrid_path: Optional[str],
    target_segment_seconds: float,
    min_segment_seconds: float,
    max_segment_seconds: float,
    min_silence_seconds: float,
    print_silence_stats: bool,
) -> Tuple[List[Tuple[float, float]], float, Optional[str]]:
    duration = get_video_duration(video_path)
    textgrid_path = ensure_textgrid_file(video_path, textgrid_root, explicit_textgrid_path)

    if textgrid_path:
        intervals = parse_textgrid_words(textgrid_path)
        words, silences = split_words_and_silences(intervals)
        min_silence_for_cut = derive_min_silence_for_cut(silences, manual_threshold=min_silence_seconds)
        if print_silence_stats:
            print(f"Silence stats for {Path(video_path).name}: {summarize_silences(silences)}")
            print(f"Using min_silence_for_cut={min_silence_for_cut:.3f}s")
        segments = build_segments_from_textgrid(
            words,
            silences,
            target_seconds=target_segment_seconds,
            min_seconds=min_segment_seconds,
            max_seconds=max_segment_seconds,
            min_silence_for_cut=min_silence_for_cut,
        )
        if segments:
            return segments, duration, textgrid_path

    # Fallback: keep full clip as one segment.
    return [(0.0, duration)], duration, textgrid_path


def transcribe_video(
    pipeline: InferencePipeline,
    video_path: str,
    output_path: str,
    target_fps: int,
    textgrid_root: Optional[str],
    explicit_textgrid_path: Optional[str],
    target_segment_seconds: float,
    min_segment_seconds: float,
    max_segment_seconds: float,
    min_silence_seconds: float,
    print_silence_stats: bool,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    segments, duration, textgrid_path = build_segments(
        video_path=video_path,
        textgrid_root=textgrid_root,
        explicit_textgrid_path=explicit_textgrid_path,
        target_segment_seconds=target_segment_seconds,
        min_segment_seconds=min_segment_seconds,
        max_segment_seconds=max_segment_seconds,
        min_silence_seconds=min_silence_seconds,
        print_silence_stats=print_silence_stats,
    )

    if textgrid_path:
        print(f"TextGrid: {textgrid_path}")
    else:
        print("TextGrid not found; using full-duration inference.")

    seg_info = ", ".join(f"[{s:.2f},{e:.2f}]" for s, e in segments)
    print(f"Segments ({len(segments)}) -> {seg_info}")

    with tempfile.TemporaryDirectory(prefix="avsr_unified_") as tmp_dir:
        video_25fps = enforce_fps(video_path, target_fps=target_fps, work_dir=tmp_dir)

        use_direct = (
            len(segments) == 1
            and segments[0][0] <= 1e-3
            and abs(segments[0][1] - duration) <= 1e-3
        )

        if use_direct:
            transcript = infer_entire_video(pipeline, video_25fps)
        else:
            transcript = infer_segments(pipeline, video_25fps, segments, tmp_dir, target_fps=target_fps)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(transcript)
    print(f"Saved transcript: {output_path}")


def process_directory(
    pipeline: InferencePipeline,
    source_dir: str,
    target_root: str,
    target_fps: int,
    textgrid_root: Optional[str],
    target_segment_seconds: float,
    min_segment_seconds: float,
    max_segment_seconds: float,
    min_silence_seconds: float,
    print_silence_stats: bool,
) -> None:
    source_dir = os.path.abspath(source_dir)
    target_root = os.path.abspath(target_root)
    if not os.path.isdir(source_dir):
        raise ValueError(f"source_dir does not exist: {source_dir}")

    print(f"Processing tree: {source_dir}")
    print(f"Output root: {target_root}")

    for root, _, files in os.walk(source_dir):
        video_files = sorted(f for f in files if f.lower().endswith(".mp4"))
        if not video_files:
            continue
        rel = os.path.relpath(root, source_dir)

        for video_file in video_files:
            video_path = os.path.join(root, video_file)
            out_dir = os.path.join(target_root, rel)
            out_path = os.path.join(out_dir, f"{Path(video_file).stem}.txt")

            if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
                print(f"Skipping existing transcript: {out_path}")
                continue

            print(f"\nProcessing video: {video_path}")
            try:
                transcribe_video(
                    pipeline=pipeline,
                    video_path=video_path,
                    output_path=out_path,
                    target_fps=target_fps,
                    textgrid_root=textgrid_root,
                    explicit_textgrid_path=None,
                    target_segment_seconds=target_segment_seconds,
                    min_segment_seconds=min_segment_seconds,
                    max_segment_seconds=max_segment_seconds,
                    min_silence_seconds=min_silence_seconds,
                    print_silence_stats=print_silence_stats,
                )
            except Exception as exc:
                print(f"Skipping {video_path} due to error: {exc}")
                maybe_clear_cuda_cache(pipeline.device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified VSR inference with 25fps enforcement and TextGrid silence segmentation"
    )
    parser.add_argument("--video-path", type=str, default=VIDEO_PATH, help="Single video path")
    parser.add_argument("--video-dir", type=str, default=None, help="Root directory of videos")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Transcript output root")
    parser.add_argument("--model-conf", type=str, default=MODEL_CONF, help="Model config JSON path")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Model checkpoint path")
    parser.add_argument("--target-fps", type=int, default=DEFAULT_TARGET_FPS, help="Input fps for inference")

    parser.add_argument(
        "--textgrid-root",
        type=str,
        default=None,
        help="TextGrid root containing either <speaker>/<clip>.TextGrid or <clip>.TextGrid",
    )
    parser.add_argument(
        "--textgrid-path",
        type=str,
        default=None,
        help="Explicit TextGrid path for single-video mode",
    )

    parser.add_argument(
        "--target-segment-seconds",
        type=float,
        default=DEFAULT_TARGET_SEGMENT_SECONDS,
        help="Target segment length (seconds), default 8.0",
    )
    parser.add_argument(
        "--min-segment-seconds",
        type=float,
        default=DEFAULT_MIN_SEGMENT_SECONDS,
        help="Minimum segment length (seconds)",
    )
    parser.add_argument(
        "--max-segment-seconds",
        type=float,
        default=DEFAULT_MAX_SEGMENT_SECONDS,
        help="Hard maximum segment length (seconds)",
    )
    parser.add_argument(
        "--min-silence-seconds",
        type=float,
        default=DEFAULT_MIN_SILENCE_SECONDS,
        help="Minimum empty-interval duration in words tier; <=0 auto-derives from TextGrid p75",
    )
    parser.add_argument(
        "--print-silence-stats",
        action="store_true",
        help="Print silence duration summary from TextGrid (analysis/debug)",
    )

    parser.add_argument("--detector", type=str, default="mediapipe", help="Landmark detector")
    parser.add_argument("--face-track", dest="face_track", action="store_true")
    parser.add_argument("--no-face-track", dest="face_track", action="store_false")
    parser.set_defaults(face_track=True)
    parser.add_argument("--device", type=str, default=None, help="cuda:0 or cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device:
        device = args.device
        print(f"Using device: {device}")
    elif torch.cuda.is_available():
        device = "cuda:0"
        print(f"CUDA available. Using {device}: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA unavailable. Using CPU")

    base_dir = Path(__file__).resolve().parent

    model_conf = Path(args.model_conf)
    if not model_conf.is_absolute():
        model_conf = base_dir / model_conf

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = base_dir / model_path

    pipeline = InferencePipeline(
        modality="video",
        model_path=str(model_path),
        model_conf=str(model_conf),
        detector=args.detector,
        face_track=args.face_track,
        device=device,
    )

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = base_dir / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    textgrid_root = args.textgrid_root
    if textgrid_root and not os.path.isabs(textgrid_root):
        textgrid_root = str(base_dir / textgrid_root)

    if args.video_dir:
        source_dir = Path(args.video_dir)
        if not source_dir.is_absolute():
            source_dir = base_dir / source_dir

        process_directory(
            pipeline=pipeline,
            source_dir=str(source_dir),
            target_root=str(output_root),
            target_fps=args.target_fps,
            textgrid_root=textgrid_root,
            target_segment_seconds=args.target_segment_seconds,
            min_segment_seconds=args.min_segment_seconds,
            max_segment_seconds=args.max_segment_seconds,
            min_silence_seconds=args.min_silence_seconds,
            print_silence_stats=args.print_silence_stats,
        )
        return

    video_path = Path(args.video_path)
    if not video_path.is_absolute():
        video_path = base_dir / video_path
    if not video_path.exists():
        fallback = base_dir / "data" / "clip.mp4"
        print(f"Provided video not found. Falling back to {fallback}")
        video_path = fallback

    explicit_textgrid_path = args.textgrid_path
    if explicit_textgrid_path and not os.path.isabs(explicit_textgrid_path):
        explicit_textgrid_path = str(base_dir / explicit_textgrid_path)

    output_path = output_root / f"{video_path.stem}.txt"
    transcribe_video(
        pipeline=pipeline,
        video_path=str(video_path),
        output_path=str(output_path),
        target_fps=args.target_fps,
        textgrid_root=textgrid_root,
        explicit_textgrid_path=explicit_textgrid_path,
        target_segment_seconds=args.target_segment_seconds,
        min_segment_seconds=args.min_segment_seconds,
        max_segment_seconds=args.max_segment_seconds,
        min_silence_seconds=args.min_silence_seconds,
        print_silence_stats=args.print_silence_stats,
    )


if __name__ == "__main__":
    main()
