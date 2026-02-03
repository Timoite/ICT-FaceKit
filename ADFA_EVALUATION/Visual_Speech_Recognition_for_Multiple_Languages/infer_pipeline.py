"""Run AVSR inference on individual videos or batches with directory mirroring.

Now supports TextGrid-driven sentence segmentation to avoid overlapping clips.
"""
import argparse
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

import torch
from pipelines.model import AVSR
from pipelines.data.data_module import AVSRDataLoader
from pipelines.detectors.mediapipe.detector import LandmarksDetector

# Change this to your model paths. Make sure to download the model files first.
MODEL_CONF = "data/LRS3_V_WER19.1/model.json"
MODEL_PATH = "data/LRS3_V_WER19.1/model.pth"
VIDEO_PATH = "data/26_reamey_0_112_112.mp4"
TEXTGRID_CACHE_ROOT = "data/beat_textgrids"
TEXTGRID_REPO = "H-Liu1997/BEAT"
TEXTGRID_DATASET_PREFIX = "beat_english_v0.2.1/beat_english_v0.2.1"
MAX_INFERENCE_SECONDS = 60.0


@dataclass
class WordInterval:
    start: float
    end: float
    text: str

class InferencePipeline(torch.nn.Module):
    def __init__(self, modality, model_path, model_conf, detector="mediapipe", face_track=False, device="cuda:0"):
        super(InferencePipeline, self).__init__()
        self.device = device
        # modality configuration
        self.modality = modality
        self.dataloader = AVSRDataLoader(modality, detector=detector)
        self.model = AVSR(modality, model_path, model_conf, rnnlm=None, rnnlm_conf=None, penalty=0.0, ctc_weight=0.1, lm_weight=0.0, beam_size=40, device=device)
        if face_track and self.modality in ["video", "audiovisual"]:
            self.landmarks_detector = LandmarksDetector()
        else:
            self.landmarks_detector = None


    def process_landmarks(self, data_filename, landmarks_filename):
        if self.modality == "audio":
            return "data/clip.mp4"
        if self.modality in ["video", "audiovisual"]:
            if self.landmarks_detector is None:
                raise RuntimeError("Landmarks detector requested but not initialized. Enable face tracking.")
            landmarks = self.landmarks_detector(data_filename)
            return landmarks


    def forward(self, data_filename, landmarks_filename=None):
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        transcript = self.model.infer(data)
        return transcript

    def extract_features(self, data_filename, landmarks_filename=None, extract_resnet_feats=False):
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        if data is None:
            raise ValueError(f"Dataloader returned no data for {data_filename}")

        model_core = cast(Any, self.model.model)

        with torch.no_grad():
            if isinstance(data, tuple):
                video_tensor, audio_tensor = data
                enc_feats = model_core.encode(
                    video_tensor.to(self.device),
                    audio_tensor.to(self.device),
                    extract_resnet_feats=extract_resnet_feats,
                )
            else:
                enc_feats = model_core.encode(
                    data.to(self.device),
                    extract_resnet_feats=extract_resnet_feats,
                )
        return enc_feats

def maybe_clear_cuda_cache(device):
    if device and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def infer_entire_video(pipeline, video_path):
    try:
        return pipeline(video_path).strip()
    finally:
        maybe_clear_cuda_cache(pipeline.device)


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
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Unable to determine duration for {filename}")

def clip_video_segment(video_path: str, output_path: str, start_time: float, end_time: float) -> None:
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
        "-c:a",
        "aac",
        "-loglevel",
        "error",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def extract_speaker_id(video_path: str) -> Optional[str]:
    path = Path(video_path)
    for part in reversed(path.parts):
        if part.startswith("speaker_"):
            suffix = part.split("_", 1)[1]
            return suffix
    stem = path.stem
    prefix = stem.split("_", 1)[0]
    return prefix if prefix else None


def download_textgrids_for_speaker(speaker_id: str, cache_root: Path) -> Path:
    dest_dir = cache_root / speaker_id
    if dest_dir.is_dir() and any(dest_dir.glob("*.TextGrid")):
        return dest_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "huggingface_hub is required to download TextGrid files automatically."
        ) from exc

    temp_dir = Path(tempfile.mkdtemp(prefix="beat_dl_"))
    try:
        pattern = f"{TEXTGRID_DATASET_PREFIX}/{speaker_id}/*.TextGrid"
        downloaded_root = Path(
            snapshot_download(
                repo_id=TEXTGRID_REPO,
                repo_type="dataset",
                allow_patterns=pattern,
                local_dir=str(temp_dir),
                local_dir_use_symlinks=False,
            )
        )

        candidates = [
            downloaded_root / TEXTGRID_DATASET_PREFIX / speaker_id,
            downloaded_root / "beat_english_v0.2.1" / "beat_english_v0.2.1" / speaker_id,
            downloaded_root / speaker_id,
        ]
        source_dir = next((c for c in candidates if c.exists()), None)
        if source_dir is None:
            raise FileNotFoundError(
                f"Unable to locate TextGrid files for speaker {speaker_id} in downloaded snapshot."
            )

        dest_dir.mkdir(parents=True, exist_ok=True)
        for textgrid_file in source_dir.glob("*.TextGrid"):
            shutil.copy2(textgrid_file, dest_dir / textgrid_file.name)
        return dest_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def ensure_textgrid_file(
    video_path: str,
    speaker_id: Optional[str],
    cache_root: Optional[str],
    auto_download: bool,
) -> Optional[str]:
    if not speaker_id or not cache_root:
        return None

    cache_root_path = Path(cache_root)
    textgrid_name = f"{Path(video_path).stem}.TextGrid"
    candidate = cache_root_path / speaker_id / textgrid_name
    if candidate.is_file():
        return str(candidate)

    if not auto_download:
        return None

    try:
        download_textgrids_for_speaker(speaker_id, cache_root_path)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"Warning: Failed to download TextGrid for speaker {speaker_id}: {exc}")
        return None

    if candidate.is_file():
        return str(candidate)

    print(f"Warning: TextGrid {textgrid_name} not found for speaker {speaker_id}.")
    return None


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
                tier = line.split('=')[1].strip().strip('"')
                in_tier = tier == tier_name
                continue
            if not in_tier:
                continue
            if line.startswith("intervals ["):
                current = {}
                continue
            if line.startswith("xmin ="):
                current["start"] = float(line.split("=")[1])
                continue
            if line.startswith("xmax ="):
                current["end"] = float(line.split("=")[1])
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
                            text=current["text"],
                        )
                    )
                continue

    return intervals


def word_intervals_to_segments(
    intervals: List[WordInterval],
    silence_threshold: float,
    max_segment_seconds: float,
    min_segment_seconds: float = 3.0,
) -> List[Tuple[float, float]]:
    segments: List[Tuple[float, float]] = []
    current_start: Optional[float] = None
    last_end: Optional[float] = None

    for interval in intervals:
        if interval.text.strip():
            if current_start is None:
                current_start = interval.start
            last_end = interval.end
            continue

        if current_start is None or last_end is None:
            continue

        gap = interval.end - interval.start
        duration = last_end - current_start
        if gap >= silence_threshold or duration >= max_segment_seconds:
            if duration >= min_segment_seconds:
                segments.append((current_start, last_end))
            current_start = None
            last_end = None

    if current_start is not None and last_end is not None:
        duration = last_end - current_start
        if duration >= min_segment_seconds:
            segments.append((current_start, last_end))

    return segments


def enforce_max_duration(segments: List[Tuple[float, float]], max_seconds: float) -> List[Tuple[float, float]]:
    if max_seconds <= 0:
        return segments

    capped: List[Tuple[float, float]] = []
    for start, end in segments:
        if end <= start:
            continue
        stack = [(start, end)]
        while stack:
            s, e = stack.pop()
            length = e - s
            if length <= max_seconds + 1e-3:
                capped.append((s, e))
            else:
                mid = s + (length / 2.0)
                if mid - s < 1e-3 or e - mid < 1e-3:
                    capped.append((s, e))
                else:
                    stack.append((mid, e))
                    stack.append((s, mid))
    return sorted(capped, key=lambda x: x[0])


def infer_segments_from_textgrid(
    pipeline,
    video_path: str,
    segments: List[Tuple[float, float]],
) -> str:
    transcripts: List[str] = []
    with tempfile.TemporaryDirectory(prefix="avsr_sent_") as tmp_dir:
        for idx, (start_time, end_time) in enumerate(segments):
            segment_path = os.path.join(tmp_dir, f"segment_{idx:03d}.mp4")
            clip_video_segment(video_path, segment_path, start_time, end_time)
            text = pipeline(segment_path).strip()
            transcripts.append(text)
            maybe_clear_cuda_cache(pipeline.device)
    return " ".join(t for t in transcripts if t).strip()


def transcribe_video(
    pipeline,
    video_path,
    output_path,
    textgrid_root=None,
    auto_download_textgrid=True,
    silence_threshold=0.5,
    max_segment_seconds=15.0,
):
    dir_name = os.path.dirname(output_path) or "."
    os.makedirs(dir_name, exist_ok=True)
    print(f"Processing video: {video_path}")
    speaker_id = extract_speaker_id(video_path)
    textgrid_path = ensure_textgrid_file(video_path, speaker_id, textgrid_root, auto_download_textgrid)
    segments: List[Tuple[float, float]] = []

    if textgrid_path:
        intervals = parse_textgrid_words(textgrid_path)
        segments = word_intervals_to_segments(
            intervals,
            silence_threshold=silence_threshold,
            max_segment_seconds=max_segment_seconds,
        )

    video_duration: Optional[float] = None

    def get_cached_duration() -> float:
        nonlocal video_duration
        if video_duration is None:
            video_duration = get_video_duration(video_path)
        return video_duration

    if segments:
        segments = enforce_max_duration(segments, min(MAX_INFERENCE_SECONDS, max_segment_seconds))
    else:
        if not textgrid_path:
            print("TextGrid not found; falling back to single-pass inference.")
        segments = enforce_max_duration([(0.0, get_cached_duration())], MAX_INFERENCE_SECONDS)

    use_direct = (
        len(segments) == 1
        and segments[0][0] <= 1e-3
        and abs(segments[0][1] - get_cached_duration()) <= 1e-3
        and (segments[0][1] - segments[0][0]) <= MAX_INFERENCE_SECONDS + 1e-3
    )

    try:
        if use_direct:
            transcript = infer_entire_video(pipeline, video_path)
        else:
            transcript = infer_segments_from_textgrid(pipeline, video_path, segments)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise exc
        raise

    with open(output_path, "w") as f:
        f.write(transcript)
    print(f"Saved transcript to {output_path}")


def process_directory(
    pipeline,
    source_dir,
    target_root,
    textgrid_root=None,
    auto_download_textgrid=True,
    silence_threshold=0.5,
    max_segment_seconds=15.0,
):
    source_dir = os.path.abspath(source_dir)
    target_root = os.path.abspath(target_root)
    print(f"Processing directory tree {source_dir}\nSaving transcripts to {target_root}")

    if not os.path.isdir(source_dir):
        raise ValueError(f"source_dir {source_dir} does not exist or is not a directory")

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

            print()
            try:
                transcribe_video(
                    pipeline,
                    video_path,
                    output_path,
                    textgrid_root=textgrid_root,
                    auto_download_textgrid=auto_download_textgrid,
                    silence_threshold=silence_threshold,
                    max_segment_seconds=max_segment_seconds,
                )
            except Exception as exc:
                print(f"Skipping {video_path} due to error: {exc}")
                maybe_clear_cuda_cache(pipeline.device)
                continue


def parse_args():
    parser = argparse.ArgumentParser(description="Batch AVSR inference for directories of videos.")
    parser.add_argument("--video-path", type=str, default=VIDEO_PATH,
                        help="Path to a single video for quick testing")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Root directory that contains speaker folders with videos")
    parser.add_argument("--output-dir", type=str, default="transcripts",
                        help="Directory where transcripts will be stored")
    parser.add_argument("--model-conf", type=str, default=MODEL_CONF,
                        help="Path to model configuration JSON")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--textgrid-root", type=str, default=TEXTGRID_CACHE_ROOT,
                        help="Directory used to cache/download TextGrid files from BEAT.")
    parser.add_argument("--auto-download-textgrid", dest="auto_download_textgrid", action="store_true",
                        help="Automatically download missing TextGrid files from Hugging Face (default).")
    parser.add_argument("--no-auto-download-textgrid", dest="auto_download_textgrid", action="store_false",
                        help="Disable automatic TextGrid downloads; fall back to single-pass inference if missing.")
    parser.set_defaults(auto_download_textgrid=True)
    parser.add_argument("--silence-threshold", type=float, default=0.6,
                        help="Minimum silence duration (s) to split consecutive sentences.")
    parser.add_argument("--max-segment-seconds", type=float, default=15.0,
                        help="Maximum sentence segment length before forcing a break at the next silence.")
    parser.add_argument("--detector", type=str, default="mediapipe",
                        help="Face detector to use for landmark extraction")
    parser.add_argument("--face-track", dest="face_track", action="store_true",
                        help="Enable face tracking (landmark detection) for video modalities")
    parser.add_argument("--no-face-track", dest="face_track", action="store_false",
                        help="Disable face tracking if precomputed landmarks are supplied")
    parser.set_defaults(face_track=True)
    parser.add_argument("--device", type=str, default=None,
                        help="Device identifier, e.g., cuda:0 or cpu")
    return parser.parse_args()
    
if __name__ == "__main__":
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
    model_conf = args.model_conf if os.path.isabs(args.model_conf) else os.path.join(base_dir, args.model_conf)
    model_path = args.model_path if os.path.isabs(args.model_path) else os.path.join(base_dir, args.model_path)

    pipeline = InferencePipeline(
        modality,
        model_path,
        model_conf,
        detector=args.detector,
        face_track=args.face_track,
        device=device,
    )

    textgrid_root = args.textgrid_root if os.path.isabs(args.textgrid_root) else os.path.join(base_dir, args.textgrid_root)

    if args.video_dir:
        source_dir = args.video_dir if os.path.isabs(args.video_dir) else os.path.join(base_dir, args.video_dir)
        output_root = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(base_dir, args.output_dir)
        process_directory(
            pipeline,
            source_dir,
            output_root,
            textgrid_root=textgrid_root,
            auto_download_textgrid=args.auto_download_textgrid,
            silence_threshold=args.silence_threshold,
            max_segment_seconds=args.max_segment_seconds,
        )
    else:
        video_path = args.video_path if os.path.isabs(args.video_path) else os.path.join(base_dir, args.video_path)

        if not os.path.exists(video_path):
            video_path = os.path.join(base_dir, "data/clip.mp4")
            print(f"Provided video not found. Falling back to {video_path}")

        output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(base_dir, args.output_dir)
        transcript_name = f"{os.path.splitext(os.path.basename(video_path))[0]}.txt"
        output_path = os.path.join(output_dir, transcript_name)
        transcribe_video(
            pipeline,
            video_path,
            output_path,
            textgrid_root=textgrid_root,
            auto_download_textgrid=args.auto_download_textgrid,
            silence_threshold=args.silence_threshold,
            max_segment_seconds=args.max_segment_seconds,
        )