"""Extract SMIRK parameters and unposed FLAME vertices from a talking video."""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SmirkExtractionConfig:
    video_path: Path
    out_dir: Path
    smirk_root: Path
    checkpoint_path: Path
    flame_model_path: Path
    fps: float = 25.0
    device: str = "auto"
    crop: bool = True
    max_frames: int | None = None
    image_size: int = 224
    crop_scale: float = 1.4
    flame_lmk_embedding_path: Path | None = None


@contextlib.contextmanager
def _smirk_import_context(smirk_root: Path):
    old_cwd = Path.cwd()
    added_paths = [str(smirk_root), str(smirk_root / "src")]
    for path in reversed(added_paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(smirk_root)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def crop_face(frame: np.ndarray, landmarks: np.ndarray, scale: float = 1.4, image_size: int = 224):
    """Return SMIRK's MediaPipe similarity crop transform."""

    from skimage.transform import estimate_transform

    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)
    src_pts = np.array(
        [
            [center[0] - size / 2, center[1] - size / 2],
            [center[0] - size / 2, center[1] + size / 2],
            [center[0] + size / 2, center[1] - size / 2],
        ]
    )
    dst_pts = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    return estimate_transform("similarity", src_pts, dst_pts)


def _resolve_device(device: str):
    import torch

    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_smirk_models(config: SmirkExtractionConfig):
    import importlib
    import torch

    smirk_encoder_mod = importlib.import_module("src.smirk_encoder")
    flame_mod = importlib.import_module("src.FLAME.FLAME")
    mediapipe_mod = importlib.import_module("utils.mediapipe_utils")

    device = _resolve_device(config.device)
    encoder = smirk_encoder_mod.SmirkEncoder().to(device)
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and any("smirk_encoder" in key for key in checkpoint.keys()):
        state = {
            key.replace("smirk_encoder.", ""): value
            for key, value in checkpoint.items()
            if "smirk_encoder" in key
        }
    else:
        state = checkpoint
    encoder.load_state_dict(state, strict=False)
    encoder.eval()

    flame_lmk_path = config.flame_lmk_embedding_path or (config.smirk_root / "assets" / "landmark_embedding.npy")
    flame = flame_mod.FLAME(
        flame_model_path=str(config.flame_model_path),
        flame_lmk_embedding_path=str(flame_lmk_path),
    ).to(device)
    flame.eval()
    return encoder, flame, mediapipe_mod.run_mediapipe, device


def _tensor_to_numpy_dict(outputs: dict[str, Any]) -> dict[str, np.ndarray]:
    converted: dict[str, np.ndarray] = {}
    for key, value in outputs.items():
        converted[key] = value.detach().cpu().numpy().astype(np.float32)
    return converted


def _stack_param_dicts(param_dicts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = sorted({key for item in param_dicts for key in item.keys()})
    stacked: dict[str, np.ndarray] = {}
    for key in keys:
        values = [item[key].reshape(-1) for item in param_dicts]
        stacked[key] = np.stack(values, axis=0).astype(np.float32)
    return stacked


def _params_to_torch(params: dict[str, np.ndarray], frame_idx: int, median_shape: np.ndarray, device):
    import torch

    expression = params["expression_params"][frame_idx : frame_idx + 1]
    jaw = params["jaw_params"][frame_idx : frame_idx + 1]
    eyelid = params["eyelid_params"][frame_idx : frame_idx + 1]
    return {
        "shape_params": torch.from_numpy(median_shape[None]).to(device),
        "expression_params": torch.from_numpy(expression).to(device),
        "jaw_params": torch.from_numpy(jaw).to(device),
        "eyelid_params": torch.from_numpy(eyelid).to(device),
        "pose_params": torch.zeros((1, 3), dtype=torch.float32, device=device),
        "neck_pose_params": torch.zeros((1, 3), dtype=torch.float32, device=device),
        "eye_pose_params": torch.zeros((1, 6), dtype=torch.float32, device=device),
    }


def _neutral_params_to_torch(params: dict[str, np.ndarray], median_shape: np.ndarray, device):
    import torch

    return {
        "shape_params": torch.from_numpy(median_shape[None]).to(device),
        "expression_params": torch.zeros((1, params["expression_params"].shape[1]), dtype=torch.float32, device=device),
        "jaw_params": torch.zeros((1, params["jaw_params"].shape[1]), dtype=torch.float32, device=device),
        "eyelid_params": torch.zeros((1, params["eyelid_params"].shape[1]), dtype=torch.float32, device=device),
        "pose_params": torch.zeros((1, 3), dtype=torch.float32, device=device),
        "neck_pose_params": torch.zeros((1, 3), dtype=torch.float32, device=device),
        "eye_pose_params": torch.zeros((1, 6), dtype=torch.float32, device=device),
    }


def _process_frame(image: np.ndarray, run_mediapipe, config: SmirkExtractionConfig, previous_tform):
    import cv2
    from skimage.transform import warp

    valid = True
    tform = previous_tform
    kpt_mediapipe = run_mediapipe(image)

    if config.crop:
        if kpt_mediapipe is not None:
            landmarks_2d = kpt_mediapipe[..., :2]
            tform = crop_face(image, landmarks_2d, scale=config.crop_scale, image_size=config.image_size)
        elif tform is None:
            valid = False
        else:
            valid = False

        if tform is not None:
            image = warp(
                image,
                tform.inverse,
                output_shape=(config.image_size, config.image_size),
                preserve_range=True,
            ).astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config.image_size, config.image_size))
    return image, tform, valid


def extract_smirk_sequence(config: SmirkExtractionConfig) -> tuple[Path, Path]:
    """Run SMIRK over a video and save params plus unposed FLAME vertices."""

    import cv2
    import torch

    config.out_dir.mkdir(parents=True, exist_ok=True)
    params_path = config.out_dir / "smirk_params.npz"
    vertices_path = config.out_dir / "smirk_flame_vertices.npz"

    for required_path, label in [
        (config.video_path, "video"),
        (config.smirk_root, "SMIRK root"),
        (config.checkpoint_path, "SMIRK checkpoint"),
        (config.flame_model_path, "FLAME model"),
    ]:
        if not required_path.exists():
            raise FileNotFoundError(f"Missing {label}: {required_path}")

    with _smirk_import_context(config.smirk_root):
        encoder, flame, run_mediapipe, device = _load_smirk_models(config)
        cap = cv2.VideoCapture(str(config.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {config.video_path}")

        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or config.fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target_step = 1.0 / float(config.fps)
        next_target_time = 0.0
        frame_idx = 0
        processed = 0
        previous_tform = None
        param_dicts: list[dict[str, np.ndarray]] = []
        frame_times: list[float] = []
        valid_frames: list[bool] = []

        while True:
            ret, image = cap.read()
            if not ret:
                break

            frame_time = frame_idx / source_fps
            frame_idx += 1
            if frame_time + 1e-6 < next_target_time:
                continue
            next_target_time += target_step

            cropped_rgb, previous_tform, valid = _process_frame(
                image, run_mediapipe, config, previous_tform
            )
            tensor = (
                torch.from_numpy(cropped_rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(device)
                / 255.0
            )
            with torch.no_grad():
                outputs = encoder(tensor)
            params_np = {key: value.reshape(-1) for key, value in _tensor_to_numpy_dict(outputs).items()}
            param_dicts.append(params_np)
            frame_times.append(frame_time)
            valid_frames.append(valid)
            processed += 1

            if processed % 25 == 0:
                print(f"  SMIRK frame {processed} processed ({frame_idx}/{total_frames or '?'})")
            if config.max_frames is not None and processed >= config.max_frames:
                break

        cap.release()

        if not param_dicts:
            raise RuntimeError("SMIRK extraction produced no frames")

        params = _stack_param_dicts(param_dicts)
        valid_mask = np.asarray(valid_frames, dtype=bool)
        if not np.any(valid_mask):
            raise RuntimeError("No valid face detections were available for SMIRK extraction")

        median_shape = np.median(params["shape_params"][valid_mask], axis=0).astype(np.float32)
        vertices: list[np.ndarray] = []
        with torch.no_grad():
            neutral_out = flame.forward(_neutral_params_to_torch(params, median_shape, device))
            neutral_vertices = neutral_out["vertices"].detach().cpu().numpy()[0].astype(np.float32)
            faces = flame.faces_tensor.detach().cpu().numpy().astype(np.int32)
            for idx in range(len(frame_times)):
                flame_params = _params_to_torch(params, idx, median_shape, device)
                flame_out = flame.forward(flame_params)
                vertices.append(flame_out["vertices"].detach().cpu().numpy()[0].astype(np.float32))

    np.savez_compressed(
        params_path,
        **params,
        median_shape_params=median_shape,
        frame_times=np.asarray(frame_times, dtype=np.float32),
        valid_frames=valid_mask,
        source_fps=np.float32(source_fps),
        fps=np.float32(config.fps),
        source_video=str(config.video_path),
    )
    np.savez_compressed(
        vertices_path,
        vertices=np.stack(vertices, axis=0).astype(np.float32),
        neutral_vertices=neutral_vertices,
        faces=faces,
        frame_times=np.asarray(frame_times, dtype=np.float32),
        valid_frames=valid_mask,
        source_fps=np.float32(source_fps),
        fps=np.float32(config.fps),
        source_video=str(config.video_path),
    )
    print(f"Saved SMIRK params: {params_path}")
    print(f"Saved SMIRK vertices: {vertices_path}")
    return params_path, vertices_path
