#!/usr/bin/env python3
"""Persistent JSONL inference worker for ICT-FaceKit batch evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from infer_pipeline import InferencePipeline, load_decode_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read video inference requests as JSONL on stdin.")
    parser.add_argument("--model-conf", default="data/LRS3_V_WER19.1/model.json")
    parser.add_argument("--model-path", default="data/LRS3_V_WER19.1/model.pth")
    parser.add_argument("--decode-config", required=True)
    parser.add_argument("--detector", default="mediapipe")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve_path(base_dir: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else base_dir / path)


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    decode_settings = load_decode_settings(resolve_path(base_dir, args.decode_config))
    pipeline = InferencePipeline(
        modality="video",
        model_path=resolve_path(base_dir, args.model_path),
        model_conf=resolve_path(base_dir, args.model_conf),
        detector=args.detector,
        face_track=True,
        device=device,
        decode_settings=decode_settings,
    )

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            request = json.loads(raw_line)
            request_id = request["id"]
            video_path = request["video_path"]
            hypothesis = pipeline(video_path).strip()
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            response = {"id": request_id, "ok": True, "hypothesis": hypothesis}
        except Exception as exc:
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            response = {
                "id": request.get("id") if "request" in locals() else None,
                "ok": False,
                "error": repr(exc),
            }
        print(json.dumps(response, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
