"""
Batch Speaker Video Renderer using Pyrender
============================================
Renders all BEAT speakers to video using the fast pyrender pipeline.
Creates speaker_X directories with videos.
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import shutil
import subprocess

# Add Scripts directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from download_beat_data import download_speaker_data

# Import from the specific trimesh file
try:
    from render_face_animation_trimesh import (
        FaceModelNumpy,
        PyrenderRenderer,
        render_animation,
        load_animation
    )
    from face_model_io_trimesh import load_face_model_trimesh
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure 'render_face_animation_trimesh.py' and 'face_model_io_trimesh.py' are in the folder.")
    sys.exit(1)

def main():
    # 1. Download Data
    print(f"--- Step 1: Downloading/Verifying Data for ALL Speakers ---")
    all_json_files = download_speaker_data("*")
    if not all_json_files:
        print("No files found. Exiting.")
        return

    # Group files by speaker
    files_by_speaker = {}
    for f in all_json_files:
        path = Path(f)
        # Structure: .../beat_english_v0.2.1/beat_english_v0.2.1/{speaker_id}/{filename}.json
        spk = path.parent.name

        ## FILTER: Process Speaker 1 only (or remove this check to process all)
        #if spk != "1":
        #    continue

        if spk not in files_by_speaker:
            files_by_speaker[spk] = []
        files_by_speaker[spk].append(path)

    print(f"Found {len(files_by_speaker)} speakers: {sorted(list(files_by_speaker.keys()))}")

    # 2. Setup Renderer
    print(f"\n--- Step 2: Setting up Trimesh/Pyrender ---")

    project_root = script_dir.parent
    model_dir = project_root / 'FaceXModel'

    # LOAD MODEL USING NEW TRIMESH LOADER
    print("Loading model data (Pure NumPy/Trimesh)...")
    # Note: load_identities=False is cleaner for simple expression animation
    model_data = load_face_model_trimesh(str(model_dir), load_identities=False)

    # Initialize NumPy Face Model
    face_model = FaceModelNumpy(model_data)

    # --- UPDATED INITIALIZATION ---
    # Pass the neutral vertices so the renderer can lock the camera position
    renderer = PyrenderRenderer(
        neutral_verts=face_model.neutral_verts,
        image_size=256
    )

    # 3. Render Loop
    print(f"\n--- Step 3: Batch Rendering ---")

    # Sort files
    sorted_speakers = sorted(files_by_speaker.keys())
    max_files = 0

    for spk in sorted_speakers:
        files_by_speaker[spk].sort()
        max_files = max(max_files, len(files_by_speaker[spk]))

        # Create directories
        output_base_dir = project_root / 'sample_data_out' / f'speaker_{spk}'
        output_base_dir.mkdir(parents=True, exist_ok=True)
        (output_base_dir / 'videos').mkdir(exist_ok=True)

    # Interleave tasks
    interleaved_tasks = []
    for i in range(max_files):
        for spk in sorted_speakers:
            if i < len(files_by_speaker[spk]):
                interleaved_tasks.append((spk, files_by_speaker[spk][i]))

    print(f"Total tasks: {len(interleaved_tasks)}")

    # Processing Loop
    for speaker_id, json_path in tqdm(interleaved_tasks, desc="Rendering Videos"):
        output_base_dir = project_root / 'sample_data_out' / f'speaker_{speaker_id}'

        json_path = Path(json_path)
        file_stem = json_path.stem

        video_path = output_base_dir / 'videos' / f'{file_stem}.mp4'
        if video_path.exists():
            continue

        # Load animation
        try:
            anim_data = load_animation(str(json_path))
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            continue

        # Prepare FFmpeg command
        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', '256x256',
            '-pix_fmt', 'rgb24', # Pyrender output format
            '-r', '60',
            '-i', '-',
            '-c:v', 'libopenh264',
            '-pix_fmt', 'yuv420p',
            str(video_path)
        ]

        process = None
        try:
            # Start FFmpeg process
            # stderr=subprocess.DEVNULL silences FFmpeg logs. Remove if debugging is needed.
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

            if process.stdin is None:
                raise RuntimeError("Failed to open stdin for FFmpeg process")

            # Render directly to FFmpeg stdin
            render_animation(
                face_model=face_model,
                anim_data=anim_data,
                renderer=renderer,
                output_dir=None,
                video_writer=process.stdin
            )

            # Close stdin to signal EOF to FFmpeg
            process.stdin.close()
            process.wait()

            if process.returncode != 0:
                print(f"FFmpeg error for {file_stem}")
                if process.stderr:
                    print(process.stderr.read().decode())

        except Exception as e:
            print(f"Error processing {file_stem}: {e}")
            if process:
                process.kill()

    # Cleanup
    renderer.close()
    print("\nBatch rendering complete!")

if __name__ == "__main__":
    main()
