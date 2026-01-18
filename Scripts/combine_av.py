import argparse
import subprocess
import sys
import os

def combine_av(video_path, audio_path, output_path, offset):
    """
    Combines video and audio using ffmpeg with offset and shortest length clipping.

    Args:
        video_path (str): Path to video file.
        audio_path (str): Path to audio file.
        output_path (str): Path to output file.
        offset (float): Audio offset in seconds.
                        Positive delays audio (audio starts later).
                        Negative delays video (video starts later / audio starts earlier).
    """

    cmd = ['ffmpeg', '-y'] # -y to overwrite output

    # Handle offset logic
    # Note: -itsoffset must be placed BEFORE the input file it applies to.

    # Video input configuration
    if offset < 0:
        # Negative offset means we want audio to start "earlier",
        # which is equivalent to delaying the video start.
        cmd.extend(['-itsoffset', str(abs(offset))])
    cmd.extend(['-i', video_path])

    # Audio input configuration
    if offset > 0:
        # Positive offset means we delay the audio start.
        cmd.extend(['-itsoffset', str(offset)])
    cmd.extend(['-i', audio_path])

    # Map streams explicitly:
    # 0:v refers to video stream from first input (video file)
    # 1:a refers to audio stream from second input (audio file)
    cmd.extend(['-map', '0:v', '-map', '1:a'])

    # Codecs
    # Copy video stream to avoid re-encoding quality loss and save time
    # Encode audio to aac for broad compatibility
    cmd.extend(['-c:v', 'copy', '-c:a', 'aac'])

    # -shortest: Finish encoding when the shortest input stream ends
    cmd.extend(['-shortest'])

    cmd.append(output_path)

    print(f"Executing command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully created {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'ffmpeg' command not found. Please ensure ffmpeg is installed and in your PATH.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine video and audio files with synchronization offset.")

    parser.add_argument("-v", "--video", required=True, help="Path to input video file")
    parser.add_argument("-a", "--audio", required=True, help="Path to input audio file")
    parser.add_argument("-o", "--output", required=True, help="Path to output video file")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="Offset in seconds. Positive value delays audio. Negative value delays video.")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)

    output_path = args.output
    if os.path.isdir(output_path):
        base_name = os.path.splitext(os.path.basename(args.video))[0]
        output_path = os.path.join(output_path, f"{base_name}_combined.mp4")
        print(f"Output path is a directory. Saving to: {output_path}")

    combine_av(args.video, args.audio, output_path, args.offset)
