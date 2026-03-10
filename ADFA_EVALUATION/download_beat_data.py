import argparse
from pathlib import Path
from typing import Any, Sequence, Union, cast

from huggingface_hub import snapshot_download

def download_speaker_data(speaker_id="26"):
    repo_id = "H-Liu1997/BEAT"
    local_dir = Path("data/beat_cache")
    
    allow_patterns: Union[str, Sequence[str]]

    if speaker_id == "*" or speaker_id is None:
        allow_patterns = cast(Sequence[str], [
            "beat_english_v0.2.1/beat_english_v0.2.1/*/*.json",
            "beat_english_v0.2.1/beat_english_v0.2.1/*/*.TextGrid",
            "beat_english_v0.2.1/beat_english_v0.2.1/*/*.wav",
        ])
        print(f"Downloading files from {repo_id} for ALL speakers...")
    else:
        allow_patterns = cast(Sequence[str], [
            f"beat_english_v0.2.1/beat_english_v0.2.1/{speaker_id}/*.json",
            f"beat_english_v0.2.1/beat_english_v0.2.1/{speaker_id}/*.TextGrid",
            f"beat_english_v0.2.1/beat_english_v0.2.1/{speaker_id}/*.wav",
        ])
        print(f"Downloading files from {repo_id} for speaker {speaker_id}...")
    
    print(f"Pattern: {allow_patterns}")
    
    try:
        pattern_arg: Any = allow_patterns

        # Download to a cache directory
        download_path = snapshot_download(
            repo_id=repo_id, 
            repo_type="dataset", 
            allow_patterns=pattern_arg, 
            local_dir=str(local_dir),
            local_dir_use_symlinks=False # Download actual files
        )
        
        print(f"Download complete. Files are in {download_path}")
        
        # The files will be deep inside the directory structure.
        # Let's find them and list them.
        extensions = {".json", ".textgrid", ".wav"}
        all_files = [
            f for f in Path(download_path).rglob("*")
            if f.is_file() and f.suffix.lower() in extensions
        ]
        
        if speaker_id == "*" or speaker_id is None:
            downloaded_files = all_files
            json_count = sum(f.suffix.lower() == ".json" for f in downloaded_files)
            textgrid_count = sum(f.suffix.lower() == ".textgrid" for f in downloaded_files)
            wav_count = sum(f.suffix.lower() == ".wav" for f in downloaded_files)
            print(
                f"Found {json_count} JSON files, {textgrid_count} TextGrid files, and {wav_count} WAV files across all speakers."
            )
        else:
            # Filter for the specific speaker
            downloaded_files = [f for f in all_files if f"/{speaker_id}/" in str(f).replace("\\", "/")]
            json_count = sum(f.suffix.lower() == ".json" for f in downloaded_files)
            textgrid_count = sum(f.suffix.lower() == ".textgrid" for f in downloaded_files)
            wav_count = sum(f.suffix.lower() == ".wav" for f in downloaded_files)
            print(
                f"Found {json_count} JSON files, {textgrid_count} TextGrid files, and {wav_count} WAV files for speaker {speaker_id}."
            )
        
        return downloaded_files
        
    except Exception as e:
        print(f"Error downloading: {e}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BEAT files for one speaker or all speakers.")
    parser.add_argument("--speaker-id", default="26", help='Speaker id or "*" for all speakers')
    args = parser.parse_args()

    files = download_speaker_data(args.speaker_id)
    if files:
        print("First 5 files:")
        for f in files[:5]:
            print(f)
