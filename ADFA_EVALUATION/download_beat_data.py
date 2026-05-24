import argparse
import os
import shutil
import sys
from pathlib import Path

# Avoid the Xet token endpoint when possible; the shared speech-network IP can
# hit that quota quickly on large public datasets.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import DownloadConfig, load_dataset
from huggingface_hub import HfApi


REPO_ID = "H-Liu1997/BEAT"
DATASET_ROOT = "beat_english_v0.2.1/beat_english_v0.2.1"
LOADERS = (
    (".json", "json"),
    (".TextGrid", "text"),
    (".wav", "audiofolder"),
)


def _repo_cache_dir(cache_dir: Path, repo_id: str) -> Path:
    return cache_dir / "hub" / f"datasets--{repo_id.replace('/', '--')}"


def _snapshot_dir(cache_dir: Path, repo_id: str, revision: str = "main") -> Path:
    repo_cache = _repo_cache_dir(cache_dir, repo_id)
    ref = repo_cache / "refs" / revision
    if ref.exists():
        revision = ref.read_text().strip()
    return repo_cache / "snapshots" / revision


def _data_file_pattern(speaker_id: str | None, suffix: str) -> str:
    speaker_glob = "*" if speaker_id in (None, "*") else speaker_id
    return f"hf://datasets/{REPO_ID}/{DATASET_ROOT}/{speaker_glob}/*{suffix}"


def _speaker_filter(speaker_id: str | None) -> str:
    return "*" if speaker_id in (None, "*") else speaker_id


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    if dst.exists():
        dst.unlink()
    try:
        os.link(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def _expected_repo_paths(repo_files: list[str], speaker_id: str | None, suffix: str) -> list[Path]:
    speaker = _speaker_filter(speaker_id)
    prefix = f"{DATASET_ROOT}/"
    paths = []
    for repo_file in repo_files:
        path = Path(repo_file)
        if not repo_file.startswith(prefix) or path.suffix != suffix:
            continue
        if speaker != "*" and len(path.parts) > 2 and path.parts[2] != speaker:
            continue
        paths.append(path)
    return paths


def _cached_paths(cache_dir: Path, repo_paths: list[Path]) -> list[Path]:
    snapshot = _snapshot_dir(cache_dir, REPO_ID)
    return [path for path in repo_paths if (snapshot / path).exists()]


def mirror_hf_cache_to_output(
    *,
    cache_dir: Path,
    local_dir: Path,
    speaker_id: str | None,
    repo_files: list[str] | None = None,
) -> list[Path]:
    snapshot = _snapshot_dir(cache_dir, REPO_ID)
    source_root = snapshot / DATASET_ROOT
    if not source_root.exists():
        print(f"No cached snapshot found at {source_root}")
        return []

    mirrored: list[Path] = []
    if repo_files is not None:
        repo_paths: list[Path] = []
        for suffix, _loader in LOADERS:
            repo_paths.extend(_expected_repo_paths(repo_files, speaker_id, suffix))
    else:
        repo_paths = []
        speaker_glob = _speaker_filter(speaker_id)
        for suffix, _loader in LOADERS:
            repo_paths.extend(path.relative_to(snapshot) for path in source_root.glob(f"{speaker_glob}/*{suffix}"))

    for rel in repo_paths:
        src = snapshot / rel
        if not src.exists():
            continue
        dst = local_dir / rel
        _link_or_copy(src, dst)
        mirrored.append(dst)
    return mirrored


def summarize_files(files: list[Path], speaker_id: str | None) -> None:
    json_count = sum(f.suffix.lower() == ".json" for f in files)
    textgrid_count = sum(f.suffix.lower() == ".textgrid" for f in files)
    wav_count = sum(f.suffix.lower() == ".wav" for f in files)
    scope = "across all speakers" if speaker_id in (None, "*") else f"for speaker {speaker_id}"
    print(f"Found {json_count} JSON files, {textgrid_count} TextGrid files, and {wav_count} WAV files {scope}.")


def download_speaker_data(
    speaker_id: str | None = "26",
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    num_proc: int = 1,
    max_retries: int = 3,
) -> list[Path]:
    local_dir = Path(output_dir or os.environ.get("BEAT_CACHE_DIR", "data/beat_cache")).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    datasets_cache_dir = Path(cache_dir or hf_home).expanduser()

    if speaker_id == "*" or speaker_id is None:
        print(f"Downloading files from {REPO_ID} for ALL speakers with load_dataset...")
    else:
        print(f"Downloading files from {REPO_ID} for speaker {speaker_id} with load_dataset...")
    print(f"Output directory: {local_dir.resolve()}")
    print(f"HF cache directory: {datasets_cache_dir.resolve()}")

    token = True
    download_config = DownloadConfig(
        cache_dir=str(datasets_cache_dir),
        resume_download=True,
        num_proc=num_proc,
        max_retries=max_retries,
        token=token,
    )

    success = True
    repo_files: list[str] | None = None
    try:
        api = HfApi()
        api.repo_info(repo_id=REPO_ID, repo_type="dataset", token=token)
        repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=token)
    except Exception as exc:
        success = False
        print(f"Error listing dataset files: {exc}")

    if repo_files:
        for suffix, loader in LOADERS:
            expected_paths = _expected_repo_paths(repo_files, speaker_id, suffix)
            cached_before = _cached_paths(datasets_cache_dir, expected_paths)
            print(f"Cached {len(cached_before)}/{len(expected_paths)} expected {suffix} files")
            if expected_paths and len(cached_before) == len(expected_paths):
                print(f"Skipping {suffix}: all expected files are already in HF cache")
                continue

            pattern = _data_file_pattern(speaker_id, suffix)
            print(f"load_dataset loader={loader!r} pattern={pattern}")
            try:
                ds = load_dataset(
                    loader,
                    data_files={"train": pattern},
                    split="train",
                    cache_dir=str(datasets_cache_dir),
                    download_config=download_config,
                    token=token,
                    num_proc=num_proc,
                )
                print(f"Cached {len(ds)} records for {suffix}")
            except Exception as exc:
                print(f"Warning: load_dataset failed for {suffix}: {exc}")

            cached_after = _cached_paths(datasets_cache_dir, expected_paths)
            print(f"Cached {len(cached_after)}/{len(expected_paths)} expected {suffix} files after load_dataset")
            if len(cached_after) < len(expected_paths):
                success = False

    print("Mirroring cached files into output directory...")
    mirrored_files = mirror_hf_cache_to_output(
        cache_dir=datasets_cache_dir,
        local_dir=local_dir,
        speaker_id=speaker_id,
        repo_files=repo_files,
    )
    summarize_files(mirrored_files, speaker_id)

    if not success:
        return []
    return mirrored_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BEAT files for one speaker or all speakers.")
    parser.add_argument("--speaker-id", default="26", help='Speaker id or "*" for all speakers')
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where BEAT files should be materialized. Defaults to BEAT_CACHE_DIR or data/beat_cache.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Hugging Face cache directory. Defaults to HF_HOME.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="datasets download parallelism. Keep this low on shared networks to avoid HF rate limits.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry count passed to datasets DownloadConfig.",
    )
    args = parser.parse_args()

    files = download_speaker_data(
        speaker_id=args.speaker_id,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        num_proc=args.num_proc,
        max_retries=args.max_retries,
    )
    if files:
        print("First 5 files:")
        for path in files[:5]:
            print(path)
    else:
        sys.exit(1)
