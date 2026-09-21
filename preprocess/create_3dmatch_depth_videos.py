import argparse
import pickle
import shutil
from pathlib import Path

from tqdm import trange

try:
    from preprocess.depth_video_common import write_temporal_depth_video
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from preprocess.depth_video_common import write_temporal_depth_video


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPTH_ROOT = Path("data/3DMatch/raw/test_raw")
DEFAULT_METADATA_PATH = REPO_ROOT / "datasets" / "threedmatch" / "3DMatch_test_metadata.pkl"


def make_output_name(idx, item):
    scene_name = item["scene_name"]
    src_first = item["src_frame_list"][0]
    tgt_first = item["tgt_frame_list"][0]
    _, seq_src, frame_id_src = src_first.split("/")
    _, seq_tgt, frame_id_tgt = tgt_first.split("/")
    return (
        f"{idx:06d}_{scene_name}-"
        f"{seq_src}_{frame_id_src.split('.')[0]}@"
        f"{seq_tgt}_{frame_id_tgt.split('.')[0]}.mp4"
    )


def create_videos(
    output_path,
    depth_root=DEFAULT_DEPTH_ROOT,
    metadata_path=DEFAULT_METADATA_PATH,
    fps=30,
    start_idx=0,
    limit=None,
    overwrite=False,
):
    depth_root = Path(depth_root)
    metadata_path = Path(metadata_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("rb") as f:
        metadata = pickle.load(f)

    end_idx = len(metadata) if limit is None else min(len(metadata), start_idx + limit)
    for idx in trange(start_idx, end_idx):
        item = metadata[idx]
        frame_paths = item["src_frame_list"] + item["tgt_frame_list"]
        image_paths = [depth_root / frame_path for frame_path in frame_paths]
        output_file = output_path / make_output_name(idx, item)
        if output_file.exists() and not overwrite:
            print(f"{output_file} - EXISTS")
            continue
        write_temporal_depth_video(image_paths, output_file, fps=fps)
        print(f"Video saved to: {output_file}")

    metadata_output = output_path / metadata_path.name
    if overwrite or not metadata_output.exists():
        shutil.copy2(metadata_path, metadata_output)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Create temporal 3DMatch depth videos for cgenreg.")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--depth_root", default=str(DEFAULT_DEPTH_ROOT))
    parser.add_argument("--metadata_path", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    create_videos(
        depth_root=args.depth_root,
        metadata_path=args.metadata_path,
        output_path=args.output_path,
        fps=args.fps,
        start_idx=args.start_idx,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
