#!/usr/bin/env python3
"""Prepare the minimal ScanNet data used by the SuperGlue evaluation split.

Users must obtain ScanNet through the official access process. This utility reads
the repository metadata and extracts only the required depth frames, poses, and
camera intrinsics from the downloaded .sens files.
"""

import argparse
import pickle
import shutil
import struct
import subprocess
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_METADATA_PATH = SCRIPT_DIR / "scannet_test_superglue_metadata_valid.pkl"
DEFAULT_VALID_INDICES_PATH = SCRIPT_DIR / "scannet_superglue_valid_idxs_1294_new.npy"
DEFAULT_DOWNLOAD_SCRIPT = SCRIPT_DIR / "download-scannet.py"
DATASET_DIR_NAME = "ScannetSuperGlue"
INTRINSIC_FILENAMES = (
    "intrinsic_color.txt",
    "extrinsic_color.txt",
    "intrinsic_depth.txt",
    "extrinsic_depth.txt",
)


def load_metadata(metadata_path, valid_indices_path, max_pairs=None):
    with Path(metadata_path).open("rb") as handle:
        metadata = pickle.load(handle)
    valid_indices = np.load(valid_indices_path).astype(int)
    metadata = [metadata[index] for index in valid_indices]
    if max_pairs is not None:
        metadata = metadata[:max_pairs]
    return metadata


def collect_required_frames(metadata):
    required = defaultdict(set)
    for item in metadata:
        for key in ("src_frames_paths", "tgt_frames_paths"):
            for relative_path in item[key]:
                path = Path(relative_path)
                if len(path.parts) != 3 or path.parts[1] != "depth" or path.suffix != ".png":
                    raise ValueError(f"Unexpected ScanNet frame path in metadata: {relative_path}")
                required[path.parts[0]].add(int(path.stem))
    return {scene: sorted(frame_ids) for scene, frame_ids in sorted(required.items())}


def resolve_dataset_dir(path):
    path = Path(path)
    nested = path / DATASET_DIR_NAME
    return nested if nested.is_dir() else path


def output_dataset_dir(output_root):
    output_root = Path(output_root)
    if output_root.name == DATASET_DIR_NAME:
        return output_root
    return output_root / DATASET_DIR_NAME


def read_exact(handle, size):
    value = handle.read(size)
    if len(value) != size:
        raise EOFError(f"Unexpected end of .sens file while reading {size} bytes")
    return value


def unpack(handle, fmt):
    fmt = "<" + fmt
    return struct.unpack(fmt, read_exact(handle, struct.calcsize(fmt)))


def read_matrix(handle):
    return np.frombuffer(read_exact(handle, 16 * 4), dtype="<f4").reshape(4, 4).copy()


def write_matrix(path, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, fmt="%f")


def read_sens_header(handle):
    version = unpack(handle, "I")[0]
    if version != 4:
        raise ValueError(f"Unsupported .sens version {version}; expected version 4")

    sensor_name_length = unpack(handle, "Q")[0]
    read_exact(handle, sensor_name_length)
    matrices = {
        "intrinsic_color.txt": read_matrix(handle),
        "extrinsic_color.txt": read_matrix(handle),
        "intrinsic_depth.txt": read_matrix(handle),
        "extrinsic_depth.txt": read_matrix(handle),
    }
    color_compression, depth_compression = unpack(handle, "ii")
    color_width, color_height, depth_width, depth_height = unpack(handle, "IIII")
    depth_shift = unpack(handle, "f")[0]
    num_frames = unpack(handle, "Q")[0]
    return {
        "matrices": matrices,
        "color_compression": color_compression,
        "depth_compression": depth_compression,
        "color_size": (color_width, color_height),
        "depth_size": (depth_width, depth_height),
        "depth_shift": depth_shift,
        "num_frames": num_frames,
    }


def find_sens_files(sens_root):
    sens_root = Path(sens_root)
    sens_files = {}
    for path in sens_root.rglob("*.sens"):
        scene = path.stem
        expected_name = f"{scene}.sens"
        if path.name == expected_name:
            sens_files[scene] = path
    return sens_files


def download_required_scenes(
    download_script,
    download_root,
    requirements,
    download_python=sys.executable,
):
    download_script = Path(download_script).resolve()
    download_root = Path(download_root).resolve()
    if not download_script.is_file():
        raise FileNotFoundError(download_script)
    download_root.mkdir(parents=True, exist_ok=True)

    available = find_sens_files(download_root)
    for index, scene in enumerate(requirements, start=1):
        if scene in available:
            print(f"[{index}/{len(requirements)}] Already downloaded: {scene}")
            continue

        print(f"[{index}/{len(requirements)}] Downloading: {scene}")
        command = [
            str(download_python),
            str(download_script),
            "-o",
            str(download_root),
            "--id",
            scene,
            "--type",
            ".sens",
        ]
        subprocess.run(command, input="\n", text=True, check=True)
        available = find_sens_files(download_root)
        if scene not in available:
            raise FileNotFoundError(
                f"The ScanNet downloader completed but did not create {scene}.sens under {download_root}"
            )

    return download_root


def extract_scene(sens_path, destination, frame_ids, overwrite=False):
    requested = set(frame_ids)
    destination = Path(destination)
    depth_dir = destination / "depth"
    pose_dir = destination / "pose"
    intrinsic_dir = destination / "intrinsic"
    depth_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    intrinsic_dir.mkdir(parents=True, exist_ok=True)

    with Path(sens_path).open("rb") as handle:
        header = read_sens_header(handle)
        if header["depth_compression"] != 1:
            raise ValueError(
                f"Unsupported depth compression code {header['depth_compression']} in {sens_path}; expected zlib"
            )
        missing_ids = requested.difference(range(header["num_frames"]))
        if missing_ids:
            preview = ", ".join(str(value) for value in sorted(missing_ids)[:10])
            raise ValueError(f"{sens_path} does not contain requested frame IDs: {preview}")

        for filename, matrix in header["matrices"].items():
            target = intrinsic_dir / filename
            if overwrite or not target.exists():
                write_matrix(target, matrix)

        depth_width, depth_height = header["depth_size"]
        extracted = 0
        for frame_id in range(header["num_frames"]):
            pose = read_matrix(handle)
            unpack(handle, "QQ")  # color and depth timestamps
            color_size, depth_size = unpack(handle, "QQ")
            handle.seek(color_size, 1)

            if frame_id not in requested:
                handle.seek(depth_size, 1)
                continue

            depth_payload = read_exact(handle, depth_size)
            depth_raw = zlib.decompress(depth_payload)
            depth = np.frombuffer(depth_raw, dtype="<u2").reshape(depth_height, depth_width)
            depth_path = depth_dir / f"{frame_id}.png"
            pose_path = pose_dir / f"{frame_id}.txt"
            if overwrite or not depth_path.exists():
                if not cv2.imwrite(str(depth_path), depth):
                    raise OSError(f"Failed to write depth image: {depth_path}")
            if overwrite or not pose_path.exists():
                write_matrix(pose_path, pose)
            extracted += 1

    if extracted != len(requested):
        raise RuntimeError(f"Extracted {extracted} of {len(requested)} requested frames from {sens_path}")


def extract_from_sens(sens_root, output_root, requirements, overwrite=False):
    sens_files = find_sens_files(sens_root)
    missing_scenes = sorted(set(requirements).difference(sens_files))
    if missing_scenes:
        preview = ", ".join(missing_scenes[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_scenes)} required .sens files under {sens_root}. First missing scenes: {preview}"
        )

    dataset_dir = output_dataset_dir(output_root)
    for index, (scene, frame_ids) in enumerate(requirements.items(), start=1):
        print(f"[{index}/{len(requirements)}] Extracting {scene}: {len(frame_ids)} frames")
        extract_scene(sens_files[scene], dataset_dir / scene, frame_ids, overwrite=overwrite)
    return dataset_dir


def copy_from_exported(exported_root, output_root, requirements, overwrite=False):
    source_root = resolve_dataset_dir(exported_root)
    dataset_dir = output_dataset_dir(output_root)
    for index, (scene, frame_ids) in enumerate(requirements.items(), start=1):
        print(f"[{index}/{len(requirements)}] Copying {scene}: {len(frame_ids)} frames")
        for directory, suffix in (("depth", ".png"), ("pose", ".txt")):
            target_dir = dataset_dir / scene / directory
            target_dir.mkdir(parents=True, exist_ok=True)
            for frame_id in frame_ids:
                source = source_root / scene / directory / f"{frame_id}{suffix}"
                target = target_dir / source.name
                if not source.is_file():
                    raise FileNotFoundError(source)
                if overwrite or not target.exists():
                    shutil.copy2(source, target)

        target_intrinsic_dir = dataset_dir / scene / "intrinsic"
        target_intrinsic_dir.mkdir(parents=True, exist_ok=True)
        for filename in INTRINSIC_FILENAMES:
            source = source_root / scene / "intrinsic" / filename
            target = target_intrinsic_dir / filename
            if not source.is_file():
                raise FileNotFoundError(source)
            if overwrite or not target.exists():
                shutil.copy2(source, target)
    return dataset_dir


def compare_matrix(candidate, reference, label):
    candidate_value = np.loadtxt(candidate)
    reference_value = np.loadtxt(reference)
    if candidate_value.shape != reference_value.shape or not np.allclose(
        candidate_value, reference_value, rtol=0.0, atol=1e-6, equal_nan=True
    ):
        raise AssertionError(f"Matrix mismatch for {label}: {candidate} vs {reference}")


def verify_dataset(dataset_root, requirements, reference_root=None):
    dataset_root = resolve_dataset_dir(dataset_root)
    reference_root = resolve_dataset_dir(reference_root) if reference_root else None
    checked_frames = 0

    for scene, frame_ids in requirements.items():
        intrinsic_depth = dataset_root / scene / "intrinsic" / "intrinsic_depth.txt"
        if not intrinsic_depth.is_file():
            raise FileNotFoundError(intrinsic_depth)
        intrinsic = np.loadtxt(intrinsic_depth)
        if intrinsic.shape != (4, 4) or not np.isfinite(intrinsic).all():
            raise ValueError(f"Invalid depth intrinsics: {intrinsic_depth}")

        for frame_id in frame_ids:
            depth_path = dataset_root / scene / "depth" / f"{frame_id}.png"
            pose_path = dataset_root / scene / "pose" / f"{frame_id}.txt"
            if not depth_path.is_file():
                raise FileNotFoundError(depth_path)
            if not pose_path.is_file():
                raise FileNotFoundError(pose_path)
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            pose = np.loadtxt(pose_path)
            if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
                raise ValueError(f"Invalid 16-bit depth image: {depth_path}")
            if pose.shape != (4, 4):
                raise ValueError(f"Invalid camera pose: {pose_path}")

            if reference_root:
                reference_depth_path = reference_root / scene / "depth" / depth_path.name
                reference_pose_path = reference_root / scene / "pose" / pose_path.name
                reference_depth = cv2.imread(str(reference_depth_path), cv2.IMREAD_UNCHANGED)
                if reference_depth is None or not np.array_equal(depth, reference_depth):
                    raise AssertionError(f"Depth mismatch: {depth_path} vs {reference_depth_path}")
                compare_matrix(pose_path, reference_pose_path, f"{scene} pose {frame_id}")
            checked_frames += 1

        if reference_root:
            for filename in INTRINSIC_FILENAMES:
                compare_matrix(
                    dataset_root / scene / "intrinsic" / filename,
                    reference_root / scene / "intrinsic" / filename,
                    f"{scene} {filename}",
                )

    comparison = " and matched the reference" if reference_root else ""
    print(f"Verified {checked_frames} frames across {len(requirements)} scenes{comparison}.")


def add_metadata_arguments(parser):
    parser.add_argument("--metadata-path", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--valid-indices-path", default=str(DEFAULT_VALID_INDICES_PATH))
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Limit metadata pairs for a smoke test; omit for the complete split.",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-scenes", help="Print the required ScanNet scene IDs.")
    add_metadata_arguments(list_parser)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Download the required official ScanNet scenes, extract the minimal dataset, and verify it.",
    )
    prepare_parser.add_argument(
        "--download-script",
        default=str(DEFAULT_DOWNLOAD_SCRIPT),
        help="ScanNet downloader script (default: the official vendored script next to this file).",
    )
    prepare_parser.add_argument("--download-root", required=True, help="Directory for downloaded .sens files.")
    prepare_parser.add_argument("--output-root", required=True, help="Parent directory for ScannetSuperGlue/.")
    prepare_parser.add_argument(
        "--download-python",
        default=sys.executable,
        help="Python interpreter used to run the official downloader.",
    )
    prepare_parser.add_argument(
        "--accept-scannet-terms",
        action="store_true",
        help="Confirm that you obtained access and accepted the ScanNet Terms of Use.",
    )
    prepare_parser.add_argument("--overwrite", action="store_true")
    add_metadata_arguments(prepare_parser)

    sens_parser = subparsers.add_parser("extract-sens", help="Extract the minimal dataset from official .sens files.")
    sens_parser.add_argument("--sens-root", required=True)
    sens_parser.add_argument("--output-root", required=True)
    sens_parser.add_argument("--overwrite", action="store_true")
    add_metadata_arguments(sens_parser)

    copy_parser = subparsers.add_parser(
        "extract-exported", help="Create the minimal dataset from an already exported ScanNet directory."
    )
    copy_parser.add_argument("--exported-root", required=True)
    copy_parser.add_argument("--output-root", required=True)
    copy_parser.add_argument("--overwrite", action="store_true")
    add_metadata_arguments(copy_parser)

    verify_parser = subparsers.add_parser("verify", help="Verify completeness and optionally compare a reference export.")
    verify_parser.add_argument("--dataset-root", required=True)
    verify_parser.add_argument("--reference-root")
    add_metadata_arguments(verify_parser)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    metadata = load_metadata(args.metadata_path, args.valid_indices_path, args.max_pairs)
    requirements = collect_required_frames(metadata)

    if args.command == "list-scenes":
        for scene in requirements:
            print(scene)
    elif args.command == "prepare":
        if not args.accept_scannet_terms:
            raise ValueError(
                "Pass --accept-scannet-terms only after ScanNet has approved your access and you accepted its terms."
            )
        sens_root = download_required_scenes(
            args.download_script,
            args.download_root,
            requirements,
            download_python=args.download_python,
        )
        dataset_dir = extract_from_sens(sens_root, args.output_root, requirements, args.overwrite)
        verify_dataset(dataset_dir, requirements)
    elif args.command == "extract-sens":
        dataset_dir = extract_from_sens(args.sens_root, args.output_root, requirements, args.overwrite)
        verify_dataset(dataset_dir, requirements)
    elif args.command == "extract-exported":
        dataset_dir = copy_from_exported(args.exported_root, args.output_root, requirements, args.overwrite)
        verify_dataset(dataset_dir, requirements)
    elif args.command == "verify":
        verify_dataset(args.dataset_root, requirements, args.reference_root)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, EOFError, FileNotFoundError, OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
