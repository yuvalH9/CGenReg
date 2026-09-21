import argparse
import os
import pickle
import random
import sys
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from configs.geotrans_3dmatch_config import make_cfg
from datasets.scannet.scannet_datasets import load_and_backproject_depths, voxel_downsample_o3d
from geotransformer.utils.data import calibrate_neighbors_stack_mode, registration_collate_fn_stack_mode
from models.geotransformer.model_3dmatch import GeoTransformer


DEFAULT_CKPT_PATH = Path("weights/geotransformer-3dmatch.pth.tar")
DEFAULT_3DMATCH_PLY_ROOT = Path("data/3DMatch/test")
DEFAULT_SCANNET_DEPTH_ROOT = Path("data/ScanNetSuperGlue")
DEFAULT_3DMATCH_METADATA = REPO_ROOT / "datasets" / "threedmatch" / "3DMatch_test_metadata.pkl"
DEFAULT_SCANNET_METADATA = REPO_ROOT / "datasets" / "scannet" / "scannet_test_superglue_metadata_valid.pkl"
DEFAULT_SCANNET_VALID_IDXS = REPO_ROOT / "datasets" / "scannet" / "scannet_superglue_valid_idxs_1294_new.npy"
DEFAULT_3DMATCH_FEAT_ROOT = Path("outputs/3DMatch_GeoTrans_Feat")
DEFAULT_SCANNET_FEAT_ROOT = Path("outputs/Scannet_Superglue_GeoTrans_feat")
FEATURE_KEYS = ("ref_points_f", "ref_feats_f", "src_points_f", "src_feats_f")


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def seed_everything_deterministic(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def load_ply_points(path):
    point_cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(point_cloud.points, dtype=np.float32)
    if points.size == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return points


class ThreeDMatchFeatureDataset(Dataset):
    def __init__(self, ply_root, metadata_path=DEFAULT_3DMATCH_METADATA, point_limit=50000):
        self.ply_root = Path(ply_root)
        self.metadata_path = Path(metadata_path)
        self.point_limit = point_limit
        self.base_seed = 0
        with self.metadata_path.open("rb") as f:
            self.metadata_list = pickle.load(f)

    def __len__(self):
        return len(self.metadata_list)

    def _load_fragment(self, scene_name, frag_id, index):
        points = load_ply_points(self.ply_root / scene_name / f"cloud_bin_{frag_id}.ply")
        points = points[np.linalg.norm(points, axis=-1) > 0]
        points = voxel_downsample_o3d(points, voxel_size=0.015).astype(np.float32)
        if self.point_limit is not None and len(points) > self.point_limit:
            rng = np.random.default_rng(self.base_seed + index)
            points = points[rng.choice(len(points), self.point_limit, replace=False)]
        return points

    def __getitem__(self, index):
        metadata = self.metadata_list[index]
        scene_name = metadata["scene_name"]
        frag_id0 = metadata["frag_id0"]
        frag_id1 = metadata["frag_id1"]

        ref_points = self._load_fragment(scene_name, frag_id0, index)
        src_points = self._load_fragment(scene_name, frag_id1, index)

        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = metadata["rotation"]
        transform[:3, 3] = metadata["translation"]
        transform = np.linalg.inv(transform).astype(np.float32)

        return {
            "scene_name": scene_name,
            "ref_frame": f"cloud_bin_{frag_id0}.ply",
            "src_frame": f"cloud_bin_{frag_id1}.ply",
            "ref_points": ref_points,
            "src_points": src_points,
            "ref_feats": np.ones((ref_points.shape[0], 1), dtype=np.float32),
            "src_feats": np.ones((src_points.shape[0], 1), dtype=np.float32),
            "transform": transform,
            "global_idx": index,
            "save_file_name": f"{index}__{scene_name}@{frag_id0}@{frag_id1}.pkl",
        }


class ScannetSuperGlueFeatureDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        metadata_path=DEFAULT_SCANNET_METADATA,
        valid_idxs_path=DEFAULT_SCANNET_VALID_IDXS,
        point_limit=50000,
    ):
        self.dataset_root = Path(dataset_root)
        self.metadata_path = Path(metadata_path)
        self.valid_idxs_path = Path(valid_idxs_path)
        self.point_limit = point_limit
        with self.metadata_path.open("rb") as f:
            metadata_list = pickle.load(f)

        valid_idxs = np.load(self.valid_idxs_path)
        metadata_list = [metadata_list[int(idx)] for idx in valid_idxs]
        self.metadata_list = metadata_list

    def __len__(self):
        return len(self.metadata_list)

    def _make_fragment(self, frame_paths):
        first_pose_path = self.dataset_root / frame_paths[0].replace("depth", "pose").replace(".png", ".txt")
        depth_pcs, _ = load_and_backproject_depths(
            [str(self.dataset_root / frame_path) for frame_path in frame_paths],
            str(first_pose_path),
        )
        frag_pc = np.stack(depth_pcs)
        frag_pc = frag_pc[np.linalg.norm(frag_pc, axis=-1) > 0]
        frag_pc = voxel_downsample_o3d(frag_pc, voxel_size=0.015).astype(np.float32)
        if self.point_limit is not None and len(frag_pc) > self.point_limit:
            frag_pc = frag_pc[np.random.permutation(frag_pc.shape[0])[: self.point_limit]]
        return frag_pc

    def __getitem__(self, index):
        metadata = self.metadata_list[index]
        scene_name = metadata["scene_name"]
        src_id = metadata["src_id"]
        tgt_id = metadata["tgt_id"]

        source_points = self._make_fragment(metadata["src_frames_paths"])
        target_points = self._make_fragment(metadata["tgt_frames_paths"])

        # Match the cached ScanNet SuperGlue convention consumed by scannet_datasets.py:
        # ref_* is the target side and src_* is the source side.
        ref_points = target_points
        src_points = source_points
        transform = np.linalg.inv(metadata["gt_tform"]).astype(np.float32)

        return {
            "scene_name": scene_name,
            "ref_frame": metadata["src_frames_paths"][0],
            "src_frame": metadata["tgt_frames_paths"][0],
            "ref_points": ref_points,
            "src_points": src_points,
            "ref_feats": np.ones((ref_points.shape[0], 1), dtype=np.float32),
            "src_feats": np.ones((src_points.shape[0], 1), dtype=np.float32),
            "transform": transform,
            "global_idx": index,
            "save_file_name": f"{index}__{scene_name}@{src_id}@{tgt_id}.pkl",
        }


def build_dataset(args):
    if args.benchmark == "3dmatch":
        dataset = ThreeDMatchFeatureDataset(
            ply_root=args.threedmatch_ply_path,
            metadata_path=args.threedmatch_metadata_path,
            point_limit=args.point_limit,
        )
    elif args.benchmark == "scannet_superglue":
        dataset = ScannetSuperGlueFeatureDataset(
            dataset_root=Path(args.scannet_depth_path) / "ScannetSuperGlue",
            metadata_path=args.scannet_metadata_path,
            valid_idxs_path=args.scannet_valid_idxs_path,
            point_limit=args.point_limit,
        )
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")

    if args.max_samples is not None:
        dataset = Subset(dataset, np.arange(min(args.max_samples, len(dataset))))
    return dataset


def save_features(model, batch, output_dir, overwrite=False):
    save_file_name = batch["save_file_name"]
    if isinstance(save_file_name, (list, tuple)):
        save_file_name = save_file_name[0]
    output_path = output_dir / save_file_name
    if output_path.exists() and not overwrite:
        print(f"{output_path} - EXISTS")
        return output_path

    with torch.no_grad():
        output_dict = model(batch)

    save_dict = {key: output_dict[key].detach().cpu().numpy() for key in FEATURE_KEYS}
    with output_path.open("wb") as f:
        pickle.dump(save_dict, f)
    print(f"Saved: {output_path}")
    return output_path


def create_features(args):
    seed_everything_deterministic(args.seed)
    cfg = make_cfg()
    cfg.test.num_workers = args.workers
    cfg.fine_matching.num_refinement_steps = 0

    dataset = build_dataset(args)
    print(f"Dataset size: {len(dataset)}")

    neighbor_limits = calibrate_neighbors_stack_mode(
        dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    print(f"Calibrate neighbors: {neighbor_limits}.")

    collate_fn = partial(
        registration_collate_fn_stack_mode,
        num_stages=cfg.backbone.num_stages,
        voxel_size=cfg.backbone.init_voxel_size,
        search_radius=cfg.backbone.init_radius,
        neighbor_limits=neighbor_limits,
        precompute_data=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    device = torch.device("cuda:0" if args.gpus > 0 and torch.cuda.is_available() else "cpu")
    model = GeoTransformer(cfg).to(device)
    print(f'Loading from "{args.ckpt_path}".')
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict["model"], strict=True)
    model.eval()
    print("Model has been loaded.")

    output_dir = Path(args.feat_save_path) / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for batch in tqdm(loader):
        if args.batch_size != 1:
            raise ValueError("GeoTransformer feature saving expects --batch_size 1.")
        batch = move_to_device(batch, device)
        saved_paths.append(save_features(model, batch, output_dir, overwrite=args.overwrite))
    return saved_paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Create GeoTransformer features for cgenreg inference.")
    parser.add_argument("--benchmark", choices=["3dmatch", "scannet_superglue"], required=True)
    parser.add_argument("--save_name", default="geotransformer_features")
    parser.add_argument("--feat_save_path", default=None)
    parser.add_argument("--output_subdir", default="geotransformer_feat_3dmatch")
    parser.add_argument("--ckpt_path", default=str(DEFAULT_CKPT_PATH))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--point_limit", type=int, default=50000)
    parser.add_argument("--threedmatch_ply_path", default=str(DEFAULT_3DMATCH_PLY_ROOT))
    parser.add_argument("--threedmatch_metadata_path", default=str(DEFAULT_3DMATCH_METADATA))
    parser.add_argument("--scannet_depth_path", default=str(DEFAULT_SCANNET_DEPTH_ROOT))
    parser.add_argument("--scannet_metadata_path", default=str(DEFAULT_SCANNET_METADATA))
    parser.add_argument("--scannet_valid_idxs_path", default=str(DEFAULT_SCANNET_VALID_IDXS))
    args = parser.parse_args(argv)

    if args.feat_save_path is None:
        args.feat_save_path = str(DEFAULT_3DMATCH_FEAT_ROOT if args.benchmark == "3dmatch" else DEFAULT_SCANNET_FEAT_ROOT)
    return args


def main(argv=None):
    args = parse_args(argv)
    create_features(args)


if __name__ == "__main__":
    main()
