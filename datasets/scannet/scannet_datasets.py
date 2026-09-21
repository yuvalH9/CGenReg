import copy
import os
import numpy as np
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
import pandas as pd
import cv2
from pathlib import Path
import imageio.v2 as imageio
import open3d as o3d


def voxel_downsample_o3d(points: np.ndarray, voxel_size: float = 0.015):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)
    points_ds = np.asarray(pcd_ds.points, dtype=np.float32)
    return points_ds

def resize_video_pointcloud(video_np, h_new, w_new, interpolation=cv2.INTER_AREA):
    """
    Resize a point cloud video (T x H x W x 3), preserving zero-valued (invalid) pixels.

    Args:
        video_np (np.ndarray): Input point cloud video, shape (T, H, W, 3)
        h_new (int): New height
        w_new (int): New width
        interpolation: cv2 interpolation method

    Returns:
        np.ndarray: Resized video, shape (T, h_new, w_new, 3)
    """
    T, H, W, C = video_np.shape
    assert C == 3, "Expected last dimension to be 3 (xyz)"

    resized_video = np.zeros((T, h_new, w_new, 3), dtype=video_np.dtype)

    for t in range(T):
        frame = video_np[t]  # H x W x 3
        norm = np.linalg.norm(frame, axis=-1)  # H x W
        mask = (norm > 1e-6).astype(np.uint8)  # Valid pixels mask

        # Resize each channel with mask
        resized_mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
        for c in range(3):
            resized_channel = cv2.resize(frame[..., c], (w_new, h_new), interpolation=interpolation)
            resized_channel[resized_mask == 0] = 0  # Set to 0 if invalid
            resized_video[t, ..., c] = resized_channel

    return resized_video


def read_mp4_to_tensor(path):
    cap = cv2.VideoCapture(path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (OpenCV default) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    # Convert to NumPy array: (N, H, W, 3)
    video_np = np.stack(frames, axis=0)

    # Convert to (N, 3, H, W)
    video_np = video_np.transpose(0, 3, 1, 2)

    return video_np  # dtype=np.uint8, shape=(N, 3, H, W)


def load_and_backproject_depths(depth_paths, first_pose_path=""):
    """
    Args:
        depth_paths (list of str): Paths to depth PNG files

    Returns:
        list of np.ndarray of shape (H, W, 3): point clouds in the coordinate system of the first frame
    """
    pointclouds = []

    # Parse common fields from the first path
    first_path = Path(depth_paths[0])
    scene_name = first_path.parts[-3]
    base_path = Path(*first_path.parts[:-3])
    first_pose = None
    intrinsic_matrix = None

    for i, depth_path in enumerate(depth_paths):
        depth_path = Path(depth_path)
        scene_path = depth_path.parent.parent  # <base>/<scene>/
        seq_id = depth_path.parent.name        # seq-<id>

        # 1. Load depth map
        depth = imageio.imread(depth_path)

        # Replace invalid depth values (0) with 0.0 (float) and create mask
        depth = depth.astype(np.float32)
        valid_mask = depth > 0

        # If redkitchen, convert mm to meters
        depth /= 1000.0

        depth_max = 5.0
        depth = np.clip(depth, 0, depth_max)


        H, W = depth.shape

        # 2. Load pose matrix
        pose_path = str(depth_path).replace("depth", "pose").replace('.png', '.txt')
        pose = np.loadtxt(pose_path)
        if i == 0 and first_pose_path == "":
            first_pose = pose
        else:
            first_pose = np.loadtxt(first_pose_path)

        # 3. Load intrinsics once per scene
        if intrinsic_matrix is None:
            intrinsics_path = scene_path / "intrinsic" / "intrinsic_depth.txt"
            intrinsic_matrix = np.loadtxt(intrinsics_path)  # 3x3
        K = intrinsic_matrix

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # 4. Create meshgrid and backproject to camera space
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        cam_points = np.stack((x, y, z), axis=-1)  # (H, W, 3)
        cam_points_h = np.concatenate([cam_points, np.ones_like(z[..., None])], axis=-1)  # (H, W, 4)

        # 5. Transform to world space
        world_points_h = cam_points_h @ pose.T  # (H, W, 4)
        world_points = world_points_h[..., :3]

        # 6. Store first frame pose and compute inverse
        if i == 0:
            # first_pose = pose
            inv_first_pose = np.linalg.inv(first_pose)

        # 7. Transform to first frame coordinates
        world_points_h = np.concatenate([world_points, np.ones_like(z[..., None])], axis=-1)
        aligned_points_h = world_points_h @ inv_first_pose.T
        aligned_points = aligned_points_h[..., :3]  # (H, W, 3)

        # 8. Set invalid points to (0, 0, 0)
        aligned_points[~valid_mask] = 0.0

        pointclouds.append(aligned_points.astype(np.float32))

    return pointclouds, first_pose


def nn1_from_bhw_numpy(
    pc_bhw: np.ndarray,   # (B, H, W, 3)
    pc_n: np.ndarray,     # (N, 3)
    d_thr: float,
):
    """
    For each point in pc_n (N,3), find its 1-NN among VALID points in pc_bhw (B,H,W,3)
    with ||p - q|| <= d_thr. Valid points: norm>0.

    Returns:
        nn_points: (N,3) float32 — nearest valid points (NaN if none)
        nn_bhw_idx: (N,3) int64 — (b,h,w) indices of the nearest valid point (-1 if none)
        nn_dist: (N,) float32 — Euclidean distance to match (inf if none)
    """
    assert pc_bhw.ndim == 4 and pc_bhw.shape[-1] == 3
    assert pc_n.ndim == 2 and pc_n.shape[-1] == 3

    B, H, W, _ = pc_bhw.shape
    N = pc_n.shape[0]

    # 1) Mask valid points
    norms = np.linalg.norm(pc_bhw, axis=-1)
    valid_mask = norms > 0
    num_valid = int(valid_mask.sum())
    if num_valid == 0:
        return (
            np.full((N, 3), np.nan, dtype=np.float32),
            np.full((N, 3), -1, dtype=np.int64),
            np.full((N,), np.inf, dtype=np.float32),
        )

    bhw_idx = np.argwhere(valid_mask)             # (M,3) [b,h,w]
    valid_points = pc_bhw[valid_mask]             # (M,3)
    valid_points = valid_points.astype(np.float32)
    pc_n = pc_n.astype(np.float32)
    d_thr2 = float(d_thr) ** 2

    nn_points_out = np.full((N, 3), np.nan, dtype=np.float32)
    nn_bhw_out = np.full((N, 3), -1, dtype=np.int64)
    nn_dist_out = np.full((N,), np.inf, dtype=np.float32)

    tree = cKDTree(valid_points)
    d, i = tree.query(pc_n, k=1, workers=-1)
    ok = np.isfinite(d) & (d <= d_thr)
    if np.any(ok):
        nn_points_out[ok] = valid_points[i[ok]]
        nn_bhw_out[ok] = bhw_idx[i[ok]]
        nn_dist_out[ok] = d[ok]

    return nn_points_out, nn_bhw_out, nn_dist_out

class ScannetSuperGlueTestDataset(Dataset):
    def __init__(self, depth_path, gen_vid_path, gen_spec_config_path, num_views=20, gen_safe_margin=3,
                 base_geo_feat_path=""):
        self.depth_path = depth_path
        self.gen_vid_folder_path = gen_vid_path
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.metadata = pd.read_pickle(os.path.join(base_path, "scannet_test_superglue_metadata_valid.pkl"))
        self.num_views = num_views
        self.gen_safe_margin = gen_safe_margin
        self.base_geo_feat_path = base_geo_feat_path
        self.geo_files = sorted(os.listdir(self.base_geo_feat_path))
        self.geo_file_by_key = {}
        for geo_file in self.geo_files:
            _, geo_key = geo_file.split('__', 1)
            self.geo_file_by_key[geo_key] = geo_file
        self.geo_metadata_idxs = set()
        for metadata_idx, metadata in enumerate(self.metadata):
            geo_key = f"{metadata['scene_name']}@{metadata['src_id']}@{metadata['tgt_id']}.pkl"
            if geo_key in self.geo_file_by_key:
                self.geo_metadata_idxs.add(metadata_idx)

        # Get selected file idxes
        gen_spec_cfg = pd.read_json(gen_spec_config_path, lines=True)
        gen_spec_cfg = list(gen_spec_cfg['control_overrides'])
        gen_vid_input_files = [e['depth']['input_control'] for e in gen_spec_cfg]
        self.selected_file_idxs = [int(os.path.split(e)[1].split('_')[0]) for e in gen_vid_input_files]

    def get_non_empty_videos_idxs(self):
        non_empty_idxs = []
        folders = os.listdir(self.gen_vid_folder_path)
        folders.sort(key=lambda x: int(x.split("_")[1]))
        folders = [os.path.join(self.gen_vid_folder_path, ee) for ee in folders]
        for i, folder in enumerate(folders):
            # `os.listdir(folder)` lists the contents of the folder
            # If the folder has at least one entry => not empty
            if os.listdir(folder):
                non_empty_idxs.append(i)

        return list(non_empty_idxs)

    def __len__(self):
        return len(self.selected_file_idxs)

    def __getitem__(self, idx):
        max_num_frames = 50
        actual_num_frames = max_num_frames - self.gen_safe_margin
        file_idx = self.selected_file_idxs[idx]
        data_dict = copy.deepcopy(self.metadata[file_idx])

        # Get depth frames ids
        src_frames = data_dict['src_frames_paths']
        tgt_frames = data_dict['tgt_frames_paths']


        sel_idxs = np.arange(actual_num_frames)[::(actual_num_frames//self.num_views)][:self.num_views]

        # Get generated RGB videos.
        gen_vid_path = os.path.join(self.gen_vid_folder_path, f"video_{idx}", 'output.mp4')
        gen_rgb_vid = read_mp4_to_tensor(gen_vid_path)
        src_rgb_frames = gen_rgb_vid[:(max_num_frames - self.gen_safe_margin)]
        tgt_rgb_frames = gen_rgb_vid[(max_num_frames + self.gen_safe_margin):]
        src_rgb_frames = src_rgb_frames[sel_idxs]
        tgt_rgb_frames = tgt_rgb_frames[sel_idxs]

        # Gen PCs from depth maps
        src_first_pose_path = src_frames[0].replace('depth', 'pose').replace('.png', '.txt')
        src_first_pose_path = os.path.join(self.depth_path, src_first_pose_path)
        tgt_first_pose_path = tgt_frames[0].replace('depth', 'pose').replace('.png', '.txt')
        tgt_first_pose_path = os.path.join(self.depth_path, tgt_first_pose_path)
        data_dict['src_frame_list'] = src_frames[:(max_num_frames - self.gen_safe_margin)]
        data_dict['tgt_frame_list'] = tgt_frames[self.gen_safe_margin:]
        data_dict['src_frame_list'] = [data_dict['src_frame_list'][k] for k in sel_idxs]
        data_dict['tgt_frame_list'] = [data_dict['tgt_frame_list'][k] for k in sel_idxs]
        src_depth_pcs, src_pose_0 = load_and_backproject_depths([os.path.join(self.depth_path, e) for e in data_dict['src_frame_list']], src_first_pose_path)
        tgt_depth_pcs, tgt_pose_0 = load_and_backproject_depths(
            [os.path.join(self.depth_path, e) for e in data_dict['tgt_frame_list']], tgt_first_pose_path)

        # Resize detph to gen vid size
        w_new = 960
        h_new = 704
        inter_method = cv2.INTER_NEAREST  # cv2.INTER_LINEAR
        src_depth_pcs = resize_video_pointcloud(np.stack(src_depth_pcs), h_new, w_new, interpolation=inter_method)
        tgt_depth_pcs = resize_video_pointcloud(np.stack(tgt_depth_pcs), h_new, w_new, interpolation=inter_method)

        # Load PC Frag (agg PCs)
        src_frag_pc = np.stack(src_depth_pcs)
        src_frag_pc = src_frag_pc[np.linalg.norm(src_frag_pc, axis=-1) > 0]
        src_frag_pc = voxel_downsample_o3d(src_frag_pc, voxel_size=0.03)
        src_frag_pc = src_frag_pc.astype(np.float32)

        tgt_frag_pc = np.stack(tgt_depth_pcs)
        tgt_frag_pc = tgt_frag_pc[np.linalg.norm(tgt_frag_pc, axis=-1) > 0]
        tgt_frag_pc = voxel_downsample_o3d(tgt_frag_pc, voxel_size=0.03)
        tgt_frag_pc = tgt_frag_pc.astype(np.float32)

        # Load GT tform (src to tgt)
        gt_tform = np.linalg.inv(tgt_pose_0) @ src_pose_0
        gt_tform = gt_tform.astype(np.float32)

        # Make Frag Pcs have same relative tform as depth pcs
        new_tfrom = np.linalg.inv(gt_tform) @ data_dict['gt_tform']
        src_frag_pc = src_frag_pc @ new_tfrom[:3, :3].T + new_tfrom[:3, 3]


        output_dict = {'src_rgb_frames': src_rgb_frames,
                       'tgt_rgb_frames': tgt_rgb_frames,
                       'src_depth_pcs': src_depth_pcs,
                       'tgt_depth_pcs': tgt_depth_pcs,
                       'src_pc_frag': src_frag_pc,
                       'tgt_pc_frag': tgt_frag_pc,
                       'gt_tform': gt_tform,
                       'global_idx': idx,
                       'src_pose_0': src_pose_0,
                       'tgt_pose_0': tgt_pose_0}

        # Get GeoFeat
        geo_key = f"{data_dict['scene_name']}@{data_dict['src_id']}@{data_dict['tgt_id']}.pkl"
        geo_file_name = self.geo_file_by_key[geo_key]
        geo_feat_dict_raw = pd.read_pickle(os.path.join(self.base_geo_feat_path, geo_file_name))

        # Get relevant fields only
        tgt_geo_pts = geo_feat_dict_raw['ref_points_f']
        tgt_geo_feat = geo_feat_dict_raw['ref_feats_f']
        src_geo_pts = geo_feat_dict_raw['src_points_f']
        src_geo_feat = geo_feat_dict_raw['src_feats_f']

        # Store for output
        output_dict['src_geo_pts'] = src_geo_pts
        output_dict['src_geo_feat'] = src_geo_feat
        output_dict['tgt_geo_pts'] = tgt_geo_pts
        output_dict['tgt_geo_feat'] = tgt_geo_feat

        # Get mapping between geo_pts to depth_pts
        d_thr = 0.05
        _, src_nn_idx, _ = nn1_from_bhw_numpy(output_dict['src_depth_pcs'], output_dict['src_geo_pts'], d_thr)
        _, tgt_nn_idx, _ = nn1_from_bhw_numpy(output_dict['tgt_depth_pcs'], output_dict['tgt_geo_pts'], d_thr)
        output_dict['geo2rgb_map_src'] = src_nn_idx
        output_dict['geo2rgb_map_tgt'] = tgt_nn_idx

        return output_dict

