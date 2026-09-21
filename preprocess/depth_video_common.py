from pathlib import Path

import cv2
import numpy as np


def load_depth_frames_mm(image_paths):
    depth_list = []
    for path in image_paths:
        img_mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img_mm is None:
            raise FileNotFoundError(f"Could not read depth frame: {path}")
        img_mm = img_mm.astype(np.uint16)
        img_mm[img_mm == 65535] = 0
        depth_m = img_mm.astype(np.float32) / 1000.0
        depth_m = np.clip(depth_m, 0, 4.0)
        depth_list.append(depth_m)
    return depth_list


def normalize_depth_to_uint8_like_depth_anything(depth, eps=1e-8):
    depth = np.asarray(depth, dtype=np.float32)
    valid_mask = depth > 0
    if not np.any(valid_mask):
        return np.zeros(depth.shape, dtype=np.uint8)

    inverse_depth = np.zeros_like(depth)
    inverse_depth[valid_mask] = 1.0 / (depth[valid_mask] + eps)

    min_inv = np.min(inverse_depth[valid_mask])
    max_inv = np.max(inverse_depth[valid_mask])
    normalized_inv = np.zeros_like(depth)
    normalized_inv[valid_mask] = (inverse_depth[valid_mask] - min_inv) / (max_inv - min_inv + eps)
    return (normalized_inv * 255.0).astype(np.uint8)


def write_temporal_depth_video(image_paths, output_path, fps=30):
    depth_list = load_depth_frames_mm(image_paths)
    if not depth_list:
        raise ValueError("Cannot create a depth video from an empty frame list")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = depth_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output_path}")

    try:
        for depth_m in depth_list:
            if depth_m.shape != (height, width):
                raise ValueError(
                    f"All depth frames must have the same shape. "
                    f"Expected {(height, width)}, got {depth_m.shape}"
                )
            norm_uint8 = normalize_depth_to_uint8_like_depth_anything(depth_m)
            writer.write(cv2.cvtColor(norm_uint8, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()

