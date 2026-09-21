import argparse
import importlib.util
import json
import os
import pickle
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as tvf
from easydict import EasyDict as edict
from PIL import Image
from torch.utils.data import DataLoader, Subset

MAST3R_PROJECT_PATH = os.environ.get('MAST3R_PROJECT_PATH')
if not MAST3R_PROJECT_PATH:
    raise RuntimeError('MAST3R_PROJECT_PATH must point to the MASt3R project checkout.')

REPO_ROOT = Path(__file__).resolve().parent
_repo_sys_paths = []
for _path_entry in list(sys.path):
    _path = Path(_path_entry or os.getcwd()).resolve()
    if _path == REPO_ROOT:
        _repo_sys_paths.append(_path_entry)
        sys.path.remove(_path_entry)
sys.path.insert(0, os.path.join(MAST3R_PROJECT_PATH, 'dust3r', 'croco'))
sys.path.append(MAST3R_PROJECT_PATH)
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.image import _resize_pil_image
for _path_entry in reversed(_repo_sys_paths):
    sys.path.insert(0, _path_entry)

from datasets.scannet.scannet_datasets import ScannetSuperGlueTestDataset
from datasets.threedmatch.threedmatch_datasets import My3DMatchTestDataset

SC2PCR_MODULE_PATH = Path(__file__).resolve().parent / 'models' / 'SC2PCR' / 'SC2_PCR_MY.py'
SC2PCR_SPEC = importlib.util.spec_from_file_location('cgenreg_sc2pcr', SC2PCR_MODULE_PATH)
if SC2PCR_SPEC is None or SC2PCR_SPEC.loader is None:
    raise RuntimeError(f'Could not load SC2-PCR module from {SC2PCR_MODULE_PATH}.')
SC2PCR_MODULE = importlib.util.module_from_spec(SC2PCR_SPEC)
SC2PCR_SPEC.loader.exec_module(SC2PCR_MODULE)
Matcher = SC2PCR_MODULE.Matcher
transform = SC2PCR_MODULE.transform

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class MyMatcher(Matcher):
    def __init__(self, inlier_threshold=0.10,
                 num_node='all',
                 use_mutual=True,
                 d_thre=0.1,
                 num_iterations=10,
                 ratio=0.2,
                 nms_radius=0.1,
                 max_points=8000,
                 k1=30,
                 k2=20,
                 select_scene=None, ):
        super(MyMatcher, self).__init__(inlier_threshold,
                                        num_node,
                                        use_mutual,
                                        d_thre,
                                        num_iterations,
                                        ratio,
                                        nms_radius,
                                        max_points,
                                        k1,
                                        k2,
                                        select_scene)

    def estimator(self, src_keypts_corr, tgt_keypts_corr):
        """
               Input:
                   - src_keypts: [bs, num_corr, 3]
                   - tgt_keypts: [bs, num_corr, 3]
               Output:
                   - pred_trans:   [bs, 4, 4], the predicted transformation matrix
                   - pred_trans:   [bs, num_corr], the predicted inlier/outlier label (0,1)
                   - src_keypts_corr:  [bs, num_corr, 3], the source points in the matched correspondences
                   - tgt_keypts_corr:  [bs, num_corr, 3], the target points in the matched correspondences
               """

        #################################
        # use the proposed SC2-PCR to estimate the rigid transformation
        #################################
        pred_trans = self.SC2_PCR(src_keypts_corr, tgt_keypts_corr)

        frag1_warp = transform(src_keypts_corr, pred_trans)
        distance = torch.sum((frag1_warp - tgt_keypts_corr) ** 2, dim=-1) ** 0.5
        pred_labels = (distance < self.inlier_threshold).float()

        return pred_trans[0]


def create_view_dict_batch_multi_views(img0, img1, match='full'):
    num_views = img0.shape[0]

    idx_list_src = np.arange(num_views)
    idx_list_tgt = np.arange(num_views, num_views * 2)

    view1_arr = [{'img': img0[ii][None],
                  'true_shape': torch.tensor(img0[ii].shape[-2:])[None],
                  'idx': idx,
                  'instance': f'{idx}'} for ii, idx in enumerate(idx_list_src)]
    view2_arr = [{'img': img1[ii][None],
                  'true_shape': torch.tensor(img1[ii].shape[-2:])[None],
                  'idx': idx,
                  'instance': f'{idx}'} for ii, idx in enumerate(idx_list_tgt)]

    if match == 'full':
        pairs = list(product(view1_arr, view2_arr))
        gen_view1 = {'img': torch.cat([e[0]['img'] for e in pairs]),
                     'true_shape': torch.cat([e[0]['true_shape'] for e in pairs]),
                     'idx': [e[0]['idx'] for e in pairs],
                     'instance': [e[0]['instance'] for e in pairs]}

        gen_view2 = {'img': torch.cat([e[1]['img'] for e in pairs]),
                     'true_shape': torch.cat([e[1]['true_shape'] for e in pairs]),
                     'idx': [e[1]['idx'] for e in pairs],
                     'instance': [e[1]['instance'] for e in pairs]}
    else:
        gen_view1 = {'img': torch.cat([e['img'] for e in view1_arr]),
                     'true_shape': torch.cat([e['true_shape'] for e in view1_arr]),
                     'idx': [e['idx'] for e in view1_arr],
                     'instance': [e['instance'] for e in view1_arr]}

        gen_view2 = {'img': torch.cat([e['img'] for e in view2_arr]),
                     'true_shape': torch.cat([e['true_shape'] for e in view2_arr]),
                     'idx': [e['idx'] for e in view2_arr],
                     'instance': [e['instance'] for e in view2_arr]}

    return gen_view1, gen_view2, idx_list_src, idx_list_tgt


def get_mast3r_features(view1, view2, model, original_size=(480, 640)):
    # Get Unique Inputs for encoder
    view1_unique_ids, idxs, inv_idxs1 = np.unique(view1['idx'], return_index=True, return_inverse=True)
    view1_encoder_input = {'img': view1['img'][idxs],
                           'true_shape': view1['true_shape'][idxs],
                           'idx': [view1['idx'][ii] for ii in idxs],
                           'instance': [view1['instance'][ii] for ii in idxs]}
    view2_unique_ids, idxs, inv_idxs2 = np.unique(view2['idx'], return_index=True, return_inverse=True)
    view2_encoder_input = {'img': view2['img'][idxs],
                           'true_shape': view2['true_shape'][idxs],
                           'idx': [view2['idx'][ii] for ii in idxs],
                           'instance': [view2['instance'][ii] for ii in idxs]}

    # encode the two images --> B,S,D
    with torch.no_grad():
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = model._encode_symmetrized(view1_encoder_input,
                                                                                   view2_encoder_input)

        # Rebuild inputs to decoder
        shape1 = shape1[inv_idxs1]
        feat1 = feat1[inv_idxs1]
        pos1 = pos1[inv_idxs1]
        shape2 = shape2[inv_idxs2]
        feat2 = feat2[inv_idxs2]
        pos2 = pos2[inv_idxs2]

        # combine all ref images into object-centric representation
        dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)

    res2['pts3d_in_other_view'] = res2.pop('pts3d')

    res1['desc'] = res1['desc'][:, :-9]
    res2['desc'] = res2['desc'][:, :-9]
    res1['desc_conf'] = res1['desc_conf'][:, :-9]
    res2['desc_conf'] = res2['desc_conf'][:, :-9]

    feat1 = res1['desc'].permute(0, 3, 1, 2)
    conf1 = res1['desc_conf'][..., None].permute(0, 3, 1, 2)
    feat2 = res2['desc'].permute(0, 3, 1, 2)
    conf2 = res2['desc_conf'][..., None].permute(0, 3, 1, 2)

    feat1 = F.interpolate(feat1, size=original_size, mode='bilinear', align_corners=False)
    conf1 = F.interpolate(conf1, size=original_size, mode='bilinear', align_corners=False)
    feat2 = F.interpolate(feat2, size=original_size, mode='bilinear', align_corners=False)
    conf2 = F.interpolate(conf2, size=original_size, mode='bilinear', align_corners=False)

    # Make output dict
    res1['desc'] = feat1.permute(0, 2, 3, 1)
    res1['desc_conf'] = conf1.permute(0, 2, 3, 1)[..., 0]
    res2['desc'] = feat2.permute(0, 2, 3, 1)
    res2['desc_conf'] = conf2.permute(0, 2, 3, 1)[..., 0]

    return res1, res2

def weighted_svd(src_points: torch.Tensor, ref_points: torch.Tensor, weights=None,
                 orthogonalization=True):
    """Compute rigid transformation from `src_points` to `ref_points` using weighted SVD (Kabsch).

    Args:
        src_points: torch.Tensor (B, N, 3) or (N, 3)
        ref_points: torch.Tensor (B, N, 3) or (N, 3)
        weights: torch.Tensor (B, N) or (N,) (default: None)

    Returns:
    transform: torch.Tensor (B, 4, 4) or (4, 4)
    """

    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    else:
        weights = torch.clamp(weights, 0.)
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-5)
    weights = weights.unsqueeze(2)  # (B, N, 1)

    src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    ref_centroid = torch.sum(ref_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    src_points_centered = src_points - src_centroid  # (B, N, 3)
    ref_points_centered = ref_points - ref_centroid  # (B, N, 3)

    H = src_points_centered.permute(0, 2, 1) @ (weights * ref_points_centered)
    U, _, V = torch.svd(H.cpu())  # H = USV^T, SVD operates faster on CPU than on GPU
    Ut, V = U.transpose(1, 2).to(H.device), V.to(H.device)
    eye = torch.eye(3, device=H.device).unsqueeze(0).repeat(batch_size, 1, 1)
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    R = V @ eye @ Ut

    if orthogonalization:
        rot_0 = R[..., 0] / torch.norm(R[..., 0], dim=-1, keepdim=True)
        rot_1 = R[..., 1] - torch.sum(R[..., 1] * rot_0, dim=-1, keepdim=True) * rot_0
        rot_1 = rot_1 / torch.norm(rot_1, dim=-1, keepdim=True)
        rot_2 = R[..., 2] - torch.sum(R[..., 2] * rot_0, dim=-1, keepdim=True) * rot_0 \
                - torch.sum(R[..., 2] * rot_1, dim=-1, keepdim=True) * rot_1
        rot_2 = rot_2 / torch.norm(rot_2, dim=-1, keepdim=True)
        R = torch.stack([rot_0, rot_1, rot_2], dim=-1)

    t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    transform[:, :3, :3], transform[:, :3, 3] = R, t.squeeze(2)
    if squeeze_first: transform = transform.squeeze(0)
    return transform


class InferenceModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args

        mast3r_ckpt = os.path.join(MAST3R_PROJECT_PATH, 'checkpoints', 'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth')
        self.model = AsymmetricMASt3R.from_pretrained(mast3r_ckpt).eval()

        sc2pcr_config_path = Path(__file__).resolve().parent / "models/SC2PCR/config_3DMatch.json"
        config = json.load(open(sc2pcr_config_path, 'r'))
        config = edict(config)

        self.matcher = MyMatcher(inlier_threshold=config.inlier_threshold,
                                 num_node=config.num_node,
                                 use_mutual=config.use_mutual,
                                 d_thre=config.d_thre,
                                 num_iterations=config.num_iterations,
                                 ratio=config.ratio,
                                 nms_radius=config.nms_radius,
                                 max_points=config.max_points,
                                 k1=config.k1,
                                 k2=config.k2)

        self.local_results = []

    def forward(self, batch):
        dd = batch
        src_matches_pts, tgt_matches_pts, matches_sim = self.mast3r_forward(batch)

        try:
            transform_hat = self.matcher.estimator(src_matches_pts[None], tgt_matches_pts[None])
        except Exception:
            print(f"SC2PCR - Fail do wsvd (Idx: {dd['global_idx']})")
            transform_hat = weighted_svd(src_matches_pts, tgt_matches_pts, matches_sim)

        gt_tform = dd['gt_tform']
        rre = 0.5 * ((transform_hat[:3, :3].T @ gt_tform[:3, :3]).trace() - 1.0)
        rre = 180.0 * torch.arccos(rre.clamp(-1., 1.)) / np.pi
        rte = torch.norm(gt_tform[:3, 3] - transform_hat[:3, 3], dim=-1)
        src_matches_pts_tform = src_matches_pts @ gt_tform[:3, :3].T + gt_tform[:3, 3]
        err = (src_matches_pts_tform - tgt_matches_pts).norm(dim=-1)
        ir = (err <= 0.1).float().mean()

        res_dict = {'ir': float(ir), 'rre': float(rre), 'rte': float(rte), 'global_idx': dd['global_idx'],
                    'tform_hat': transform_hat.cpu().numpy()}

        return res_dict

    def mast3r_forward(self, batch):
        dd = batch
        gen_view1, gen_view2, idx_list_src, idx_list_tgt = create_view_dict_batch_multi_views(
            dd['src_rgb_frames'], dd['tgt_rgb_frames'], match='full'
        )
        original_size = (704, 960)

        max_bs = 16
        bs = gen_view1['img'].shape[0]
        if bs > max_bs:  # split input
            gen_view1_arr = split_dict_by_k(gen_view1, max_bs)
            gen_view2_arr = split_dict_by_k(gen_view2, max_bs)
            num_split = len(gen_view1_arr)
            gen_view1_current_arr = None
            gen_view2_current_arr = None
            for b_idx in range(num_split):
                gen_view1_current = gen_view1_arr[b_idx]
                gen_view2_current = gen_view2_arr[b_idx]
                new_idxs = np.arange(gen_view1_current['img'].shape[0])
                gen_view1_current['idx'] = new_idxs
                gen_view1_current['instance'] = [str(e) for e in new_idxs]
                gen_view2_current['idx'] = new_idxs
                gen_view2_current['instance'] = [str(e) for e in new_idxs]
                gen_src_feat_current_b, gen_tgt_feat_current_b = get_mast3r_features(gen_view1_current,
                                                                                 gen_view2_current,
                                                                                 self.model,
                                                                                 original_size)
                if gen_view1_current_arr is None:
                    gen_view1_current_arr = gen_src_feat_current_b['desc']
                    gen_view2_current_arr = gen_tgt_feat_current_b['desc']
                else:
                    gen_view1_current_arr = torch.cat([gen_view1_current_arr, gen_src_feat_current_b['desc']])
                    gen_view2_current_arr = torch.cat([gen_view2_current_arr, gen_tgt_feat_current_b['desc']])

            desc1 = gen_view1_current_arr
            desc2 = gen_view2_current_arr

        else:
            gen_src_feat_current, gen_tgt_feat_current = get_mast3r_features(gen_view1,
                                                                             gen_view2,
                                                                             self.model,
                                                                             original_size)
            desc1 = gen_src_feat_current['desc']
            desc2 = gen_tgt_feat_current['desc']
            torch.cuda.empty_cache()

        src_ids = np.array(gen_view1['idx'])
        tgt_ids = np.array(gen_view2['idx']) - self.args.num_views
        geo2rgb_map_src = dd['geo2rgb_map_src']
        geo2rgb_map_tgt = dd['geo2rgb_map_tgt']
        sim_rgb, sim_valid_rgb = pair_conditioned_similarity_matrix(
            desc1, desc2, src_ids, tgt_ids, geo2rgb_map_src, geo2rgb_map_tgt
        )
        src_geo_feat = torch.nn.functional.normalize(dd['src_geo_feat'], dim=-1)
        tgt_geo_feat = torch.nn.functional.normalize(dd['tgt_geo_feat'], dim=-1)
        sim_geo = src_geo_feat @ tgt_geo_feat.T
        sim, valid = fuse_similarities(
            sim_rgb,
            sim_valid_rgb,
            sim_geo,
            torch.ones_like(sim_valid_rgb),
            fuse=self.args.fuse_method,
            T_rgb=0.1,
            T_geo=0.1,
        )
        matches, matches_sim = topk_matches(sim, K=self.args.num_matches)
        src_matches_pts = dd['src_geo_pts'][matches[:, 0]]
        tgt_matches_pts = dd['tgt_geo_pts'][matches[:, 1]]

        torch.cuda.empty_cache()
        return src_matches_pts, tgt_matches_pts, matches_sim

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Lightning calls this instead of forward() during inference
        res_dict = self.forward(batch)

        self.local_results.append(res_dict)

        return res_dict

    def on_predict_epoch_end(self, results):
        # Gather Python lists from all ranks
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered = [None] * world_size
            dist.all_gather_object(gathered, self.local_results)  # every rank calls this
        else:
            gathered = [self.local_results]

        if self.global_rank == 0:
            print(f"Results Len: {len(results)}")
            all_results = [item for sublist in gathered for item in sublist]
            print(len(all_results))
            post_process_results(all_results, self.args)
            print("Done - Saving")


def _load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        script_relative = Path(__file__).resolve().parent / p
        if script_relative.exists():
            p = script_relative
        else:
            raise FileNotFoundError(f"--config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args(argv: Optional[Iterable[str]] = None):
    # 1) Pre-parse just --config to know which file to load
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        type=str,
        help="Path to JSON config whose keys match argument names (used as defaults)."
    )
    pre_args, _ = pre.parse_known_args(argv)
    cfg = _load_json(pre_args.config)

    # 2) Full parser (includes --config in help via parents)
    parser = argparse.ArgumentParser(
        parents=[pre],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Script with JSON-config-as-defaults + CLI overrides"
    )
    parser.add_argument('--threedmatch_depth_path', type=str, default="")
    parser.add_argument('--scannet_depth_path', type=str, default="")
    parser.add_argument(
        '--gen_vid_path',
        type=str,
        default="",
        help="Generated-RGB directory containing video_<index> subdirectories.",
    )
    parser.add_argument('--gen_spec_config', type=str, default="")
    parser.add_argument('--results_path', type=str, default="results")
    parser.add_argument('--num_views', type=int, default=4)
    parser.add_argument('--num_matches', type=int, default=8000)
    parser.add_argument('--gen_frames_margin', type=int, default=3)
    parser.add_argument('--benchmark', type=str, default='3dmatch', choices=['3dmatch', 'scannet_superglue'])
    parser.add_argument('--base_geo_scannet_superglue_feat_path', type=str, default="")
    parser.add_argument('--base_geo_3dmatch_feat_path', type=str, default="")
    parser.add_argument('--fuse_method', type=str, default="noisy_and", choices=["noisy_or", 'noisy_and'])

    parser.add_argument('--save_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_samples', type=int, default=None, help='Optional validation-only cap on samples.')

    # 3) Apply JSON as defaults (ignore unknown keys, warn once)
    if isinstance(cfg, dict):
        arg_dests = {a.dest for a in parser._actions if a.dest}
        unknown = set(cfg.keys()) - arg_dests
        if unknown:
            print(f"[warn] Ignoring unknown keys in {pre_args.config}: {sorted(unknown)}")
        defaults = {}
        for a in parser._actions:
            d = a.dest
            if d and d in cfg:
                defaults[d] = cfg[d]
        if defaults:
            parser.set_defaults(**defaults)

    args = parser.parse_args(argv)
    args.effective_config = {
        a.dest: getattr(args, a.dest)
        for a in parser._actions
        if a.dest not in (None, "help")
    }
    print(json.dumps(args.effective_config, indent=2, sort_keys=True, default=str))

    return args


def post_process_results(results, args):
    all_idx = np.array([ee['global_idx'] for ee in results])
    rre_arr = np.array([ee['rre'] for ee in results])
    rte_arr = np.array([ee['rte'] for ee in results])
    ir_arr = np.array([ee['ir'] for ee in results])
    tform_hat_arr = np.stack([ee['tform_hat'] for ee in results])

    sorted_idxs = np.argsort(all_idx)
    all_idx = all_idx[sorted_idxs]
    rre_arr = rre_arr[sorted_idxs]
    rte_arr = rte_arr[sorted_idxs]
    ir_arr = ir_arr[sorted_idxs]
    tform_hat_arr = tform_hat_arr[sorted_idxs]

    # Print Results
    mean_rre = np.mean(rre_arr)
    mead_rre = np.median(rre_arr)
    max_rre = np.max(rre_arr)
    min_rre = np.min(rre_arr)

    mean_rte = np.mean(rte_arr) * 100
    mead_rte = np.median(rte_arr) * 100
    max_rte = np.max(rte_arr) * 100
    min_rte = np.min(rte_arr) * 100

    mean_ir = np.mean(ir_arr) * 100
    mead_ir = np.median(ir_arr) * 100
    max_ir = np.max(ir_arr) * 100
    min_ir = np.min(ir_arr) * 100

    succ_cond = (np.array(rre_arr) <= 15) & (np.array(rte_arr) <= 0.3)
    rr = np.mean(succ_cond) * 100

    print('Results')
    print(f'RRE[deg]| Mean:{mean_rre:0.3f} | Median: {mead_rre:0.3f} | Min: {min_rre:0.3f} | Max: {max_rre:0.3f}')
    print(f'RTE[cm]| Mean:{mean_rte:0.3f} | Median: {mead_rte:0.3f} | Min: {min_rte:0.3f} | Max: {max_rte:0.3f}')
    print(f'IR[%]| Mean:{mean_ir:0.3f} | Median: {mead_ir:0.3f} | Min: {min_ir:0.3f} | Max: {max_ir:0.3f}')
    print(f'RR[%]| {rr:0.3f}')
    print(f'Fail Idxs: {np.where(~succ_cond)}')

    # Save Results
    results_dict = {'ir_arr': ir_arr, 'rre_arr': rre_arr, 'rte_arr': rte_arr,
                    'all_idx': all_idx,
                    'tform_hat_arr': tform_hat_arr}

    save_path = os.path.join(args.results_path, args.save_name + ".pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(results_dict, f)
    print_read_my_results_summary(save_path, rre_arr, rte_arr)


def print_read_my_results_summary(res_path, rre, rte):
    print(f"Load Results: {res_path}")
    print(f"mRRE: {np.mean(rre):0.3f}[deg]| medRRE: {np.median(rre):0.03f}[deg]")
    print(f"Rot Acc| <=5[deg]: {100 * np.mean(rre <= 5):0.2f}[%]|"
          f" <=10[deg]: {100 * np.mean(rre <= 10):0.2f}[%]|"
          f" <=45[deg]: {100 * np.mean(rre <= 45):0.2f}[%]")
    print(f"mRTE: {100 * np.mean(rte):0.3f}[cm]| medRTE: {100 * np.median(rte):0.03f}[cm]")
    print(f"Trans Acc| <=5[cm]: {100 * np.mean(rte <= 0.05):0.2f}[%]|"
          f" <=10[cm]: {100 * np.mean(rte <= 0.10):0.2f}[%]|"
          f" <=45[cm]: {100 * np.mean(rte <= 0.25):0.2f}[%]")


def deafult_collate_fn(data_list):
    data_dict = data_list[0]
    img0 = [ImgNorm(np.asarray(_resize_pil_image(Image.fromarray(np.transpose(e, (1, 2, 0))), 512))) for e in
            data_dict['src_rgb_frames']]
    img0 = torch.stack(img0)
    img0 = torch.concatenate([img0, torch.zeros_like(img0)[:, :, :9]], dim=2)
    img1 = [ImgNorm(np.asarray(_resize_pil_image(Image.fromarray(np.transpose(e, (1, 2, 0))), 512))) for e in
            data_dict['tgt_rgb_frames']]
    img1 = torch.stack(img1)
    img1 = torch.concatenate([img1, torch.zeros_like(img1)[:, :, :9]], dim=2)

    data_dict['src_rgb_frames'] = img0
    data_dict['tgt_rgb_frames'] = img1

    for k in data_dict.keys():
        if k in ['src_rgb_frames', 'tgt_rgb_frames', 'global_idx']:
            continue
        else:
            data_dict[k] = torch.from_numpy(data_dict[k])

    return data_dict


def pair_conditioned_similarity_matrix(
        src_feat: torch.Tensor,  # (K, H, W, d)
        tgt_feat: torch.Tensor,  # (K, H, W, d)
        src_ids: torch.Tensor,  # (K,) values in [0, NUM_VIEWS-1]
        tgt_ids: torch.Tensor,  # (K,) values in [0, NUM_VIEWS-1]
        geo2rgb_map_src: torch.Tensor,  # (N1, 3) rows: (src_view, h, w) or (-1,-1,-1)
        geo2rgb_map_tgt: torch.Tensor,  # (N2, 3) rows: (tgt_view, h, w) or (-1,-1,-1)
        metric: str = "cosine",  # "cosine" or "dot"
        eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build N1 x N2 similarity matrix where entry (i,j) compares the src pixel from geo2rgb_map_src[i]
    and the tgt pixel from geo2rgb_map_tgt[j], using the *pair-conditioned* features determined by
    (src_view_i, tgt_view_j).

    Returns:
        sim:   (N1, N2) float tensor
        valid: (N1, N2) bool tensor (True iff both points valid AND the (src_view, tgt_view) pair exists)
    """
    # Basic checks
    assert src_feat.ndim == 4 and tgt_feat.ndim == 4, "feat must be (K,H,W,d)"
    assert src_feat.shape == tgt_feat.shape, "src_feat and tgt_feat must have the same shape"
    assert src_ids.shape == tgt_ids.shape and src_ids.ndim == 1, "ids must be (K,)"
    assert geo2rgb_map_src.ndim == 2 and geo2rgb_map_src.shape[1] == 3
    assert geo2rgb_map_tgt.ndim == 2 and geo2rgb_map_tgt.shape[1] == 3

    device = src_feat.device
    K, H, W, d = src_feat.shape
    N1 = geo2rgb_map_src.shape[0]
    N2 = geo2rgb_map_tgt.shape[0]

    # Typically V_src == V_tgt == NUM_VIEWS
    V = int(src_ids.max().item()) + 1

    # Build (src_view, tgt_view) -> k lookup table
    pair_k_map = torch.full((V, V), -1, dtype=torch.long, device=device)
    # If duplicates exist, last one wins; expected unique mapping for your case
    pair_k_map[src_ids, tgt_ids] = torch.arange(K, device=device, dtype=torch.long)

    # Validate rows/cols
    src_valid = (geo2rgb_map_src >= 0).all(dim=1)
    tgt_valid = (geo2rgb_map_tgt >= 0).all(dim=1)

    # Prepare outputs
    sim = torch.zeros((N1, N2), dtype=src_feat.dtype, device=device)
    valid = torch.zeros((N1, N2), dtype=torch.bool, device=device)

    # Group rows by src_view and columns by tgt_view (only valid rows/cols)
    # For each v in [0..V-1], get indices of rows/cols with that view
    # Note: If some views have no queries, groups will be empty (skipped).
    for s_view in range(V):
        rows_mask = src_valid & (geo2rgb_map_src[:, 0] == s_view)
        if not rows_mask.any():
            continue
        row_idx = rows_mask.nonzero(as_tuple=False).squeeze(1)
        sh = geo2rgb_map_src[row_idx, 1]
        sw = geo2rgb_map_src[row_idx, 2]

        for t_view in range(V):
            k = pair_k_map[s_view, t_view]
            if k < 0:
                continue  # no pair-conditioned features for this (s_view, t_view)

            cols_mask = tgt_valid & (geo2rgb_map_tgt[:, 0] == t_view)
            if not cols_mask.any():
                continue
            col_idx = cols_mask.nonzero(as_tuple=False).squeeze(1)
            th = geo2rgb_map_tgt[col_idx, 1]
            tw = geo2rgb_map_tgt[col_idx, 2]

            # Gather features for this block in one shot
            # Source features for all selected rows, from the k-th src map
            src_k = src_feat[k]  # (H,W,d)
            src_vecs = src_k[sh, sw]  # (len(row_idx), d)

            # Target features for all selected cols, from the k-th tgt map
            tgt_k = tgt_feat[k]  # (H,W,d)
            tgt_vecs = tgt_k[th, tw]  # (len(col_idx), d)

            # Compute similarity block: (R, d) @ (d, C) -> (R, C)
            if metric == "cosine":
                src_norm = torch.nn.functional.normalize(src_vecs, dim=-1, eps=eps)
                tgt_norm = torch.nn.functional.normalize(tgt_vecs, dim=-1, eps=eps)
                block = src_norm @ tgt_norm.T  # (R, C)
            elif metric == "dot":
                block = src_vecs @ tgt_vecs.T
            else:
                raise ValueError(f"Unsupported metric '{metric}'. Use 'cosine' or 'dot'.")

            # Write block
            sim.index_put_((row_idx.unsqueeze(1), col_idx.unsqueeze(0)), block)
            valid.index_put_((row_idx.unsqueeze(1), col_idx.unsqueeze(0)),
                             torch.ones((row_idx.numel(), col_idx.numel()), dtype=torch.bool, device=device))

    return sim, valid


@torch.no_grad()
def topk_matches(
        F: torch.Tensor,  # (N1, N2) fused similarity matrix, values in [0,1]
        K: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select the K highest values in F and return their (i,j) indices.

    Args:
        F: (N1, N2) similarity matrix
        K: number of matches to extract

    Returns:
        matches: (K, 2) long tensor with (i,j) indices
        scores:  (K,) tensor with the corresponding scores
    """
    N1, N2 = F.shape
    flat = F.flatten()  # (N1*N2,)
    K_eff = min(K, flat.numel())  # safety

    scores, idx = torch.topk(flat, k=K_eff)
    i = idx // N2
    j = idx % N2
    matches = torch.stack([i, j], dim=1)
    return matches, scores


def _prob_from_similarity(S: torch.Tensor, mask: torch.Tensor, *, mode="rc_softmax", T=0.1):
    """Map raw similarities to probabilities in [0,1] under mask."""
    assert S.shape == mask.shape and mask.dtype == torch.bool
    if mode == "sigmoid":
        P = torch.sigmoid(S / T)
        return torch.where(mask, P, torch.zeros_like(P))
    elif mode == "rc_softmax":
        neg_inf = torch.finfo(S.dtype).min / 4
        Smask = torch.where(mask, S, torch.full_like(S, neg_inf))
        Pr = torch.softmax(Smask / T, dim=1)
        Pc = torch.softmax((Smask / T).transpose(0, 1), dim=1).transpose(0, 1)
        P = torch.sqrt(Pr * Pc + 1e-9)
        return torch.where(mask, P, torch.zeros_like(P))
    else:
        raise ValueError("mode must be 'sigmoid' or 'rc_softmax'")


def _topk_renorm(P: torch.Tensor, V: torch.Tensor, k: int, dim: int, eps: float = 1e-12):
    """
    Keep top-k along 'dim' (respecting validity mask V), zero the rest, then renormalize along 'dim'.
    No-op if k<=0. Works for dim=1 (rows) or dim=0 (cols).
    """
    if k is None or k <= 0:
        return P

    # Mask invalid to -inf for selection (but renorm uses original P values)
    neg_inf = torch.finfo(P.dtype).min / 4
    P_sel = torch.where(V, P, torch.full_like(P, neg_inf))

    k_eff = min(k, P.shape[dim])
    top_vals, top_idx = torch.topk(P_sel, k=k_eff, dim=dim)  # indices into axis 'dim'

    # Build a zero tensor and scatter the ORIGINAL probabilities at top-k positions
    kept = torch.zeros_like(P)
    kept.scatter_(dim, top_idx, torch.gather(P, dim, top_idx))

    # Renormalize along 'dim' only where sum>0
    sums = kept.sum(dim=dim, keepdim=True)
    denom = torch.clamp(sums, min=eps)
    renormed = kept / denom
    # Where sums==0, keep zeros (no valid entries in that slice)
    renormed = torch.where(sums > 0, renormed, kept)
    return renormed

def noisy_and_fusion(P_img: torch.Tensor, P_geo: torch.Tensor) -> torch.Tensor:
    """
    Compute fused probability matrix:
        p_fuse_ij = [p_img_ij * p_geo_ij * (1 - pi_ij)] /
                    [p_img_ij * p_geo_ij * (1 - pi_ij) + (1 - p_img_ij) * (1 - p_geo_ij) * pi_ij]
    assuming uniform pi_ij = pi (default 0.5).

    Args:
        P_img (torch.Tensor): Image branch probabilities [N, M].
        P_geo (torch.Tensor): Geometry branch probabilities [N, M].
        pi (float): Uniform prior π_ij value (default 0.5).

    Returns:
        torch.Tensor: Fused probability matrix [N, M].
    """
    pi = 1/P_img.numel()
    numerator = P_img * P_geo * (1 - pi)
    denominator = numerator + (1 - P_img) * (1 - P_geo) * pi
    P_fuse = numerator / (denominator + 1e-8)  # add small epsilon for stability
    return P_fuse

def fuse_similarities(
        S_rgb: torch.Tensor, V_rgb: torch.Tensor,
        S_geo: torch.Tensor, V_geo: torch.Tensor,
        *,
        calib="rc_softmax",
        T_rgb=0.07,  # lower -> sharper
        T_geo=0.10,
        fuse="noisy_and",
):
    """
    Returns fused matrix F in [0,1] and fused validity mask.
    """
    assert S_rgb.shape == S_geo.shape == V_rgb.shape == V_geo.shape
    V = V_rgb | V_geo

    # 1) Calibrate to probabilities
    P_rgb = _prob_from_similarity(S_rgb, V_rgb, mode=calib, T=T_rgb)
    P_geo = _prob_from_similarity(S_geo, V_geo, mode=calib, T=T_geo)

    # 2) Fill missing entries with 0 (respect masks) and fuse
    Pr = torch.where(V_rgb, P_rgb, torch.zeros_like(P_rgb))
    Pg = torch.where(V_geo, P_geo, torch.zeros_like(P_geo))

    if fuse == "noisy_or":
        F = 1.0 - (1.0 - Pr) * (1.0 - Pg)
    elif fuse == 'noisy_and':
        F = noisy_and_fusion(Pr, Pg)
    else:
        raise ValueError(f"Unknown fuse method: {fuse}")

    F = torch.where(V, F, torch.zeros_like(F))
    return F, V

def split_dict_by_k(data_dict, K):
    """
    Splits a dict of lists/tensors into multiple dicts, each with chunk size K.

    Args:
        data_dict (dict):
            Keys → str
            Values → either:
                - list of Tensors, length = N
                - Tensor with shape (N, ...)
        K (int): chunk size

    Returns:
        list_of_dicts: list of dicts with chunked data
    """
    # Determine N from the first element
    first_key = next(iter(data_dict))
    first_val = data_dict[first_key]

    if isinstance(first_val, list):
        N = len(first_val)
    elif isinstance(first_val, torch.Tensor):
        N = first_val.shape[0]
    else:
        raise TypeError("Values must be list[Tensor] or Tensor")

    output = []

    # Iterate over chunk indices
    for start in range(0, N, K):
        end = min(start + K, N)
        chunk_dict = {}

        for key, val in data_dict.items():

            if isinstance(val, list):
                # List of tensors → slice the list
                chunk_dict[key] = val[start:end]

            elif isinstance(val, torch.Tensor):
                # Tensor → slice along dim 0
                chunk_dict[key] = val[start:end].clone()

            else:
                raise TypeError(f"Unsupported type for key {key}")

        output.append(chunk_dict)

    return output




def build_dataset(args):
    common_kwargs = dict(
        num_views=args.num_views,
        gen_safe_margin=args.gen_frames_margin,
    )
    if args.benchmark == '3dmatch':
        dset = My3DMatchTestDataset(
            args.threedmatch_depth_path,
            args.gen_vid_path,
            os.path.join(args.gen_spec_config, '3dmatch_test.json'),
            base_geo_feat_path=args.base_geo_3dmatch_feat_path,
            **common_kwargs,
        )
    elif args.benchmark == 'scannet_superglue':
        dset = ScannetSuperGlueTestDataset(
            os.path.join(args.scannet_depth_path, 'ScannetSuperGlue'),
            args.gen_vid_path,
            os.path.join(args.gen_spec_config, 'scannet_superglue_test.json'),
            base_geo_feat_path=args.base_geo_scannet_superglue_feat_path,
            **common_kwargs,
        )
    else:
        raise NotImplementedError(f"Unsupported benchmark: {args.benchmark}")

    subset_idxs = dset.get_non_empty_videos_idxs()
    if args.benchmark == 'scannet_superglue':
        geo_idxs = dset.geo_metadata_idxs
        subset_idxs = [
            idx for idx in subset_idxs
            if isinstance(idx, (int, np.integer)) and idx in geo_idxs
        ]
    dset = Subset(dset, subset_idxs)
    if args.max_samples is not None:
        dset = Subset(dset, np.arange(min(args.max_samples, len(dset))))
    return dset


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.results_path, exist_ok=True)
    dset = build_dataset(args)
    print(len(dset))

    module = InferenceModule(args)
    dloader = DataLoader(
        dset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=deafult_collate_fn,
    )
    trainer = pl.Trainer(
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus,
        strategy='ddp' if args.gpus > 1 else None,
        logger=False,
    )
    trainer.predict(module, dloader)
    print('Done')


if __name__ == '__main__':
    main()
