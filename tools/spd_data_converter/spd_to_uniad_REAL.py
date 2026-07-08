#----------------------------------------------------------------#
# UniV2X: End-to-End Autonomous Driving through V2X Cooperation  #
# Source code: https://github.com/AIR-THU/UniV2X                 #
# Copyright (c) DAIR-V2X. All rights reserved.                   #
#----------------------------------------------------------------#

import mmcv
import numpy as np
import os
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.prediction import PredictHelper
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import Dict, List, Tuple, Union

from mmdet3d.core.bbox.box_np_ops import points_cam2img
from mmdet3d.datasets import NuScenesDataset

import os.path as osp
import argparse
import json
import random
import string
from tqdm import tqdm
import uuid
from scipy.linalg import polar
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import yaml  # pip install pyyaml

from tools.utils import x_to_world
INF_W, INF_H = 1920, 1080  # TODO: V2X_REAL HARD CODING 

def _build_and_save_v2xreal_split_json(root_path_base: str,
                                       split_json_path: str,
                                       splits=("train", "val", "test"),
                                       skip_noinfra: bool = True) -> Dict[str, Dict[str, List[str]]]:
    """
    - split_part별 root = root_path_base/data/<split_part>
    - (skip_noinfra) -1, -2 둘다 없으면 해당 scene 스킵
    - vehicle-side 기준: +1 있으면 +1 우선, 없으면 +2
    - ✅ 저장은 frame(000000 등) 없이 scene_token만 저장
    """
    root_path_base = str(root_path_base)
    split_json_path = Path(split_json_path).expanduser().resolve()
    split_json_path.parent.mkdir(parents=True, exist_ok=True)

    batch_split: Dict[str, List[str]] = {}

    for split_part in splits:
        root_path = Path(root_path_base) / f"data/{split_part}"
        scene_tokens_out: List[str] = []

        if not root_path.exists():
            batch_split[split_part] = []
            continue

        run_paths = sorted([p for p in root_path.iterdir() if p.is_dir()])

        for run_path in run_paths:
            scene_token = run_path.name

            if skip_noinfra:
                infra_possible_1 = run_path / "-1"
                infra_possible_2 = run_path / "-2"
                if (not infra_possible_1.is_dir()) and (not infra_possible_2.is_dir()):
                    continue

            # +1 우선, 없으면 +2 (존재 확인용)
            p = run_path / "1"
            if not p.is_dir():
                p = run_path / "2"
            if not p.is_dir():
                continue

            # (선택) 프레임 파일이 하나라도 있는 scene만 포함하고 싶으면 이 체크 유지
            has_any_frame = any(
                f.is_file() and len(f.name) >= 6 and f.name[:6].isdigit()
                for f in p.iterdir()
            )
            if not has_any_frame:
                continue

            scene_tokens_out.append(scene_token)

        # de-dup, keep order
        seen = set()
        uniq: List[str] = []
        for t in scene_tokens_out:
            if t not in seen:
                seen.add(t)
                uniq.append(t)

        batch_split[split_part] = uniq
        print(f"[split-json] {split_part}: {len(uniq)} scenes")

    payload = {"batch_split": batch_split}
    with split_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[split-json] saved -> {split_json_path}")
    return payload


import re
import json
from pathlib import Path
from typing import Dict, Set, List

def build_split_json_from_keep_mapping_tokens(
    keep_mapping_tokens_dict: Dict[str, Set[str]],
    out_json_path: str,
) -> Dict[str, Dict[str, List[str]]]:
    """
    keep_mapping_tokens_dict[split] = set of mapping_token
      mapping_token format:
        "{scene}_folder_{pair}_{frame}_folder_{pair}"

    We want new_scene_token = "{scene}_folder_{pair}" per split.
    """

    out_json_path = Path(out_json_path).expanduser().resolve()
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    # 엄격한 파싱: frame은 6-digit로 가정 (너 코드랑 동일)
    pat = re.compile(r"^(?P<scene>.+?)_folder_(?P<pair>.+?)_(?P<frame>\d{6})_folder_(?P=pair)$")

    batch_split: Dict[str, List[str]] = {}

    for split_part, keep_set in keep_mapping_tokens_dict.items():
        new_scenes = []
        seen = set()

        for s in keep_set:
            m = pat.match(s)
            if not m:
                # 포맷이 살짝 다른게 섞여있으면 여기서 무시/로그
                # print(f"[WARN] unmatched mapping_token: {s}")
                continue

            scene = m.group("scene")
            pair  = m.group("pair")
            new_scene_token = f"{scene}_folder_{pair}"

            if new_scene_token not in seen:
                seen.add(new_scene_token)
                new_scenes.append(new_scene_token)

        batch_split[split_part] = new_scenes
        print(f"[split-json-from-keep] {split_part}: {len(new_scenes)} new_scenes")

    payload = {"batch_split": batch_split}
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[split-json-from-keep] saved -> {out_json_path}")
    return payload


def project_to_so3(R):
    U, _, Vt = np.linalg.svd(R)
    R_hat = U @ Vt
    if np.linalg.det(R_hat) < 0:
        U[:, -1] *= -1
        R_hat = U @ Vt
    return R_hat

def yaw_from_qeg_flat(q_eg: Quaternion) -> float:
    """q_eg(ego->global)에서 roll/pitch 무시한 yaw만 추출 (rad)"""
    R = q_eg.rotation_matrix
    fwd_g = R @ np.array([1.0, 0.0, 0.0])   # ego x-forward in global
    fwd_g[2] = 0.0                          # flatten
    n = np.linalg.norm(fwd_g)
    if n < 1e-9:
        return 0.0
    fwd_g /= n
    return float(np.arctan2(fwd_g[1], fwd_g[0]))

def global_point_to_ego_yaw_only(p_g, t_eg, q_eg):
    """roll/pitch 무시하고 yaw만으로 global->ego 변환"""
    p_g = np.asarray(p_g, dtype=np.float64)
    yaw = yaw_from_qeg_flat(q_eg)                 # ego heading (global z-up)
    q_yaw = Quaternion(axis=[0,0,1], angle=yaw)   # ego->global yaw-only
    return q_yaw.inverse.rotate(p_g - t_eg)

def load_yaml(path: str):
    """Load a YAML file and return Python object (dict/list).
    Raises helpful errors if file missing or YAML is invalid.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML: {path}\n{e}") from e

    # empty file -> None 방지
    if data is None:
        data = {}

    return data

def iterative_closest_point(A, num_iterations=100):
    R = A.copy()

    for _ in range(num_iterations):
        U, _ = polar(R)
        R = U

    return R


from math import pi

class Box3D():
    def __init__(self):
        self.center = None
        self.wlh = None
        self.orientation_yaw_pitch_roll = None
        self.name = None
        self.token = None
        self.instance_token = None
        self.track_id = None
        self.prev_token = None
        self.next_token = None
        self.timestamp = None
        self.visibility = None
        self.gt_velocity = None
        self.prev = None
        self.next = None


def get_cam_intr(calib_path):
    try:
        intr = np.array(load_json(calib_path)['P']).reshape(3, 4)[:, :3]
    except:
        intr = np.array(load_json(calib_path)['cam_K']).reshape(3, 3)

    return intr


def mul_matrix(rotation_1, translation_1, rotation_2, translation_2):
    rotation_1 = np.matrix(rotation_1)
    translation_1 = np.matrix(translation_1).reshape(3, 1)
    rotation_2 = np.matrix(rotation_2)
    translation_2 = np.matrix(translation_2).reshape(3, 1)

    rotation = rotation_2 * rotation_1
    translation = rotation_2 * translation_1 + translation_2
    rotation = np.array(rotation)
    translation = np.array(translation).reshape(3)

    return rotation, translation


visibility_mappings = {
    0: 4,
    1: 3,
    2: 2,
    3: 1
}

# TODO: static_object 는 내가 만든건데, 혹시 문제 생기면 여기서 바꿔주기.... 일단 저렇게 했음.
class_names_nuscenes_mappings = {
    'Car': 'car',
    'Truck': 'car',
    'Van': 'car',
    'Bus': 'car',
    'ConcreteTruck': 'car',
    'PoliceCar': 'car',
    'LongVehicle': 'car',
    'MotorcyleRider': 'bicycle',
    'ScooterRider': 'bicycle',
    'BicycleRider': 'bicycle',
    'Motorcyclist': 'bicycle',
    'Cyclist': 'bicycle',
    'Tricyclist': 'bicycle',
    'Barrowlist': 'bicycle',
    'Pedestrian': 'pedestrian',
    'Child': 'pedestrian',
    'TrafficCone': 'traffic_cone',
    'car': 'car',
    'bicycle': 'bicycle',
    'pedestrian': 'pedestrian',
    'traffic_cone': 'traffic_cone',
    'TrashCan': 'traffic_cone',
    'FireHydrant': 'traffic_cone',
    'RoadWorker': 'pedestrian',
    'Motorcycle': 'bicycle',
    'Scooter': 'bicycle',
    'ConstructionCart': 'traffic_cone'
}

def _try_get_image_wh_from_jpeg_path(jpeg_path: str):
    # PIL 있으면 정확히 얻고, 없으면 None
    try:
        from PIL import Image
        with Image.open(jpeg_path) as im:
            return im.size  # (W, H)
    except Exception:
        return (None, None)

def project_global_point_to_infra_cam_uv(true_pose_veh,
                                        infra_yaml,
                                        cam_key,
                                        img_path=None):
    """
    infra_yaml: i_path/{token}.yaml 로드한 dict
    cam_key: 'cam1' or 'cam2'
    return: (u, v, depth, W, H)  (못하면 (None,None,None,W,H))
    """
    if cam_key not in infra_yaml:
        return None, None, None, None, None

    E = np.array(infra_yaml[cam_key]["extrinsic"], dtype=np.float64)  # cam->lidar (4x4)
    K = np.array(infra_yaml[cam_key]["intrinsic"], dtype=np.float64)  # 3x3

    R_cam2lidar = E[:3, :3]
    t_cam2lidar = E[:3, 3].reshape(3)

    # lidar->cam
    R_lidar2cam = R_cam2lidar.T
    t_lidar2cam = (-R_cam2lidar.T @ t_cam2lidar.reshape(3, 1)).reshape(3)

    # global -> infra ego(lidar) : (p_g - t_eg) rotated by q_eg^{-1}
    true_pose_inf = infra_yaml.get("true_ego_pose", None)
    if true_pose_inf is None:
        return None, None, None, None, None

    # T_ego_to_world = x_to_world(true_ego_pose)
    # p_g = np.array([
    #     annotation['3d_location']['x'],
    #     annotation['3d_location']['y'],
    #     annotation['3d_location']['z']
    # ], dtype=np.float64)

    # T_obj_to_world = x_to_world(np.concatenate([p_g, 
    #                                             np.array(annotation.get('angle', np.zeros(3)))]))


    # T_obj_to_ego = np.linalg.inv(T_ego_to_world) @ T_obj_to_world
    
    # 우리 방식대로 코드 수정 버전: 
    T_ego_to_world = x_to_world(true_pose_inf)
    pose = np.array(true_pose_veh, dtype=np.float64).reshape(-1)
    T_obj_to_world = x_to_world(pose)
    T_obj_to_ego = np.linalg.inv(T_ego_to_world) @ T_obj_to_world   # veh을 inf 기준 좌표계로 이동
    # T_obj_to_ego[:3, 3]
    
    p_lidar = T_obj_to_ego[:3, 3]

    # lidar -> cam
    p_cam = (R_lidar2cam @ p_lidar.reshape(3, 1) + t_lidar2cam.reshape(3, 1)).reshape(3)
    depth = float(p_cam[2])

    if depth <= 1e-6:
        W, H = (None, None)
        if img_path is not None:
            W, H = _try_get_image_wh_from_jpeg_path(img_path)
        return None, None, depth, W, H

    uv = (K @ p_cam.reshape(3, 1)).reshape(3)
    u = float(uv[0] / uv[2])
    v = float(uv[1] / uv[2])

    W, H = (None, None)
    if img_path is not None:
        W, H = _try_get_image_wh_from_jpeg_path(img_path)

    return u, v, depth, W, H




def save_keep_tokens_json(path: Path, keep_tokens: set):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "count": len(keep_tokens),
        "tokens": sorted(list(keep_tokens)),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

def load_keep_tokens_json(path: Path) -> set:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    toks = payload.get("tokens", [])
    return set(toks)

def ego_visible_in_any_infra_cam_fixedWH(veh_yaml: dict,
                                        infra_yaml: dict,
                                        W: int, H: int,
                                        cam_keys=("cam1", "cam2")) -> bool:
    true_pose_veh = veh_yaml.get("true_ego_pose", None)
    if true_pose_veh is None:
        return False

    # t_veh, q_veh = q_ego_to_global_from_true_pose(true_pose_veh, order="roll_yaw_pitch", degrees=True)
    # p_global = np.asarray(t_veh, dtype=np.float64).reshape(3)

    for cam in cam_keys:
        u, v, depth, _, _ = project_global_point_to_infra_cam_uv(
            true_pose_veh, infra_yaml, cam_key=cam, img_path=None
        )
        if depth is None or depth <= 1e-6:
            continue
        if u is None or v is None:
            continue
        if 0.0 <= u < float(W) and 0.0 <= v < float(H):
            return True
    return False

# def build_keep_mapping_tokens_coop_cached(root_path_base: str,
#                                           split_part: str,
#                                           cache_dir: str,
#                                           skip_noinfra: bool = True,
#                                           force_recompute: bool = False):
#     """
#     Returns: set of mapping_token (e.g., "{scene}_{token}_folder_{pair}")
#     Cache: cache_dir/keep_mapping_tokens_{split}.json
#     """
#     cache_dir = Path(cache_dir)
#     cache_path = cache_dir / f"keep_mapping_tokens_{split_part}.json"

#     # 1) load if exists
#     if (not force_recompute) and cache_path.exists():
#         keep = load_keep_tokens_json(cache_path)
#         print(f"[VIS-FILTER][CACHE HIT] {split_part}: {len(keep)} tokens <- {cache_path}")
#         return keep

#     # 2) compute
#     root_path = Path(root_path_base) / f"data/{split_part}"
#     keep = set()

#     if not root_path.exists():
#         print(f"[VIS-FILTER] {split_part}: root missing -> {root_path}")
#         save_keep_tokens_json(cache_path, keep)
#         return keep

#     run_paths = sorted([p for p in root_path.iterdir() if p.is_dir()])

#     # estimate total for progress (optional)
#     est_total = 0
#     scene_meta = []
#     for run_path in run_paths:
#         scene_token = run_path.name

#         if skip_noinfra:
#             if not (run_path / "-1").is_dir() and not (run_path / "-2").is_dir():
#                 continue

#         p1, p2 = run_path / "1", run_path / "2"
#         r1, r2 = run_path / "-1", run_path / "-2"

#         p_list = [(name, path) for name, path in [("1", p1), ("2", p2), ("-1", r1), ("-2", r2)] if path.is_dir()]
#         if len(p_list) == 0:
#             continue

#         veh_list  = [(n, p) for (n, p) in p_list if not n.startswith("-")]
#         infra_list = [(n, p) for (n, p) in p_list if n.startswith("-")]
#         pairs_sample = [(f"{v_name}_{i_name}", v_path, i_path)
#                         for (v_name, v_path) in veh_list
#                         for (i_name, i_path) in infra_list]
#         if len(pairs_sample) == 0:
#             continue

#         token_src = veh_list[0][1] if len(veh_list) > 0 else p_list[0][1]
#         tokens = sorted({
#             f.name[:6]
#             for f in token_src.iterdir()
#             if f.is_file() and f.name[:6].isdigit()
#         })
#         if len(tokens) == 0:
#             continue

#         est_total += len(pairs_sample) * len(tokens)
#         scene_meta.append((run_path, scene_token, pairs_sample, tokens))

#     scene_pbar = tqdm(scene_meta, desc=f"[{split_part}] scenes", total=len(scene_meta))
#     token_pbar = tqdm(total=est_total, desc=f"[{split_part}] pairs×tokens", leave=False)

#     for run_path, scene_token, pairs_sample, tokens in scene_pbar:
#         for pair_name, v_path, i_path in pairs_sample:
#             for token in tokens:
#                 token_pbar.update(1)

#                 veh_yaml_path = osp.join(str(v_path), f"{token}.yaml")
#                 inf_yaml_path = osp.join(str(i_path), f"{token}.yaml")

#                 # 파일 없으면 스킵(데이터 불완전 대비)
#                 if not osp.exists(veh_yaml_path) or not osp.exists(inf_yaml_path):
#                     continue

#                 veh_yaml = load_yaml(veh_yaml_path)
#                 inf_yaml = load_yaml(inf_yaml_path)


#                 # TODO: Delete Wrong Dataset
#                 if v_path == Path("datasets/v2xreal/data/train/2023-04-04-15-58-18_30_0/2"):
#                     # keep.add(f"{scene_token}_folder_{pair_name}_{token}_folder_{pair_name}")
#                     print("Delete v_path: ", v_path)
#                     continue

#                 if ego_visible_in_any_infra_cam_fixedWH(
#                     veh_yaml, inf_yaml, W=INF_W, H=INF_H, cam_keys=("cam1", "cam2")
#                 ):
#                     keep.add(f"{scene_token}_folder_{pair_name}_{token}_folder_{pair_name}")
                


#     token_pbar.close()
#     scene_pbar.close()

#     ## TODO: Delete Wwong Datasets Scenario
#     # keep.add("")
#     # PosixPath('datasets/v2xreal/data/train/2023-04-04-15-58-18_30_0/2')

#     # /home/user/nvme1/v2x/datasets/v2xreal/data/train/2023-04-04-15-58-18_30_0/2/000143.yaml

#     # 3) save
#     save_keep_tokens_json(cache_path, keep)
#     print(f"[VIS-FILTER][CACHE SAVE] {split_part}: {len(keep)} tokens -> {cache_path}")
#     return keep

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import os.path as osp

def build_keep_mapping_tokens_coop_cached(root_path_base: str,
                                          split_part: str,
                                          cache_dir: str,
                                          skip_noinfra: bool = True,
                                          force_recompute: bool = False,
                                          min_visible_per_pair: int = 30):
    """
    Rule (pair-level):
      - For each (scene_token, pair_name):
          if #visible_tokens >= min_visible_per_pair:
              keep ALL tokens for that (scene_token, pair_name)
          else:
              keep only visible tokens for that (scene_token, pair_name)
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"keep_mapping_tokens_{split_part}.json"

    if (not force_recompute) and cache_path.exists():
        keep = load_keep_tokens_json(cache_path)
        print(f"[VIS-FILTER][CACHE HIT] {split_part}: {len(keep)} tokens <- {cache_path}")
        return keep

    root_path = Path(root_path_base) / f"data/{split_part}"
    keep = set()

    if not root_path.exists():
        print(f"[VIS-FILTER] {split_part}: root missing -> {root_path}")
        save_keep_tokens_json(cache_path, keep)
        return keep

    run_paths = sorted([p for p in root_path.iterdir() if p.is_dir()])

    # ---- collect meta ----
    scene_meta = []  # (scene_token, pairs_sample, tokens)
    est_total = 0

    for run_path in run_paths:
        scene_token = run_path.name

        if skip_noinfra:
            if not (run_path / "-1").is_dir() and not (run_path / "-2").is_dir():
                continue

        p1, p2 = run_path / "1", run_path / "2"
        r1, r2 = run_path / "-1", run_path / "-2"

        p_list = [(name, path) for name, path in [("1", p1), ("2", p2), ("-1", r1), ("-2", r2)] if path.is_dir()]
        if len(p_list) == 0:
            continue

        veh_list  = [(n, p) for (n, p) in p_list if not n.startswith("-")]
        infra_list = [(n, p) for (n, p) in p_list if n.startswith("-")]
        pairs_sample = [(f"{v_name}_{i_name}", v_path, i_path)
                        for (v_name, v_path) in veh_list
                        for (i_name, i_path) in infra_list]
        if len(pairs_sample) == 0:
            continue

        token_src = veh_list[0][1] if len(veh_list) > 0 else p_list[0][1]
        tokens = sorted({
            f.name[:6]
            for f in token_src.iterdir()
            if f.is_file() and f.name[:6].isdigit()
        })
        if len(tokens) == 0:
            continue

        est_total += len(pairs_sample) * len(tokens)
        scene_meta.append((scene_token, pairs_sample, tokens))

    # ---- pass1: count visible per (scene_token, pair_name), and record visible tokens ----
    visible_count = defaultdict(int)                # (scene_token, pair_name) -> count
    visible_tokens = defaultdict(list)              # (scene_token, pair_name) -> [token,...]

    scene_pbar = tqdm(scene_meta, desc=f"[{split_part}] scenes", total=len(scene_meta))
    token_pbar = tqdm(total=est_total, desc=f"[{split_part}] pairs×tokens", leave=False)

    for scene_token, pairs_sample, tokens in scene_pbar:
        for pair_name, v_path, i_path in pairs_sample:
            key = (scene_token, pair_name)
            for token in tokens:
                token_pbar.update(1)

                veh_yaml_path = osp.join(str(v_path), f"{token}.yaml")
                inf_yaml_path = osp.join(str(i_path), f"{token}.yaml")
                if not osp.exists(veh_yaml_path) or not osp.exists(inf_yaml_path):
                    continue

                # (네 기존 제외 룰 유지)
                if v_path.as_posix().endswith("datasets/v2xreal/data/train/2023-04-04-15-58-18_30_0/2"):
                    continue

                veh_yaml = load_yaml(veh_yaml_path)
                inf_yaml = load_yaml(inf_yaml_path)

                if ego_visible_in_any_infra_cam_fixedWH(
                    veh_yaml, inf_yaml, W=INF_W, H=INF_H, cam_keys=("cam1", "cam2")
                ):
                    visible_count[key] += 1
                    visible_tokens[key].append(token)

    token_pbar.close()
    scene_pbar.close()

    # ---- pass2: decide keep ----
    for scene_token, pairs_sample, tokens in scene_meta:
        for pair_name, v_path, i_path in pairs_sample:
            key = (scene_token, pair_name)

            if visible_count[key] >= min_visible_per_pair:
                # ✅ 이 (scene,pair) 조합 "시나리오 전체" 포함: 모든 token keep
                for token in tokens:
                    keep.add(f"{scene_token}_folder_{pair_name}_{token}_folder_{pair_name}")
            else:
                # visible token만 keep
                keep_sub_visible_token = False      # if True, 일부라도 보이면 시나리오를 잘라서라도 keep. False면 다 날림.
                if keep_sub_visible_token == True:
                    for token in visible_tokens.get(key, []):
                        keep.add(f"{scene_token}_folder_{pair_name}_{token}_folder_{pair_name}")
                else:
                    pass

    save_keep_tokens_json(cache_path, keep)
    print(f"[VIS-FILTER][CACHE SAVE] {split_part}: {len(keep)} tokens -> {cache_path}")
    return keep

def create_spd_infos_coop(root_path_base,
                          out_path_base,
                          v2x_side,
                          split_path,
                          can_bus_root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10,
                          flag_save=True,
                          skip_noinfra=True):
    """Create info file of spd dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'vehicle-side'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    out_path = osp.join(out_path_base, v2x_side)



    train_spd_infos = []
    val_spd_infos = []
    spd_infos = []
    test_spd_infos = []    
    debug_spd_infos = []
    sample_info_mappings = {}
    total_annotations = {}
    veh_lidar_ego_global_infos = {} 
    inf_lidar_ego_global_infos = {} 
    keep_mapping_tokens_dict = {}
    
    # NOTE: for infra json
    spd_infos_infra = []
    
    for split_part in ["train", "val", "test"]:  # "debug"
    # for split_part in ["debug"]:
    # for split_part in ["debug_train", "debug_val", "debug_test"]:
    
        keep_mapping_tokens = build_keep_mapping_tokens_coop_cached(
            root_path_base, split_part,
            cache_dir=osp.join(out_path_base, "cache_visfilter"),  # 아무 폴더나 OK
            skip_noinfra=skip_noinfra,
            force_recompute=True
        )
        
        print(f"[VIS-FILTER] {split_part}: keep {len(keep_mapping_tokens)} mapping_tokens")
        root_path = osp.join(root_path_base, f"data/{split_part}") 
    

        ## Generate  sample_info_mappings, secene_frame_mappings, total_annotations, instance_token_mappings
        sample_infos, sample_info_mappings = _generate_sample_infos_coop(root_path, sample_info_mappings, skip_noinfra, keep_mapping_tokens)
        secene_frame_mappings = _get_secene_frame_mappings(sample_info_mappings)
        
        # NOTE: annotation 을 불러오는 샘플은 new scene token 기준으로 가져와서, annotation 불러오는 거는 기존 코드로 충분함. (sample_info_mappings_infra로 안써도 됨.)
        total_annotations = _get_total_annotations_coop(root_path, sample_info_mappings, total_annotations, skip_noinfra, keep_mapping_tokens)
        
        instance_token_mappings = _get_instance_token_mappings(total_annotations, sample_info_mappings)

        # get lidar2ego info
        veh_lidar_ego_global_infos = get_lidar_ego_global_infos_coop(root_path, "vehicle-side", veh_lidar_ego_global_infos, skip_noinfra, keep_mapping_tokens)
        inf_lidar_ego_global_infos = get_lidar_ego_global_infos_coop(root_path, "infrastructure-side", inf_lidar_ego_global_infos, skip_noinfra, keep_mapping_tokens)
        
        ## interpolate boxes for unvisible objects
        # NOTE: 이건 veh 기준으로 unvisible 쳐냄.
        total_annotations =  _generate_unvisible_annotations("cooperative",sample_info_mappings,secene_frame_mappings,instance_token_mappings,total_annotations, veh_lidar_ego_global_infos)

        ## update instance_token_mappings
        instance_token_mappings = _get_instance_token_mappings(total_annotations, sample_info_mappings)

        ## add velocity and prev/next, update total_annotations and instance_token_mappings
        # NOTE: veh를 기준으로 anotation 들을 후처리
        total_annotations, instance_token_mappings = _add_annotation_velocity_prev_next(total_annotations, instance_token_mappings, veh_lidar_ego_global_infos)
        # breakpoint()

        root_path = Path(root_path)
        run_paths = sorted([p for p in root_path.iterdir() if p.is_dir()])
        skip_sample_number = 0
        for run_path in tqdm(run_paths):
            scene_token = run_path.name
            
            if skip_noinfra:
                infra_possible_1 = run_path / "-1"
                infra_possible_2 = run_path / "-2"
                if (not infra_possible_1.is_dir()) and (not infra_possible_2.is_dir()):
                    continue
                
                
                
            ## NOTE: 기원 수정중 ======================= ##
            p1 = run_path / "1"
            p2 = run_path / "2"
            r1 = run_path / "-1"
            r2 = run_path / "-2"

            p_list = [(name, path) for name, path in [("1", p1), ("2", p2), ("-1", r1), ("-2", r2)] if path.is_dir()]
            veh_list  = [(n, p) for (n, p) in p_list if not n.startswith("-")]   # "1","2"
            infra_list = [(n, p) for (n, p) in p_list if n.startswith("-")]      # "-1","-2"
            pairs = [((v_name, v_path), (i_name, i_path))
                    for (v_name, v_path) in veh_list
                    for (i_name, i_path) in infra_list]
            pairs_sample = [ (f"{v_name}_{i_name}", v_path, i_path) for ((v_name,v_path),(i_name,i_path)) in pairs ]

            name, path = p_list[0]
                
            # 한 시나리오에 +1, +2, -1, -2 모두 같은 frame 수만큼 들어있는 것으로 이해하고 코드 작성.
            tokens = sorted({
                f.name[:6]
                for f in path.iterdir()
                if f.is_file() and f.name[:6].isdigit()
            })
            
            for pair_name, v_path, i_path in pairs_sample:
                for token in tokens:
                    sample_token = token      # TODO: 이렇게 해야 구분이 가능해짐
                    # breakpoint()
                    mapping_token = scene_token + "_folder_" + pair_name + "_" + sample_token + "_folder_" + pair_name
                    if mapping_token not in keep_mapping_tokens:
                        # print("33passed mapping_token: ", mapping_token)
                        skip_sample_number += 1
                        continue
                    sample_info = sample_info_mappings[mapping_token]
        
                    info = {
                        'token': sample_info['token'],
                        'frame_idx': sample_info['frame_idx'],
                        'scene_token': sample_info['scene_token'],
                        'location': sample_info['location'],
                        'timestamp': sample_info['timestamp'],
                        'prev': sample_info['prev'],
                        'next': sample_info['next'],
                        'other_agent_info_dict': {}
                    }
        
                    other_agent_info = {
                        'token': sample_info['token_inf'],
                        'frame_idx': sample_info['frame_idx'],
                        'scene_token': sample_info['scene_token'],
                        'location': sample_info['location'],
                        'timestamp': sample_info['timestamp_inf'],
                        'prev': sample_info_mappings[sample_info['prev']]['token_inf'] if sample_info['prev'] else '',
                        'next': sample_info_mappings[sample_info['next']]['token_inf'] if sample_info['next'] else '',
                        # 'system_error_offset':sample_info['system_error_offset']
                    }

                    ## Step 2: build camera sensor infos
                    # Step 2.1: build ego_vehicle sensor info
                    # datasets/~ 부터 저장 버전
                    # pointcloud_path = osp.join(p, f"{token}.bin")
                    
                    # train/~ 부터 저장 버전
                    # p2 = p[p.find("train/"):]  
                    # key = f"{split_part}/"
                    # p2 = p[p.index(key):] 
                    p2 = Path(v_path).relative_to(Path(root_path_base) / "data").as_posix()
                    pointcloud_path = osp.join(p2, f"{token}.bin")
                    info['lidar_path'] = pointcloud_path
                    info['lidar2ego_rotation'] = veh_lidar_ego_global_infos[mapping_token]['lidar2ego_rotation']
                    info['lidar2ego_translation'] = veh_lidar_ego_global_infos[mapping_token]['lidar2ego_translation']
                    info['ego2global_rotation'] = veh_lidar_ego_global_infos[mapping_token]['ego2global_rotation']
                    info['ego2global_translation'] = veh_lidar_ego_global_infos[mapping_token]['ego2global_translation']
                    
                    info['cams'] = {}
                        
                    for camera_type in ['cam1', 'cam2', 'cam3', 'cam4']:        # NOTE:  V2X-real에서는 camera가 4대...
                        # datasets/~ 부터 저장 버전
                        # cam_image_path = osp.join(p, f"{token}_{camera_type}.jpeg")
                        
                        # train/~ 부터 저장버전
                        cam_image_path = osp.join(p2, f"{token}_{camera_type}.jpeg")
                        info['cams'][camera_type] = {}
                        info['cams'][camera_type]['data_path'] = cam_image_path

                        key_calib_lidar2cam = 'calib_lidar_to_camera_path'
                        if v2x_side == 'infrastructure-side':   # TODO: 여기는 아직 안함.
                            key_calib_lidar2cam = 'calib_virtuallidar_to_camera_path'                

                        annotation_path = osp.join(v_path, f"{token}.yaml")
                        anno_yaml_veh = load_yaml(annotation_path)          # dict로 로드됨
                        
                        if camera_type in anno_yaml_veh:
                            camera_type_new = camera_type
                        elif camera_type + "_left" in anno_yaml_veh:
                            raise ValueError("위에서 전처리가 됐어여함. 이거 문제있는 폴더임.")
                            camera_type_new = camera_type + "_left"
                        else:
                            raise KeyError(f"camera_type '{camera_type}' or '{camera_type}_left' not found in anno_yaml")

                        anno_yaml_veh[camera_type_new]["cords"]
                        anno_yaml_veh[camera_type_new]["extrinsic"]
                        anno_yaml_veh[camera_type_new]["intrinsic"]

                        E = np.array(anno_yaml_veh[camera_type_new]["extrinsic"], dtype=np.float64)   # 4x4
                        K = np.array(anno_yaml_veh[camera_type_new]["intrinsic"], dtype=np.float64)   # 3x3

                        R_cam2lidar = E[:3, :3]                 # cam -> lidar
                        t_cam2lidar = E[:3, 3].reshape(3,)      # cam -> lidar translation

                        # 2) sensor2lidar (camera -> lidar): use as-is from YAML
                        info['cams'][camera_type]['sensor2lidar_rotation'] = R_cam2lidar
                        info['cams'][camera_type]['sensor2lidar_translation'] = t_cam2lidar

                        # 3) lidar2cam (lidar -> camera): inverse of (cam -> lidar)
                        # inv([R, t]) = [R^T, -R^T t]
                        R_lidar2cam = R_cam2lidar.T
                        t_lidar2cam = (-R_cam2lidar.T @ t_cam2lidar.reshape(3, 1)).reshape(3,)

                        info['cams'][camera_type]['lidar2cam_rotation'] = R_lidar2cam
                        info['cams'][camera_type]['lidar2cam_translation'] = t_lidar2cam
                        
                        cam2ego_r, cam2ego_t = mul_matrix(
                            R_cam2lidar, t_cam2lidar,
                            Quaternion(info['lidar2ego_rotation']).rotation_matrix,
                            np.array(info['lidar2ego_translation'], dtype=np.float64)
                        )
                        
                        # NOTE: 260204 추가 - GW
                        R = np.array(cam2ego_r)               # (3,3)
                        
                        if R.shape != (3, 3):
                            print("hi sample_info['token']: ", sample_info['token'])
                            print("hi sample_info['scene_token']: ", sample_info['scene_token'])
                            raise ValueError(f"cam2ego_r should be (3,3) rotation matrix, got {R.shape}: {R}")
                        # NOTE: det=1.0만족 X 하는 경우 보정 (샘플 한두개만 문제인듯)
                        det = np.linalg.det(R)
                        ortho_err = np.linalg.norm(R.T @ R - np.eye(3))

                        if abs(det - 1.0) > 1e-3 or ortho_err > 1e-3:
                            print(f"[FIX ROT] det={det:.6f} ortho_err={ortho_err:.6e} -> projecting to SO(3)")
                            print("sample_info['token']: ", sample_info['token'])
                            print("sample_info['scene_token']: ", sample_info['scene_token'])
                            R = project_to_so3(R)
                    
                        
                        
                        cam2ego_r = Quaternion(matrix=R) 
                        cam2ego_r = np.array(list(cam2ego_r), dtype=np.float64)
                        
                        info['cams'][camera_type]['sensor2ego_rotation'] = cam2ego_r
                        info['cams'][camera_type]['sensor2ego_translation'] = cam2ego_t.reshape(3)

                        # 5) intrinsics directly from YAML
                        info['cams'][camera_type]['cam_intrinsic'] = K

                        ##################################################################

                    # Step 2.2: build inf sensor info
                    # datasets/~ 부터 저장 버전
                    # pointcloud_path = osp.join(r, f"{token}.bin")
                    
                    # train/~ 부터 저장 버전
                    # key = f"{split_part}/"
                    # r2 = r[r.index(key):] 
                    r2 = Path(i_path).relative_to(Path(root_path_base) / "data").as_posix()
                    pointcloud_path = osp.join(r2, f"{token}.bin")
                    # breakpoint()
                    
                    other_agent_info['lidar_path'] = pointcloud_path
                    other_agent_info['lidar2ego_rotation'] = inf_lidar_ego_global_infos[mapping_token]['lidar2ego_rotation']
                    other_agent_info['lidar2ego_translation'] = inf_lidar_ego_global_infos[mapping_token]['lidar2ego_translation']
                    other_agent_info['ego2global_rotation'] = inf_lidar_ego_global_infos[mapping_token]['ego2global_rotation']
                    other_agent_info['ego2global_translation'] = inf_lidar_ego_global_infos[mapping_token]['ego2global_translation']
                    
                    
                    other_agent_info['cams'] = {}
                        
                    for camera_type in ['cam1', 'cam2']:        # NOTE:  V2X-real에서는 inf camera가 2대...
                        # datasets/~ 부터 저장 버전
                        # cam_image_path = osp.join(r, f"{token}_{camera_type}.jpeg")
                        
                        # train/~ 부터 저장버전
                        cam_image_path = osp.join(r2, f"{token}_{camera_type}.jpeg")
                        other_agent_info['cams'][camera_type] = {}
                        other_agent_info['cams'][camera_type]['data_path'] = cam_image_path             

                        annotation_path = osp.join(i_path, f"{token}.yaml")
                        anno_yaml_inf = load_yaml(annotation_path)          # dict로 로드됨
                        
                        anno_yaml_inf[camera_type]["cords"]
                        anno_yaml_inf[camera_type]["extrinsic"]
                        anno_yaml_inf[camera_type]["intrinsic"]

                        E = np.array(anno_yaml_inf[camera_type]["extrinsic"], dtype=np.float64)   # 4x4
                        K = np.array(anno_yaml_inf[camera_type]["intrinsic"], dtype=np.float64)   # 3x3

                        R_cam2lidar = E[:3, :3]                 # cam -> lidar
                        t_cam2lidar = E[:3, 3].reshape(3,)      # cam -> lidar translation

                        # 2) sensor2lidar (camera -> lidar): use as-is from YAML
                        other_agent_info['cams'][camera_type]['sensor2lidar_rotation'] = R_cam2lidar
                        other_agent_info['cams'][camera_type]['sensor2lidar_translation'] = t_cam2lidar

                        # 3) lidar2cam (lidar -> camera): inverse of (cam -> lidar)
                        # inv([R, t]) = [R^T, -R^T t]
                        R_lidar2cam = R_cam2lidar.T
                        t_lidar2cam = (-R_cam2lidar.T @ t_cam2lidar.reshape(3, 1)).reshape(3,)

                        other_agent_info['cams'][camera_type]['lidar2cam_rotation'] = R_lidar2cam
                        other_agent_info['cams'][camera_type]['lidar2cam_translation'] = t_lidar2cam
                        
                        cam2ego_r, cam2ego_t = mul_matrix(
                            R_cam2lidar, t_cam2lidar,
                            Quaternion(other_agent_info['lidar2ego_rotation']).rotation_matrix,
                            np.array(other_agent_info['lidar2ego_translation'], dtype=np.float64)
                        )

                        # NOTE: 260204 추가 - GW
                        R = np.array(cam2ego_r)               # (3,3)
                        cam2ego_r = Quaternion(matrix=R) 
                        cam2ego_r = np.array(list(cam2ego_r), dtype=np.float64)
                        
                        other_agent_info['cams'][camera_type]['sensor2ego_rotation'] = cam2ego_r
                        other_agent_info['cams'][camera_type]['sensor2ego_translation'] = cam2ego_t.reshape(3)

                        # 5) intrinsics directly from YAML
                        other_agent_info['cams'][camera_type]['cam_intrinsic'] = K

                        ##################################################################

                    # UniV2X TODO: complete this part
                    info['sweeps'] = {}
                    info['can_bus'] = np.zeros(18)

                    other_agent_info['sweeps'] = {}
                    other_agent_info['can_bus'] = np.zeros(18)
                    
                    # NOTE: infra 쪽 json 만들때 필요한 정보들 모음 (box 관련은 필요없음 사용안함.)
                    infra_info = {
                        'token': sample_info['token'],
                        'frame_idx': sample_info['frame_idx'],
                        'scene_token': sample_info['scene_token'],
                        'location': sample_info['location'],
                        'timestamp': sample_info['timestamp'],
                        'prev': sample_info['prev'],
                        'next': sample_info['next']
                    }
                    infra_info['lidar_path'] = other_agent_info['lidar_path']
                    infra_info['lidar2ego_rotation'] = other_agent_info['lidar2ego_rotation']
                    infra_info['lidar2ego_translation'] = other_agent_info['lidar2ego_translation']
                    infra_info['ego2global_rotation'] = other_agent_info['ego2global_rotation']
                    infra_info['ego2global_translation'] = other_agent_info['ego2global_translation']
                    infra_info['cams'] = other_agent_info['cams']
                    infra_info['sweeps'] = other_agent_info['sweeps']
                    infra_info['can_bus'] = other_agent_info['can_bus']
                    
                    
                    ## Step 3: build annotation information
                    # breakpoint()
                    annotations = total_annotations[mapping_token]
                    boxes = []
                    # breakpoint()
                    
                    ## NOTE: Box들을 ego 좌표계 중심으로 변경
                    box_to_ego_coord = True
                    if box_to_ego_coord:
                        true_ego_pose = anno_yaml_veh.get("true_ego_pose", None)
                        if true_ego_pose is None:
                            raise ValueError(f"[{mapping_token}] true_ego_pose missing in yaml: {annotation_path}")
                        
                        T_ego_to_world = x_to_world(true_ego_pose)

                        for anno_token in annotations.keys():
                            annotation = annotations[anno_token]

                            # (A) global location -> ego location
                            p_g = np.array([
                                annotation['3d_location']['x'],
                                annotation['3d_location']['y'],
                                annotation['3d_location']['z']
                            ], dtype=np.float64)

                            T_obj_to_world = x_to_world(np.concatenate([p_g, 
                                                                        np.array(annotation.get('angle', np.zeros(3)))]))

                            T_obj_to_ego = np.linalg.inv(T_ego_to_world) @ T_obj_to_world
                            
                            box3d = Box3D()
                            box3d.center = T_obj_to_ego[:3, 3].tolist()
                            box3d.wlh = [
                                annotation['3d_dimensions']['w'],
                                annotation['3d_dimensions']['l'],
                                annotation['3d_dimensions']['h']
                            ]

                            # ⚠️ 핵심: 여기는 float(rad)로 넣어야 아래 rots reshape(-1,1)이 정상 동작
                            box3d.orientation_yaw_pitch_roll = np.arctan2(T_obj_to_ego[1, 0], T_obj_to_ego[0, 0])

                            box3d.name = annotation['type']
                            box3d.token = annotation['token']
                            box3d.instance_token = annotation['instance_token']
                            box3d.track_id = int(annotation['track_id'])
                            box3d.timestamp = float(sample_info['timestamp'])
                            box3d.gt_velocity = annotation['gt_velocity']
                            box3d.prev = annotation['prev']
                            box3d.next = annotation['next']
                            box3d.visibility = visibility_mappings[annotation['occluded_state']]
                            boxes.append(box3d)
                                    
                    else:
                        raise ValueError("Must Convert to ego coordinate!!")
                        for anno_token in annotations.keys():
                            annotation = annotations[anno_token]
                            box3d = Box3D()
                            box3d.center = [annotation['3d_location']['x'], annotation['3d_location']['y'],
                                            annotation['3d_location']['z']]
                            box3d.wlh = [annotation['3d_dimensions']['w'], annotation['3d_dimensions']['l'],
                                            annotation['3d_dimensions']['h']]
                            box3d.orientation_yaw_pitch_roll = annotation['rotation']
                            box3d.name = annotation['type']
                            box3d.token = annotation['token']
                            box3d.instance_token = annotation['instance_token']
                            box3d.track_id = int(annotation['track_id'])
                            box3d.timestamp = float(sample_info['timestamp'])
                            box3d.visibility = visibility_mappings[annotation['occluded_state']] # NOTE: V2X-REAL에서는 이거 없어서 날림
                            box3d.gt_velocity = annotation['gt_velocity']
                            box3d.prev = annotation['prev']
                            box3d.next = annotation['next']

                            boxes.append(box3d)
                    # breakpoint()
                    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
                    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
                    rots = np.array([b.orientation_yaw_pitch_roll
                                        for b in boxes]).reshape(-1, 1)

                    gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
                    names = np.array([b.name for b in boxes])
                    instance_tokens = np.array([b.instance_token for b in boxes])
                    instance_inds = np.array([b.track_id for b in boxes])
                    box_tokens = np.array([b.token for b in boxes])
                    timestamps = np.array([b.timestamp for b in boxes])
                    visibility_tokens = np.array([b.visibility for b in boxes])
                    gt_velocity = np.array([b.gt_velocity for b in boxes])
                    prev_anno_tokens = np.array([b.prev for b in boxes])
                    next_anno_tokens = np.array([b.next for b in boxes])

                    # TODO: complete this part
                    valid_flag = np.array([True for b in boxes])
                    num_lidar_pts = np.array([1 for b in boxes])
                    # breakpoint()
                    info['gt_boxes'] = gt_boxes
                    info['gt_names'] = names
                    info['gt_ins_tokens'] = instance_tokens
                    info['gt_inds'] = instance_inds
                    info['anno_tokens'] = box_tokens
                    info['valid_flag'] = valid_flag
                    info['num_lidar_pts'] = num_lidar_pts
                    info['timestamps'] = timestamps
                    info['visibility_token'] = visibility_tokens
                    info['gt_velocity'] = gt_velocity
                    info['prev_anno_tokens'] = prev_anno_tokens
                    info['next_anno_tokens'] = next_anno_tokens

                    ## Step X: save spd infos   # NOTE: 이건 굳이 train val 나눌 필요 없음.
                    # if data_info['sequence_id'] in train_scenes:
                    #     train_spd_infos.append(info)
                    # elif data_info['sequence_id'] in val_scenes:
                    #     val_spd_infos.append(info)
                    info['other_agent_info_dict']['model_other_agent_inf'] = other_agent_info
                    if split_part == "train":
                        train_spd_infos.append(info)
                    elif split_part == "val":
                        val_spd_infos.append(info)
                    elif split_part == "test":
                        test_spd_infos.append(info)
                    elif split_part == "debug":
                        debug_spd_infos.append(info)
                    else:
                        raise ValueError("Not correct splitpart")
                    spd_infos.append(info)
                    
                    spd_infos_infra.append(infra_info)

        print("skip_sample_number in create_spd_infos_coop: ", skip_sample_number)

        keep_mapping_tokens_dict[split_part] = keep_mapping_tokens

        if flag_save:
            metadata = dict(version=version)
            if split_part == "train":
                data = dict(infos=train_spd_infos, metadata=metadata)
            elif split_part == "val":
                data = dict(infos=val_spd_infos, metadata=metadata)
            elif split_part == "test":
                data = dict(infos=test_spd_infos, metadata=metadata)
            elif split_part == "debug":
                data = dict(infos=debug_spd_infos, metadata=metadata)
            else: 
                raise ValueError("Not correct splitpart")
            info_path = osp.join(out_path,
                                    f'{info_prefix}_infos_temporal_{split_part}.pkl')
            mmcv.dump(data, info_path)

    if split_path is not None and len(str(split_path)) > 0:
        build_split_json_from_keep_mapping_tokens(
            keep_mapping_tokens_dict=keep_mapping_tokens_dict,
            out_json_path=split_path,   # 기존 split-file 경로 그대로 재활용 가능
        )

    return total_annotations, sample_info_mappings, spd_infos, spd_infos_infra

def get_lidar_ego_global_infos_coop(root_path, v2x_side, lidar_ego_global_infos, skip_noinfra, keep_mapping_tokens):
    root_path = Path(root_path)
    run_paths = sorted([p for p in root_path.iterdir() if p.is_dir()])
    skip_sample_number = 0
    for run_path in run_paths:
        scene_token = run_path.name
        
        if skip_noinfra:
            infra_possible_1 = run_path / "-1"
            infra_possible_2 = run_path / "-2"
            if (not infra_possible_1.is_dir()) and (not infra_possible_2.is_dir()):
                continue
            
        # if v2x_side == "vehicle-side":
        #     p = run_path / "1"
        #     if not p.is_dir():
        #         p = run_path / "2"
                
        # elif v2x_side == "infrastructure-side":
    #         p = run_path / "-1"
    #         if not p.is_dir():
    #             p = run_path / "-2" 

                    
        # elif v2x_side == "cooperative": # NOTE: Cooperative 에서는 차량 중심으로 함
        #     p = run_path / "1"
        #     if not p.is_dir():
        #         p = run_path / "2"    
        # else:
        #     raise ValueError("Wrong Type of v2x_side")
        
        
        ## NOTE: 기원 수정중 ======================= ##
        p1 = run_path / "1"
        p2 = run_path / "2"
        r1 = run_path / "-1"
        r2 = run_path / "-2"

        p_list = [(name, path) for name, path in [("1", p1), ("2", p2), ("-1", r1), ("-2", r2)] if path.is_dir()]
        veh_list  = [(n, p) for (n, p) in p_list if not n.startswith("-")]   # "1","2"
        infra_list = [(n, p) for (n, p) in p_list if n.startswith("-")]      # "-1","-2"
        pairs = [((v_name, v_path), (i_name, i_path))
                for (v_name, v_path) in veh_list
                for (i_name, i_path) in infra_list]
        pairs_sample = [ (f"{v_name}_{i_name}", v_path, i_path) for ((v_name,v_path),(i_name,i_path)) in pairs ]

        name, path = p_list[0]
            
        # 한 시나리오에 +1, +2, -1, -2 모두 같은 frame 수만큼 들어있는 것으로 이해하고 코드 작성.
        tokens = sorted({
            f.name[:6]
            for f in path.iterdir()
            if f.is_file() and f.name[:6].isdigit()
        })
        
        for pair_name, v_path, i_path in pairs_sample:
            for token in tokens:
                # TODO: 아래로 수정해야함
                ## Step 1: build basic information
                if v2x_side == "vehicle-side":
                    p = v_path
                elif v2x_side == "infrastructure-side":
                    p = i_path
                elif v2x_side == "cooperative": # NOTE: Cooperative 에서는 차량 중심으로 함
                    p = v_path
                else:
                    raise ValueError("Wrong Type of v2x_side")

                # breakpoint()
                    
                sample_token = token
                # mapping_token = scene_token + "_" + sample_token
                # breakpoint()
                mapping_token = scene_token + "_folder_" + pair_name + "_" + sample_token + "_folder_" + pair_name
                
                if mapping_token not in keep_mapping_tokens:
                    # print("22passed mapping_token: ", mapping_token)
                    # print("")
                    skip_sample_number += 1
                    continue
                # breakpoint()
                # new_scene_token = scene_token + "_folder_" + pair_name
                lidar_ego_global_infos[mapping_token] = {}

                annotation_path = osp.join(p, f"{token}.yaml")
                anno_yaml = load_yaml(annotation_path)

                # lidar == ego (V2X-REAL 가정 유지)
                lidar_ego_global_infos[mapping_token]['lidar2ego_rotation'] = np.array(
                    list(Quaternion(matrix=np.eye(3, dtype=np.float64))), dtype=np.float64
                )
                lidar_ego_global_infos[mapping_token]['lidar2ego_translation'] = np.array([0.0, 0.0, 0.0], dtype=np.float64)

                true_ego_pose = anno_yaml.get("true_ego_pose", None)
                if true_ego_pose is None:
                    raise ValueError(f"[{mapping_token}] true_ego_pose missing in yaml: {annotation_path}")
                
                #### NOTE 수정 by hm
                T_ego_to_world = x_to_world(true_ego_pose)          # (4,4), ego->global
                R_ego_to_world = T_ego_to_world[:3, :3].astype(np.float64)
                t_ego_to_world = T_ego_to_world[:3, 3].astype(np.float64)

                q_eg = Quaternion(matrix=R_ego_to_world)            # ego->global

                lidar_ego_global_infos[mapping_token]['ego2global_translation'] = t_ego_to_world.reshape(3)
                lidar_ego_global_infos[mapping_token]['ego2global_rotation'] = np.array(list(q_eg), dtype=np.float64)
    print("skip_sample_number in get_lidar_ego_global_infos_coop: ", skip_sample_number)
    return lidar_ego_global_infos


def get_lidar_ego_global_infos(root_path, sample_infos, v2x_side, lidar_ego_global_infos, skip_noinfra, infra_set_use_minus1):
    root_path = Path(root_path)
    run_paths = sorted([p for p in root_path.iterdir() if p.is_dir()])
   
    for run_path in run_paths:
        scene_token = run_path.name
        
        if skip_noinfra:
            infra_possible_1 = run_path / "-1"
            infra_possible_2 = run_path / "-2"
            if (not infra_possible_1.is_dir()) and (not infra_possible_2.is_dir()):
                continue
            
        if v2x_side == "vehicle-side":
            p = run_path / "1"
            if not p.is_dir():
                p = run_path / "2"
        elif v2x_side == "infrastructure-side":
            if infra_set_use_minus1:
                p = run_path / "-1"
                if not p.is_dir():
                    p = run_path / "-2" 
            else:
                p = run_path / "-2"
                if not p.is_dir():
                    p = run_path / "-1" 
        elif v2x_side == "cooperative": # NOTE: Cooperative 에서는 차량 중심으로 함
            p = run_path / "1"
            if not p.is_dir():
                p = run_path / "2"    
        else:
            raise ValueError("Wrong Type of v2x_side")
        
        tokens = sorted({
            f.name[:6]
            for f in p.iterdir()
            if f.is_file() and f.name[:6].isdigit()
        })
        for token in tokens:
            ## Step 1: build basic information
            sample_token = token
            mapping_token = scene_token + "_" + sample_token
            lidar_ego_global_infos[mapping_token] = {}

            annotation_path = osp.join(p, f"{token}.yaml")
            anno_yaml = load_yaml(annotation_path)

            # lidar == ego (V2X-REAL 가정 유지)
            lidar_ego_global_infos[mapping_token]['lidar2ego_rotation'] = np.array(
                list(Quaternion(matrix=np.eye(3, dtype=np.float64))), dtype=np.float64
            )
            lidar_ego_global_infos[mapping_token]['lidar2ego_translation'] = np.array([0.0, 0.0, 0.0], dtype=np.float64)

            true_ego_pose = anno_yaml.get("true_ego_pose", None)
            if true_ego_pose is None:
                raise ValueError(f"[{mapping_token}] true_ego_pose missing in yaml: {annotation_path}")
            
            #### NOTE 수정 by hm
            T_ego_to_world = x_to_world(true_ego_pose)          # (4,4), ego->global
            R_ego_to_world = T_ego_to_world[:3, :3].astype(np.float64)
            t_ego_to_world = T_ego_to_world[:3, 3].astype(np.float64)

            q_eg = Quaternion(matrix=R_ego_to_world)            # ego->global

            lidar_ego_global_infos[mapping_token]['ego2global_translation'] = t_ego_to_world.reshape(3)
            lidar_ego_global_infos[mapping_token]['ego2global_rotation'] = np.array(list(q_eg), dtype=np.float64)

    return lidar_ego_global_infos

def cal_ego_velocity(data_infos,sample_info_mappings,lidar_ego_global_infos):
    #{'sample_token': [vx,xy]}
    ego_velocity = {}
    for data_info in tqdm(data_infos):
        sample_token = data_info['frame_id']
        cur_loc = lidar_ego_global_infos[sample_token]['ego2global_translation']
        cur_timestamp = float(sample_info_mappings[sample_token]['timestamp']) / 1e6

        next_sample_token = sample_info_mappings[sample_token]['next']
        if next_sample_token != '':
            next_loc = lidar_ego_global_infos[next_sample_token]['ego2global_translation']
            next_timestamp = float(sample_info_mappings[next_sample_token]['timestamp']) / 1e6
            ego_velocity[sample_token] = (next_loc - cur_loc) / (next_timestamp - cur_timestamp)
        else:
            ego_velocity[sample_token] = [0,0]
    
    return ego_velocity

def rpy_from_true_pose(true_ego_pose, order="pitch_yaw_roll", degrees=True):
    pose = np.array(true_ego_pose, dtype=np.float64).reshape(-1)
    t = pose[:3]
    a3, a4, a5 = pose[3], pose[4], pose[5]

    if order == "roll_yaw_pitch":
        roll, yaw, pitch = a3, a4, a5
    elif order == "roll_pitch_yaw":
        roll, pitch, yaw = a3, a4, a5
    elif order == "pitch_yaw_roll":
        pitch, yaw, roll = a3, a4, a5
    elif order == "pitch_roll_yaw":
        pitch, roll, yaw = a3, a4, a5
    elif order == "yaw_roll_pitch":
        yaw, roll, pitch = a3, a4, a5
    elif order == "yaw_pitch_roll":
        yaw, pitch, roll = a3, a4, a5
    else:
        raise ValueError(order)

    if degrees:
        roll  = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw   = np.deg2rad(yaw)

    return t, roll, pitch, yaw

def R_from_rpy(roll, pitch, yaw):
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr,  cr]], dtype=np.float64)

    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]], dtype=np.float64)

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]], dtype=np.float64)

    return Rz @ Ry @ Rx   # ego->global

def T_ge_from_true_pose(true_ego_pose, order="pitch_yaw_roll", degrees=True):
    t_eg, roll, pitch, yaw = rpy_from_true_pose(true_ego_pose, order=order, degrees=degrees)
    R_eg = R_from_rpy(roll, pitch, yaw)     # ego->global

    T_ge = np.eye(4, dtype=np.float64)
    T_ge[:3, :3] = R_eg.T
    T_ge[:3,  3] = -R_eg.T @ t_eg
    return T_ge


### NOTE: Ego 좌표계로 보내는 코드 ================================= ###
def q_ego_to_global_from_true_pose(true_ego_pose, order="pitch_yaw_roll", degrees=True):
    """
    true_ego_pose: [x, y, z, a3, a4, a5]

    order 의미: a3,a4,a5가 어떤 순서로 (roll,pitch,yaw)에 대응되는지.
    지원:
      - roll_pitch_yaw
      - roll_yaw_pitch
      - pitch_roll_yaw
      - pitch_yaw_roll
      - yaw_roll_pitch
      - yaw_pitch_roll

    return:
      t_eg: (3,)
      q_eg: Quaternion (ego -> global), composed as q = Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    pose = np.array(true_ego_pose, dtype=np.float64).reshape(-1)
    t_eg = pose[:3]
    a3, a4, a5 = pose[3], pose[4], pose[5]

    if order == "roll_yaw_pitch":
        roll, yaw, pitch = a3, a4, a5
    elif order == "roll_pitch_yaw":
        roll, pitch, yaw = a3, a4, a5
    elif order == "pitch_yaw_roll":
        pitch, yaw, roll = a3, a4, a5
    elif order == "pitch_roll_yaw":
        pitch, roll, yaw = a3, a4, a5
    elif order == "yaw_roll_pitch":
        yaw, roll, pitch = a3, a4, a5
    elif order == "yaw_pitch_roll":
        yaw, pitch, roll = a3, a4, a5
    else:
        raise ValueError(
            f"Unknown order: {order}. "
            "Use one of: roll_yaw_pitch, roll_pitch_yaw, pitch_yaw_roll, "
            "pitch_roll_yaw, yaw_roll_pitch, yaw_pitch_roll"
        )

    if degrees:
        roll  = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw   = np.deg2rad(yaw)

    # ego->global: q = Rz(yaw) * Ry(pitch) * Rx(roll)
    qx = Quaternion(axis=[1, 0, 0], angle=roll)
    qy = Quaternion(axis=[0, 1, 0], angle=pitch)
    qz = Quaternion(axis=[0, 0, 1], angle=yaw)
    q_eg = qz * qy * qx
    return t_eg, q_eg


def global_point_to_ego(p_g, t_eg, q_eg):
    """p_e = R_eg^{-1} (p_g - t_eg)"""
    p_g = np.asarray(p_g, dtype=np.float64)
    return q_eg.inverse.rotate(p_g - t_eg)

def make_T_eg(q_eg: Quaternion, t_eg):
    """
    T_eg: ego -> global (4x4)
    [ R  t ]
    [ 0  1 ]
    """
    t_eg = np.asarray(t_eg, dtype=np.float64).reshape(3)
    R_eg = q_eg.rotation_matrix  # 3x3
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_eg
    T[:3,  3] = t_eg
    return T

def global_point_to_ego_mtx(p_g, t_eg, q_eg):
    """
    global point p_g (3,) -> ego point p_e (3,) using 4x4 matrix.
    Equivalent to: q_eg.inverse.rotate(p_g - t_eg)
    """
    p_g = np.asarray(p_g, dtype=np.float64).reshape(3)
    T_eg = make_T_eg(q_eg, t_eg)       # ego->global
    T_ge = np.linalg.inv(T_eg)         # global->ego

    p_g_h = np.ones(4, dtype=np.float64)
    p_g_h[:3] = p_g

    p_e_h = T_ge @ p_g_h
    return p_e_h[:3]

def yaw_from_quat(q: Quaternion):
    """Quaternion -> yaw (rad), assuming z-up."""
    R = q.rotation_matrix
    return np.arctan2(R[1,0], R[0,0])

def yaw_from_quat_flat(q: Quaternion):
    """roll/pitch가 섞여도 안정적으로 'z-up yaw'만 뽑기 (rad)"""
    Rm = q.rotation_matrix
    fwd = Rm @ np.array([1.0, 0.0, 0.0])  # box local +x (forward) in ego
    fwd[2] = 0.0                          # flatten to xy-plane
    n = np.linalg.norm(fwd)
    if n < 1e-9:
        return 0.0
    fwd /= n
    return np.arctan2(fwd[1], fwd[0])
### ======================================================= ###

def create_spd_infos(root_path_base,
                     out_path_base,
                     v2x_side,
                     split_path,
                     can_bus_root_path,
                     info_prefix,
                     version='v1.0-trainval',
                     max_sweeps=10,
                     flag_save=True,
                     skip_noinfra=True,
                     infra_set_use_minus1=True):
    """Create info file of spd dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'vehicle-side'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    
    if split_path is not None and len(str(split_path)) > 0:
        _build_and_save_v2xreal_split_json(
            root_path_base=root_path_base,
            split_json_path=split_path,
            splits=("train", "val", "test"),
            skip_noinfra=skip_noinfra
        )
    
    spd_infos = []
    train_spd_infos = []
    val_spd_infos = []
    test_spd_infos = []
    sample_info_mappings = {}
    total_annotations = {}
    lidar_ego_global_infos = {}
    for split_part in ["train", "val", "test"]:
    # for split_part in ["debug_val"]:
        root_path = osp.join(root_path_base, f"data/{split_part}") 
        # root_path_val = osp.join(root_path, "/data/val")
        out_path = osp.join(out_path_base, v2x_side)
        ## Step 0: load neccesary data
        # data_info_path = osp.join(root_path, 'data_info.json')
        # split_data_path = split_path

        # data_infos = load_json(data_info_path)
        # split_data = load_json(split_data_path)
        # train_scenes = split_data['batch_split']['train']
        # val_scenes = split_data['batch_split']['val']


        ## Generate  sample_info_mappings, secene_frame_mappings, total_annotations, instance_token_mappings
        sample_infos, sample_info_mappings = _generate_sample_infos(root_path, sample_info_mappings, v2x_side, skip_noinfra, infra_set_use_minus1)
        secene_frame_mappings = _get_secene_frame_mappings(sample_info_mappings)
        # breakpoint()
        total_annotations = _get_total_annotations(root_path, sample_infos, sample_info_mappings, total_annotations, v2x_side, skip_noinfra, infra_set_use_minus1)
        # breakpoint()
        instance_token_mappings = _get_instance_token_mappings(total_annotations, sample_info_mappings)

        #get lidar2ego info
        lidar_ego_global_infos = get_lidar_ego_global_infos(root_path, sample_infos, v2x_side, lidar_ego_global_infos, skip_noinfra, infra_set_use_minus1)

        ## interpolate boxes for unvisible objects
        total_annotations = _generate_unvisible_annotations(v2x_side, sample_info_mappings, secene_frame_mappings, instance_token_mappings, total_annotations, lidar_ego_global_infos)

        ##update instance_token_mappings
        instance_token_mappings = _get_instance_token_mappings(total_annotations, sample_info_mappings)

        # #cal ego_velocity
        # ego_velocity = cal_ego_velocity(data_infos,sample_info_mappings,lidar_ego_global_infos)

        ## add velocity and prev/next, update total_annotations and instance_token_mappings
        total_annotations, instance_token_mappings = _add_annotation_velocity_prev_next(total_annotations, instance_token_mappings, lidar_ego_global_infos)
        # breakpoint()
        #gen_infos
            
        root_path = Path(root_path)
        run_paths = sorted([p for p in root_path.iterdir() if p.is_dir()])
        for run_path in tqdm(run_paths):
            scene_token = run_path.name
            
            if skip_noinfra:
                infra_possible_1 = run_path / "-1"
                infra_possible_2 = run_path / "-2"
                if (not infra_possible_1.is_dir()) and (not infra_possible_2.is_dir()):
                    continue
                
            # prefer 1, fallback to 2   # TODO: 1번 차량이 있으면 1번을 기본적으로 사용. 없으면 2번 차량을 사용
            if v2x_side == "vehicle-side":
                p = run_path / "1"
                if not p.is_dir():
                    p = run_path / "2"
            elif v2x_side == "infrastructure-side":
                if infra_set_use_minus1:
                    p = run_path / "-1"
                    if not p.is_dir():
                        p = run_path / "-2" 
                else:
                    p = run_path / "-2"
                    if not p.is_dir():
                        p = run_path / "-1" 
            else:
                raise ValueError("Wrong type of v2x-side")
            tokens = sorted({
                f.name[:6]
                for f in p.iterdir()
                if f.is_file() and f.name[:6].isdigit()
            })        

            for token in tokens:
                # sample_token = token
                mapping_token = scene_token + "_" + token

                # camera_type = 'VEHICLE_CAM_FRONT' # NOTE: 기존 nuscenes 에서는 이렇게 했음.
                
                ## Step 1: build basic information
                sample_info = sample_info_mappings[mapping_token]

                info = {
                    'token': sample_info['token'],
                    'frame_idx': sample_info['frame_idx'],
                    'scene_token': sample_info['scene_token'],
                    'location': sample_info['location'],
                    'timestamp': sample_info['timestamp'],
                    'prev': sample_info['prev'],
                    'next': sample_info['next'],
                }


                ## Step 2: build sensor data infomation
                # breakpoint()
                # datasets/~ 부터 저장 버전
                # pointcloud_path = osp.join(p, f"{token}.bin")
                
                # train/~ 부터 저장 버전
                # rel = osp.relpath(full_path, osp.join(root_path_base, "data"))
                # key = f"{split_part}/"
                # p2 = p[p.index(key):] 
                p2 = Path(p).relative_to(Path(root_path_base) / "data").as_posix()
                pointcloud_path = osp.join(p2, f"{token}.bin")
                # breakpoint()
                info['lidar_path'] = pointcloud_path
                info['lidar2ego_rotation'] = lidar_ego_global_infos[mapping_token]['lidar2ego_rotation']
                info['lidar2ego_translation'] = lidar_ego_global_infos[mapping_token]['lidar2ego_translation']
                info['ego2global_rotation'] = lidar_ego_global_infos[mapping_token]['ego2global_rotation']
                info['ego2global_translation'] = lidar_ego_global_infos[mapping_token]['ego2global_translation']
                
                info['cams'] = {}
                    
                if v2x_side == "vehicle-side":
                    cams_list = ['cam1', 'cam2', 'cam3', 'cam4']
                elif v2x_side == "infrastructure-side":
                    cams_list = ['cam1', 'cam2']
                else:
                    raise ValueError("Wrong Type of v2x-side")
                
                annotation_path = osp.join(p, f"{token}.yaml")
                anno_yaml = load_yaml(annotation_path)          # dict로 로드됨
                
                for camera_type in cams_list:        # NOTE:  V2X-real에서는 camera가 4대...
                    # datasets/~ 부터 저장 버전
                    # cam_image_path = osp.join(p, f"{token}_{camera_type}.jpeg")
                    
                    # train/~ 부터 저장버전
                    cam_image_path = osp.join(p2, f"{token}_{camera_type}.jpeg")
                    info['cams'][camera_type] = {}
                    info['cams'][camera_type]['data_path'] = cam_image_path           

                    
                    anno_yaml[camera_type]["cords"]
                    anno_yaml[camera_type]["extrinsic"]
                    anno_yaml[camera_type]["intrinsic"]


                    ########## TODO: 아래 calibration이 맞는지 확인이 필요함. ###########
                    E = np.array(anno_yaml[camera_type]["extrinsic"], dtype=np.float64)   # 4x4
                    K = np.array(anno_yaml[camera_type]["intrinsic"], dtype=np.float64)   # 3x3

                    R_cam2lidar = E[:3, :3]                 # cam -> lidar
                    t_cam2lidar = E[:3, 3].reshape(3,)      # cam -> lidar translation

                    # 2) sensor2lidar (camera -> lidar): use as-is from YAML
                    info['cams'][camera_type]['sensor2lidar_rotation'] = R_cam2lidar
                    info['cams'][camera_type]['sensor2lidar_translation'] = t_cam2lidar

                    # 3) lidar2cam (lidar -> camera): inverse of (cam -> lidar)
                    # inv([R, t]) = [R^T, -R^T t]
                    R_lidar2cam = R_cam2lidar.T
                    t_lidar2cam = (-R_cam2lidar.T @ t_cam2lidar.reshape(3, 1)).reshape(3,)
                    

                    ### ======================================================= ###


                    info['cams'][camera_type]['lidar2cam_rotation'] = R_lidar2cam
                    info['cams'][camera_type]['lidar2cam_translation'] = t_lidar2cam

                    ##################################################################
                    ##################### TODO: 이 아래도 확인 바람. ######################
                    # cam2ego_r, cam2ego_t = mul_matrix(cam2lidar_r, cam2lidar_t,
                    #                                 Quaternion(info['lidar2ego_rotation']).rotation_matrix,
                    #                                 np.array(info['lidar2ego_translation']))
                    # info['cams'][camera_type]['sensor2ego_rotation'] = cam2ego_r
                    # info['cams'][camera_type]['sensor2ego_translation'] = cam2ego_t.reshape(3)        

                    # calib_cam_intrinsic_path = osp.join(root_path, data_info['calib_camera_intrinsic_path'])
                    # calib_cam_intrinsic = get_cam_intr(calib_cam_intrinsic_path)
                    # info['cams'][camera_type]['cam_intrinsic'] = calib_cam_intrinsic
                    
                    cam2ego_r, cam2ego_t = mul_matrix(
                        R_cam2lidar, t_cam2lidar,
                        Quaternion(info['lidar2ego_rotation']).rotation_matrix,
                        np.array(info['lidar2ego_translation'], dtype=np.float64)
                    )
                    # breakpoint()
                    # NOTE: 260204 추가 - GW
                    R = np.array(cam2ego_r)               # (3,3)
                    cam2ego_r = Quaternion(matrix=R) 
                    cam2ego_r = np.array(list(cam2ego_r), dtype=np.float64)
                    info['cams'][camera_type]['sensor2ego_rotation'] = cam2ego_r
                    info['cams'][camera_type]['sensor2ego_translation'] = cam2ego_t.reshape(3)

                    # 5) intrinsics directly from YAML
                    info['cams'][camera_type]['cam_intrinsic'] = K

                    ##################################################################

                info['sweeps'] = {}
                # TODO: ego speed도 넣어줄 수 있는데, 일단은 canbus에 zero value들 넣어두었음. 
                info['can_bus'] = np.zeros(18)
                # TODO: anno_yaml["ego_speed"] 로 넣어줄 수 있음. 추후 맞춰서 넣어주자
                
                
                ## Step 3: build annotation information
                annotations = total_annotations[mapping_token]
                boxes = []
                # breakpoint()
                
                ## NOTE: Box들을 ego 좌표계 중심으로 변경
                ### NOTE: 250114 ==================== ####
                box_to_ego_coord = True
                # box_to_ego_coord = False 
                #### ================================ ####
                if box_to_ego_coord:
                    true_ego_pose = anno_yaml.get("true_ego_pose", None)
                    if true_ego_pose is None:
                        raise ValueError(f"[{mapping_token}] true_ego_pose missing in yaml: {annotation_path}")

                    t_eg, q_eg = q_ego_to_global_from_true_pose(
                        true_ego_pose,
                        order="roll_yaw_pitch",  # 네 가정 유지
                        degrees=True
                    )
                    
                    T_ego_to_world = x_to_world(true_ego_pose)
                    # T_ge = np.linalg.inv(make_T_eg(q_eg, t_eg))
                    
                    # T_ge = T_ge_from_true_pose(true_ego_pose, order="pitch_yaw_roll", degrees=True)

                    for anno_token in annotations.keys():
                        annotation = annotations[anno_token]

                        # import pdb; pdb.set_trace()
                        # (A) global location -> ego location
                        p_g = np.array([
                            annotation['3d_location']['x'],
                            annotation['3d_location']['y'],
                            annotation['3d_location']['z']
                        ], dtype=np.float64)
                        
                        T_obj_to_world = x_to_world(np.concatenate([p_g, 
                                                                    np.array(annotation.get('angle', np.zeros(3)))]))
                        

                        T_obj_to_ego = np.linalg.inv(T_ego_to_world) @ T_obj_to_world
                        
                        box3d = Box3D()
                        box3d.center = T_obj_to_ego[:3, 3].tolist()
                        box3d.wlh = [
                            annotation['3d_dimensions']['w'],
                            annotation['3d_dimensions']['l'],
                            annotation['3d_dimensions']['h']
                        ]

                        # ⚠️ 핵심: 여기는 float(rad)로 넣어야 아래 rots reshape(-1,1)이 정상 동작
                        box3d.orientation_yaw_pitch_roll = np.arctan2(T_obj_to_ego[1, 0], T_obj_to_ego[0, 0])

                        box3d.name = annotation['type']
                        box3d.token = annotation['token']
                        box3d.instance_token = annotation['instance_token']
                        box3d.track_id = int(annotation['track_id'])
                        box3d.timestamp = float(sample_info['timestamp'])
                        box3d.gt_velocity = annotation['gt_velocity']
                        box3d.prev = annotation['prev']
                        box3d.next = annotation['next']
                        box3d.visibility = visibility_mappings[annotation['occluded_state']]
                        boxes.append(box3d)
                                
                else:
                    for anno_token in annotations.keys():
                        annotation = annotations[anno_token]
                        box3d = Box3D()
                        box3d.center = [annotation['3d_location']['x'], annotation['3d_location']['y'],
                                        annotation['3d_location']['z']]
                        box3d.wlh = [annotation['3d_dimensions']['w'], annotation['3d_dimensions']['l'],
                                        annotation['3d_dimensions']['h']]
                        box3d.orientation_yaw_pitch_roll = annotation['rotation']
                        box3d.name = annotation['type']
                        box3d.token = annotation['token']
                        box3d.instance_token = annotation['instance_token']
                        box3d.track_id = int(annotation['track_id'])
                        box3d.timestamp = float(sample_info['timestamp'])
                        box3d.visibility = visibility_mappings[annotation['occluded_state']] # NOTE: V2X-REAL에서는 이거 없어서 날림
                        box3d.gt_velocity = annotation['gt_velocity']
                        box3d.prev = annotation['prev']
                        box3d.next = annotation['next']

                        boxes.append(box3d)
                # breakpoint()
                locs = np.array([b.center for b in boxes]).reshape(-1, 3)
                dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
                rots = np.array([b.orientation_yaw_pitch_roll
                                    for b in boxes]).reshape(-1, 1)

                gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
                names = np.array([b.name for b in boxes])
                instance_tokens = np.array([b.instance_token for b in boxes])
                instance_inds = np.array([b.track_id for b in boxes])
                box_tokens = np.array([b.token for b in boxes])
                timestamps = np.array([b.timestamp for b in boxes])
                visibility_tokens = np.array([b.visibility for b in boxes])
                gt_velocity = np.array([b.gt_velocity for b in boxes])
                prev_anno_tokens = np.array([b.prev for b in boxes])
                next_anno_tokens = np.array([b.next for b in boxes])

                # TODO: complete this part
                valid_flag = np.array([True for b in boxes])
                num_lidar_pts = np.array([1 for b in boxes])
                # breakpoint()
                info['gt_boxes'] = gt_boxes
                info['gt_names'] = names
                info['gt_ins_tokens'] = instance_tokens
                info['gt_inds'] = instance_inds
                info['anno_tokens'] = box_tokens
                info['valid_flag'] = valid_flag
                info['num_lidar_pts'] = num_lidar_pts
                info['timestamps'] = timestamps
                info['visibility_token'] = visibility_tokens
                info['gt_velocity'] = gt_velocity
                info['prev_anno_tokens'] = prev_anno_tokens
                info['next_anno_tokens'] = next_anno_tokens

                
                # if split_part == "test":  # TODO: 현재 수정중: for closed loop test set 
                #     future_gt_boxes = []
                #     future_gt_names = []
                #     future_gt_ins_tokens = []
                #     future_gt_inds = []
                #     future_anno_tokens = []
                #     future_valid_flag = []
                #     future_num_lidar_pts = []
                #     future_timestamps = []
                #     future_visibility_tokens = []
                #     future_gt_velocity = []
                #     future_prev_anno_tokens = []
                #     future_next_anno_tokens = []
                    
                #     for future_timesteps in range(1, 11):
                #         next_token = f"{int(token) + future_timesteps:06d}"
                #         mapping_token_next = scene_token + "_" + next_token
                #         annotation_path = osp.join(p, f"{next_token}.yaml")
                        
                #         if not osp.exists(annotation_path):
                #             gt_boxes = np.zeros((0, 7), dtype=np.float32)
                #             names = np.array([], dtype=np.str_)
                #             instance_tokens = np.array([], dtype=np.str_)
                #             instance_inds = np.array([], dtype=np.int64)
                #             box_tokens = np.array([], dtype=np.str_)
                #             timestamps = np.array([], dtype=np.float64)
                #             visibility_tokens = np.array([], dtype=np.int64)
                #             gt_velocity = np.zeros((0, 2), dtype=np.float32)
                #             prev_anno_tokens = np.array([], dtype=np.str_)
                #             next_anno_tokens = np.array([], dtype=np.str_)
                #             valid_flag = np.array([], dtype=bool)
                #             num_lidar_pts = np.array([], dtype=np.int64)
                        
                #         else:
                #             annotations = total_annotations[mapping_token_next]
                #             sample_info = sample_info_mappings[mapping_token_next]
                #             boxes = []
                #             anno_yaml = load_yaml(annotation_path)  
                            
                #             if box_to_ego_coord:
                #                 true_ego_pose = anno_yaml.get("true_ego_pose", None)
                #                 if true_ego_pose is None:
                #                     raise ValueError(f"[{mapping_token_next}] true_ego_pose missing in yaml: {annotation_path}")
                                
                #                 T_ego_to_world = x_to_world(true_ego_pose)

                #                 for anno_token in annotations.keys():
                #                     annotation = annotations[anno_token]

                #                     p_g = np.array([
                #                         annotation['3d_location']['x'],
                #                         annotation['3d_location']['y'],
                #                         annotation['3d_location']['z']
                #                     ], dtype=np.float64)
                                    
                #                     T_obj_to_world = x_to_world(np.concatenate([p_g, 
                #                                                                 np.array(annotation.get('angle', np.zeros(3)))]))
                                    

                #                     T_obj_to_ego = np.linalg.inv(T_ego_to_world) @ T_obj_to_world
                                    
                #                     box3d = Box3D()
                #                     box3d.center = T_obj_to_ego[:3, 3].tolist()
                #                     box3d.wlh = [
                #                         annotation['3d_dimensions']['w'],
                #                         annotation['3d_dimensions']['l'],
                #                         annotation['3d_dimensions']['h']
                #                     ]

                #                     box3d.orientation_yaw_pitch_roll = np.arctan2(T_obj_to_ego[1, 0], T_obj_to_ego[0, 0])

                #                     box3d.name = annotation['type']
                #                     box3d.token = annotation['token']
                #                     box3d.instance_token = annotation['instance_token']
                #                     box3d.track_id = int(annotation['track_id'])
                #                     box3d.timestamp = float(sample_info['timestamp'])
                #                     box3d.gt_velocity = annotation['gt_velocity']
                #                     box3d.prev = annotation['prev']
                #                     box3d.next = annotation['next']
                #                     box3d.visibility = visibility_mappings[annotation['occluded_state']]
                #                     boxes.append(box3d)
                            
                #             else:
                #                 raise NotImplementedError("현재는 box_to_ego_coord=True 일 때만 test future box 지원함.")
                            
                #             locs = np.array([b.center for b in boxes]).reshape(-1, 3)
                #             dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
                #             rots = np.array([b.orientation_yaw_pitch_roll
                #                                 for b in boxes]).reshape(-1, 1)

                #             gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
                #             names = np.array([b.name for b in boxes])
                #             instance_tokens = np.array([b.instance_token for b in boxes])
                #             instance_inds = np.array([b.track_id for b in boxes])
                #             box_tokens = np.array([b.token for b in boxes])
                #             timestamps = np.array([b.timestamp for b in boxes])
                #             visibility_tokens = np.array([b.visibility for b in boxes])
                #             gt_velocity = np.array([b.gt_velocity for b in boxes])
                #             prev_anno_tokens = np.array([b.prev for b in boxes])
                #             next_anno_tokens = np.array([b.next for b in boxes])

                #             # TODO: complete this part
                #             valid_flag = np.array([True for b in boxes])
                #             num_lidar_pts = np.array([1 for b in boxes])
                #             # breakpoint()
                        
                #         future_gt_boxes.append(gt_boxes)
                #         future_gt_names.append(names)
                #         future_gt_ins_tokens.append(instance_tokens)
                #         future_gt_inds.append(instance_inds)
                #         future_anno_tokens.append(box_tokens)
                #         future_valid_flag.append(valid_flag)
                #         future_num_lidar_pts.append(num_lidar_pts)
                #         future_timestamps.append(timestamps)
                #         future_visibility_tokens.append(visibility_tokens)
                #         future_gt_velocity.append(gt_velocity)
                #         future_prev_anno_tokens.append(prev_anno_tokens)
                #         future_next_anno_tokens.append(next_anno_tokens)
                    
                #     info['future_gt_boxes'] = future_gt_boxes
                #     info['future_gt_names'] = future_gt_names
                #     info['future_gt_ins_tokens'] = future_gt_ins_tokens
                #     info['future_gt_inds'] = future_gt_inds
                #     info['future_anno_tokens'] = future_anno_tokens
                #     info['future_valid_flag'] = future_valid_flag
                #     info['future_num_lidar_pts'] = future_num_lidar_pts
                #     info['future_timestamps'] = future_timestamps
                #     info['future_visibility_tokens'] = future_visibility_tokens
                #     info['future_gt_velocity'] = future_gt_velocity
                #     info['future_prev_anno_tokens'] = future_prev_anno_tokens
                #     info['future_next_anno_tokens'] = future_next_anno_tokens
                        
                 
                
                ## Step X: save spd infos   # NOTE: 이건 굳이 train val 나눌 필요 없음.
                # if data_info['sequence_id'] in train_scenes:
                #     train_spd_infos.append(info)
                # elif data_info['sequence_id'] in val_scenes:
                #     val_spd_infos.append(info)
                if split_part == "train":
                    train_spd_infos.append(info)
                elif split_part == "val":
                    val_spd_infos.append(info)
                elif split_part == "test":
                    test_spd_infos.append(info)
                else:
                    raise ValueError("Not correct splitpart")
                spd_infos.append(info)

                # if flag_save:
                #     metadata = dict(version=version)
                #     data = dict(infos=train_spd_infos, metadata=metadata)
                #     info_path = osp.join(out_path,
                #                             '{}_infos_temporal_train.pkl'.format(info_prefix))
                #     mmcv.dump(data, info_path)

                #     data['infos'] = val_spd_infos
                #     info_val_path = osp.join(out_path,
                #                                 '{}_infos_temporal_val.pkl'.format(info_prefix))
                #     mmcv.dump(data, info_val_path)
        if flag_save:
            metadata = dict(version=version)
            if split_part == "train":
                data = dict(infos=train_spd_infos, metadata=metadata)
            elif split_part == "val":
                data = dict(infos=val_spd_infos, metadata=metadata)
            elif split_part == "test":
                data = dict(infos=test_spd_infos, metadata=metadata)
            else: 
                raise ValueError("Not correct splitpart")
            info_path = osp.join(out_path,
                                    f'{info_prefix}_infos_temporal_{split_part}.pkl')
            mmcv.dump(data, info_path)

    return total_annotations, sample_info_mappings, spd_infos


def gen_token(*args):
    token_name = ''
    for value in args:
        token_name += str(value)
    token = uuid.uuid3(uuid.NAMESPACE_DNS, token_name)
    return str(token)


def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)

    return data


def write_json(data, path):
    with open(path, mode="w") as f:
        json.dump(data, f, indent=2)

def get_single_sample_info(frame_id, data_infos):
    sample_info = {}
    for data in data_infos:
        if data['frame_id'] == frame_id:
            sample_info = data
            break
    return sample_info

def _generate_sample_infos_coop(root_path, sample_info_mappings, skip_noinfra, keep_mapping_tokens):
    """Get the prev and next sample token for a given `sample_data_token`.
    Args:
        data_infos (list): data_infos loaded from data_info.json file.
    Return:
        list[dict]: List of sample info
        dict: mapping sample token to sample info   
    """
    
    root_path = Path(root_path)
    run_paths = sorted([k for k in root_path.iterdir() if k.is_dir()])


    
    veh_sample_mappings = {}
    inf_sample_mappings = {}  
    coop_sample_mappings = {}  
    scene_data_dict = {}
    skip_sample_number = 0
    for run_path in run_paths:
        scene_token = run_path.name
        # if scene_token not in scene_data_dict.keys():
        #     scene_data_dict[scene_token] = []
        
        if skip_noinfra:
            infra_possible_1 = run_path / "-1"
            infra_possible_2 = run_path / "-2"
            if (not infra_possible_1.is_dir()) and (not infra_possible_2.is_dir()):
                continue
        
        ## NOTE: 기원 수정중 ======================= ##
        p1 = run_path / "1"
        p2 = run_path / "2"
        r1 = run_path / "-1"
        r2 = run_path / "-2"

        p_list = [(name, path) for name, path in [("1", p1), ("2", p2), ("-1", r1), ("-2", r2)] if path.is_dir()]
        veh_list  = [(n, p) for (n, p) in p_list if not n.startswith("-")]   # "1","2"
        infra_list = [(n, p) for (n, p) in p_list if n.startswith("-")]      # "-1","-2"
        pairs = [((v_name, v_path), (i_name, i_path))
                for (v_name, v_path) in veh_list
                for (i_name, i_path) in infra_list]
        pairs_sample = [ (f"{v_name}_{i_name}", v_path, i_path) for ((v_name,v_path),(i_name,i_path)) in pairs ]

        name, path = p_list[0]
            
        # 한 시나리오에 +1, +2, -1, -2 모두 같은 frame 수만큼 들어있는 것으로 이해하고 코드 작성.
        tokens = sorted({
            f.name[:6]
            for f in path.iterdir()
            if f.is_file() and f.name[:6].isdigit()
        })
        
        
        for pair_name, v_path, i_path in pairs_sample:
            for token in tokens:
                sample_token = token      # TODO: 이렇게 해야 구분이 가능해짐
                new_scene_token = scene_token + "_folder_" + pair_name
                mapping_token = new_scene_token + "_" + sample_token + "_folder_" + pair_name
                if mapping_token not in keep_mapping_tokens:
                    # print("passed mapping_token: ", mapping_token)
                    # set에서 아무 원소 하나만 보기
                    # sample_keep = next(iter(keep_mapping_tokens), None)
                    # print("keep_mapping_tokens sample:", sample_keep)
                    skip_sample_number += 1
                    continue
                
                
                if new_scene_token not in scene_data_dict.keys():
                    scene_data_dict[new_scene_token] = []
                
                veh_data_info = {}
                veh_data_info['frame_id'] = mapping_token
                veh_data_info['pointcloud_timestamp'] = float(token) * 1e5
                veh_data_info['image_timestamp'] = float(token) * 1e5
                veh_data_info['sequence_id'] = new_scene_token
                veh_data_info['intersection_loc'] = "v2x_real_map"
                veh_sample_mappings[mapping_token] = veh_data_info

                inf_data_info = {}
                inf_data_info['frame_id'] = mapping_token
                inf_data_info['pointcloud_timestamp'] = float(token) * 1e5
                inf_data_info['image_timestamp'] = float(token) * 1e5
                inf_data_info['sequence_id'] = new_scene_token
                inf_data_info['intersection_loc'] = "v2x_real_map"
                inf_sample_mappings[mapping_token] = inf_data_info            
                
                scene_data_dict[new_scene_token].append(mapping_token)    
                    
    print("skip_sample_number in _generate_sample_infos_coop _ step1: ", skip_sample_number)
    skip_sample_number2 = 0
    sample_infos = []
    # sample_infos_infra = []
    for run_path in run_paths:
        scene_token = run_path.name
        
        if skip_noinfra:
            infra_possible_1 = run_path / "-1"
            infra_possible_2 = run_path / "-2"
            if (not infra_possible_1.is_dir()) and (not infra_possible_2.is_dir()):
                continue
        
        ## NOTE: 기원 수정중 ======================= ##
        p1 = run_path / "1"
        p2 = run_path / "2"
        r1 = run_path / "-1"
        r2 = run_path / "-2"

        p_list = [(name, path) for name, path in [("1", p1), ("2", p2), ("-1", r1), ("-2", r2)] if path.is_dir()]
        veh_list  = [(n, p) for (n, p) in p_list if not n.startswith("-")]   # "1","2"
        infra_list = [(n, p) for (n, p) in p_list if n.startswith("-")]      # "-1","-2"
        pairs = [((v_name, v_path), (i_name, i_path))
                for (v_name, v_path) in veh_list
                for (i_name, i_path) in infra_list]
        pairs_sample = [ (f"{v_name}_{i_name}", v_path, i_path) for ((v_name,v_path),(i_name,i_path)) in pairs ]
        
        
        for pair_name, v_path, i_path in pairs_sample:
            
            new_scene_token = scene_token + "_folder_" + pair_name
            
            seq = scene_data_dict.get(new_scene_token, [])
            if len(seq) == 0:
                skip_sample_number2 += 1
                continue
            
            scene_data_dict[new_scene_token].sort()     # 해당 시퀀스의 샘플들 있음 

            for idx in range(len(scene_data_dict[new_scene_token])):
                info = {}
                info_infra = {}
                if idx == 0:
                    info['prev'] = ''
                    # info_infra['prev'] = ''
                else:
                    info['prev'] = scene_data_dict[new_scene_token][idx - 1]
                    # info_infra['prev'] = scene_data_dict[new_scene_token][idx - 1]

                if idx == len(scene_data_dict[new_scene_token]) - 1:
                    info['next'] = ''
                    # info_infra['next'] = ''
                else:
                    info['next'] = scene_data_dict[new_scene_token][idx + 1]
                    # info_infra['next'] = scene_data_dict[new_scene_token][idx + 1]

                sample_token = scene_data_dict[new_scene_token][idx]
                # print("cur_sample_token in generate_sample_infos_coop: ", sample_token)
                veh_sample_info = veh_sample_mappings[sample_token]     # NOTE: Veh이든 infrastructure든 결국 같음. 그냥 이 sample info를 들고옴 
                info['token'] = veh_sample_info['frame_id']
                info['timestamp'] = float(veh_sample_info['pointcloud_timestamp'])
                info['image_timestamp'] = float(veh_sample_info['image_timestamp'])
                info['scene_token'] = veh_sample_info['sequence_id']
                info['location'] = veh_sample_info['intersection_loc']
                info['frame_idx'] = idx
                
                inf_sample_info = inf_sample_mappings[sample_token]
                info['token_inf'] = inf_sample_info['frame_id']
                info['timestamp_inf'] = float(inf_sample_info['pointcloud_timestamp'])
                info['image_timestamp_inf'] = float(inf_sample_info['image_timestamp'])

                # info_infra['token'] = inf_sample_info['frame_id']
                # info_infra['timestamp'] = float(inf_sample_info['pointcloud_timestamp'])
                # info_infra['image_timestamp'] = float(inf_sample_info['image_timestamp'])
                # info_infra['scene_token'] = inf_sample_info['sequence_id']
                # info_infra['location'] = inf_sample_info['intersection_loc']
                # info_infra['frame_idx'] = idx

                # info['system_error_offset'] = coop_sample_info['system_error_offset'] # TODO: system_error_offset은 없는디??? 나중에 필요하면 만들기.
                sample_infos.append(info)
                # sample_infos_infra.append(info_infra)
    print("skip_sample_number in _generate_sample_infos_coop _ step2: ", skip_sample_number2)
    for sample_info in sample_infos:
        sample_token = sample_info['token']
        sample_info_mappings[sample_token] = sample_info
        
        
    # for sample_info_infra in sample_infos_infra:
    #     sample_token_infra = sample_info_infra['token']
    #     sample_info_mappings_infra[sample_token_infra] = sample_info_infra
        
    # breakpoint()
    
    return sample_infos, sample_info_mappings # , sample_infos_infra, sample_info_mappings_infra

def _generate_sample_infos(root_path, sample_info_mappings, v2x_side, skip_noinfra, infra_set_use_minus1):
    """Get the prev and next sample token for a given `sample_data_token`.
    Args:
        data_infos (list): data_infos loaded from data_info.json file.
    Return:
        list[dict]: List of sample info
        dict: mapping sample token to sample info   
    """
    
    root_path = Path(root_path)
    run_paths = sorted([p for p in root_path.iterdir() if p.is_dir()])
    
    sample_mappings = {}
    scene_data_dict = {}
    # for data_info in data_infos:
    for run_path in run_paths:
        scene_token = run_path.name
        if scene_token not in scene_data_dict.keys():
            scene_data_dict[scene_token] = []
        
        if skip_noinfra:
            infra_possible_1 = run_path / "-1"
            infra_possible_2 = run_path / "-2"
            if (not infra_possible_1.is_dir()) and (not infra_possible_2.is_dir()):
                continue
        
        # prefer 1, fallback to 2   # TODO: 1번 차량이 있으면 1번을 기본적으로 사용. 없으면 2번 차량을 사용
        if v2x_side == "vehicle-side":
            p = run_path / "1"
            if not p.is_dir():
                p = run_path / "2"
        elif v2x_side == "infrastructure-side":
            if infra_set_use_minus1:
                p = run_path / "-1"
                if not p.is_dir():
                    p = run_path / "-2" 
            else:
                p = run_path / "-2"
                if not p.is_dir():
                    p = run_path / "-1"   
                    
        tokens = sorted({
            f.name[:6]
            for f in p.iterdir()
            if f.is_file() and f.name[:6].isdigit()
        })

        for token in tokens:
            sample_token = token      # TODO: 이렇게 해야 구분이 가능해짐
            # TODO: 아래의 data_info 가 정확한지 확인하기 바람.
            mapping_token = scene_token + "_" + sample_token
            data_info = {}
            data_info['frame_id'] = mapping_token
            # TODO: 아래의 timestamp는 나중에 실제 time 간격으로 바꿔줘야 할수도??? 우선은 다 align이 
            # 맞다고 가정하고 했음.
            data_info['pointcloud_timestamp'] = float(token) * 1e5
            data_info['image_timestamp'] = float(token) * 1e5
            data_info['sequence_id'] = scene_token
            data_info['intersection_loc'] = "v2x_real_map"   # Map의 location을 의미하는데, 우리는 map id가 따로 없으므로 None대입.
            
            # TODO: data_info에 'frame_id',
            sample_mappings[mapping_token] = data_info
            scene_data_dict[scene_token].append(mapping_token)

    sample_infos = []
    for run_path in run_paths:
        scene_token = run_path.name
        scene_data_dict[scene_token].sort()     # 해당 시퀀스의 샘플들 있음 

        for idx in range(len(scene_data_dict[scene_token])):
            info = {}
            if idx == 0:
                info['prev'] = ''
            else:
                info['prev'] = scene_data_dict[scene_token][idx - 1]

            if idx == len(scene_data_dict[scene_token]) - 1:
                info['next'] = ''
            else:
                info['next'] = scene_data_dict[scene_token][idx + 1]

            sample_token = scene_data_dict[scene_token][idx]
            sample_info = sample_mappings[sample_token]
            info['token'] = sample_info['frame_id']
            info['timestamp'] = float(sample_info['pointcloud_timestamp'])
            info['image_timestamp'] = float(sample_info['image_timestamp'])
            info['scene_token'] = sample_info['sequence_id']
            info['location'] = sample_info['intersection_loc']
            info['frame_idx'] = idx

            sample_infos.append(info)

    
    for sample_info in sample_infos:
        sample_token = sample_info['token']
        sample_info_mappings[sample_token] = sample_info
    return sample_infos, sample_info_mappings

def _get_total_annotations_coop(root_path, sample_info_mappings, total_annotations, skip_noinfra, keep_mapping_tokens):
    # total_annotations = {}
    root_path = Path(root_path)
    run_paths = sorted([p for p in root_path.iterdir() if p.is_dir()])
    skip_sample_number = 0
    count = 0
    for run_path in run_paths:
        scene_token = run_path.name
        
        if skip_noinfra:
            infra_possible_1 = run_path / "-1"
            infra_possible_2 = run_path / "-2"
            if (not infra_possible_1.is_dir()) and (not infra_possible_2.is_dir()):
                continue
        
        ## NOTE: 기원 수정중 ======================= ##
        p1 = run_path / "1"
        p2 = run_path / "2"
        r1 = run_path / "-1"
        r2 = run_path / "-2"

        p_list = [(name, path) for name, path in [("1", p1), ("2", p2), ("-1", r1), ("-2", r2)] if path.is_dir()]
        veh_list  = [(n, p) for (n, p) in p_list if not n.startswith("-")]   # "1","2"
        infra_list = [(n, p) for (n, p) in p_list if n.startswith("-")]      # "-1","-2"
        pairs = [((v_name, v_path), (i_name, i_path))
                for (v_name, v_path) in veh_list
                for (i_name, i_path) in infra_list]
        pairs_sample = [ (f"{v_name}_{i_name}", v_path, i_path) for ((v_name,v_path),(i_name,i_path)) in pairs ]
        
        
        
            
        # prefer 1, fallback to 2   # TODO: 1번 차량이 있으면 1번을 기본적으로 사용. 없으면 2번 차량을 사용
        # p = run_path / "1"
        # if not p.is_dir():
        #     p = run_path / "2"
            
        # r = run_path / "-1"
        # if not r.is_dir():
        #     r = run_path / "-2"
            
        name, path = p_list[0]
        
        tokens = sorted({
            f.name[:6]
            for f in path.iterdir()
            if f.is_file() and f.name[:6].isdigit()
        })        
        # breakpoint()
        
        for pair_name, v_path, i_path in pairs_sample:
            for token in tokens:
                mapping_token = scene_token + "_folder_" + pair_name + "_" + token + "_folder_" + pair_name
                if mapping_token not in keep_mapping_tokens:
                    # breakpoint()
                    # print("11passed mapping_token: ", mapping_token)
                    skip_sample_number += 1
                    continue
                # breakpoint()
                new_scene_token = sample_info_mappings[mapping_token]['scene_token']
                total_annotations[mapping_token] = {}
                # breakpoint()
                veh_annotation_path = osp.join(v_path, f"{token}.yaml")
                veh_anno_yaml = load_yaml(veh_annotation_path)          # dict로 로드됨
                veh_vehicles = veh_anno_yaml.get("vehicles", {})        # dict: {track_id: {...}}
                
                inf_annotation_path = osp.join(i_path, f"{token}.yaml")
                inf_anno_yaml = load_yaml(inf_annotation_path)          # dict로 로드됨
                inf_vehicles = inf_anno_yaml.get("vehicles", {})        # dict: {track_id: {...}}
                
                

                # TODO: 이 아래로 잘 된건지 확인 부족. 
                for track_id_str, v in veh_vehicles.items():
                    track_id = int(track_id_str)
                    annotation = dict(v)  # 복사해서 수정
                    if class_names_nuscenes_mappings[annotation["obj_type"]] == "traffic_cone":
                        continue
                    annotation["track_id"] = track_id
                    annotation["type"] = class_names_nuscenes_mappings[annotation["obj_type"]]
                    annotation["instance_token"] = gen_token(track_id, new_scene_token)
                    # NOTE: 아래로 새로 추가
                    
                    # TODO: 두번쩨가 yaw로 되어 있다고 가정
                    annotation["rotation"] = float(np.deg2rad(annotation["angle"][1]))  # yaw(deg) -> rad

                    #### NOTE: ========================================= #### 
                    annotation["3d_location"] = {
                        "x": float(annotation["location"][0]),
                        "y": float(annotation["location"][1]),
                        "z": float(annotation["location"][2]),
                    }

                    annotation["3d_dimensions"] = {
                        "l": float(2.0 * annotation["extent"][0]),
                        "w": float(2.0 * annotation["extent"][1]),
                        "h": float(2.0 * annotation["extent"][2]),
                    }
                    
                    # TODO: 일단 occluded state는 이렇게 해둠. -> 좀 처리를 해줘야 할 수도 있을 거 같은데...
                    annotation["occluded_state"] = float(0)
                    
                    anno_token = gen_token(track_id, mapping_token)  
                    annotation["token"] = anno_token    # annotation token의 unique함이 필요함. instance마다, sample마다.

                    total_annotations[mapping_token][anno_token] = annotation    
                    
                
                not_contain_infra = False        # TODO: False가 의도한 동작임.
                if not_contain_infra == True:
                    pass
                else:
                    true_ego_pose_veh = veh_anno_yaml.get("true_ego_pose", None)
                    true_ego_pose_veh_array = np.array(true_ego_pose_veh)
                    # true_ego_pose_veh_xyz = true_ego_pose_veh_array[:3]
                    true_ego_pose_veh_xy = true_ego_pose_veh_array[:2]
                    for track_id_str, v in inf_vehicles.items():
                        
                        # annotation_xyz = np.array([
                        #     float(v["location"][0]),
                        #     float(v["location"][1]),
                        #     float(v["location"][2]),
                        # ])
                        annotation_xy = np.array([
                            float(v["location"][0]),
                            float(v["location"][1]),
                        ])
                        # NOTE: Ego vehicle 제거
                        # print("distance: ", np.linalg.norm(annotation_xy - true_ego_pose_veh_xy))
                        if np.linalg.norm(annotation_xy - true_ego_pose_veh_xy) <= 1.5:
                            # print("distance: ", np.linalg.norm(annotation_xy - true_ego_pose_veh_xy))
                            # breakpoint()
                            count += 1
                            continue
                        
                        track_id = int(track_id_str)
                        annotation = dict(v)  # 복사해서 수정
                        if class_names_nuscenes_mappings[annotation["obj_type"]] == "traffic_cone":
                            continue
                        annotation["track_id"] = track_id
                        annotation["type"] = class_names_nuscenes_mappings[annotation["obj_type"]]
                        annotation["instance_token"] = gen_token(track_id, new_scene_token)
                        # NOTE: 아래로 새로 추가
                        
                        # TODO: 두번쩨가 yaw로 되어 있다고 가정
                        annotation["rotation"] = float(np.deg2rad(annotation["angle"][1]))  # yaw(deg) -> rad

                        #### NOTE: ========================================= #### 
                        annotation["3d_location"] = {
                            "x": float(annotation["location"][0]),
                            "y": float(annotation["location"][1]),
                            "z": float(annotation["location"][2]),
                        }

                        annotation["3d_dimensions"] = {
                            "l": float(2.0 * annotation["extent"][0]),
                            "w": float(2.0 * annotation["extent"][1]),
                            "h": float(2.0 * annotation["extent"][2]),
                        }
                        
                        # TODO: 일단 occluded state는 이렇게 해둠. -> 좀 처리를 해줘야 할 수도 있을 거 같은데...
                        annotation["occluded_state"] = float(0)
                        
                        anno_token = gen_token(track_id, mapping_token)  
                        annotation["token"] = anno_token    # annotation token의 unique함이 필요함. instance마다, sample마다.

                        total_annotations[mapping_token][anno_token] = annotation    
                    
                        # TODO: 현재 여기서는 total_annotations 안에 vehicle side의 object랑 infrastructure side의 object를 모두 포함시키도록 함. unvisible 처리를 뒤에서 해주는지 확인해보자...
    print("skip_sample_number in _generate_sample_infos_coop _ step1: ", skip_sample_number)
    print("skip_annotation count. it should same with total sample number train: 12263: ", count)
    return total_annotations

def _get_total_annotations(root_path, sample_infos, sample_info_mappings, total_annotations, v2x_side, skip_noinfra, infra_set_use_minus1):      # TODO: sample_infos 기반으로 코드 쉽게 변경 가능할듯
    # total_annotations = {}
    root_path = Path(root_path)
    run_paths = sorted([p for p in root_path.iterdir() if p.is_dir()])
    for run_path in run_paths:
        scene_token = run_path.name
        
        if skip_noinfra:
            infra_possible_1 = run_path / "-1"
            infra_possible_2 = run_path / "-2"
            if (not infra_possible_1.is_dir()) and (not infra_possible_2.is_dir()):
                continue
            
        # prefer 1, fallback to 2   # TODO: 1번 차량이 있으면 1번을 기본적으로 사용. 없으면 2번 차량을 사용
        if v2x_side == "vehicle-side":
            p = run_path / "1"
            if not p.is_dir():
                p = run_path / "2"
        elif v2x_side == "infrastructure-side":
            if infra_set_use_minus1:
                p = run_path / "-1"
                if not p.is_dir():
                    p = run_path / "-2" 
            else:
                p = run_path / "-2"
                if not p.is_dir():
                    p = run_path / "-1" 
            
        tokens = sorted({
            f.name[:6]
            for f in p.iterdir()
            if f.is_file() and f.name[:6].isdigit()
        })        

        for token in tokens:
            # sample_token = token
            mapping_token = scene_token + "_" + token
            scene_token = sample_info_mappings[mapping_token]['scene_token']
            frame_idx = sample_info_mappings[mapping_token]['frame_idx']
            timestamp = sample_info_mappings[mapping_token]['timestamp']

            annotation_path = osp.join(p, f"{token}.yaml")
            anno_yaml = load_yaml(annotation_path)          # dict로 로드됨
            vehicles = anno_yaml.get("vehicles", {})        # dict: {track_id: {...}}
            total_annotations[mapping_token] = {}

            # TODO: 이 아래로 잘 된건지 확인 부족. 
            for track_id_str, v in vehicles.items():
                track_id = int(track_id_str)
                annotation = dict(v)  # 복사해서 수정
                if class_names_nuscenes_mappings[annotation["obj_type"]] == "traffic_cone":
                    continue
                annotation["track_id"] = track_id
                annotation["type"] = class_names_nuscenes_mappings[annotation["obj_type"]]
                annotation["instance_token"] = gen_token(track_id, scene_token)
                # NOTE: 아래로 새로 추가
                


                # TODO: 두번쩨가 yaw로 되어 있다고 가정
                annotation["rotation"] = float(np.deg2rad(annotation["angle"][1]))  # yaw(deg) -> rad

                #### NOTE: ========================================= #### 
                annotation["3d_location"] = {
                    "x": float(annotation["location"][0]),
                    "y": float(annotation["location"][1]),
                    "z": float(annotation["location"][2]),
                }

                annotation["3d_dimensions"] = {
                    "l": float(2.0 * annotation["extent"][0]),
                    "w": float(2.0 * annotation["extent"][1]),
                    "h": float(2.0 * annotation["extent"][2]),
                }
                
                # TODO: 일단 occluded state는 이렇게 해둠. -> 좀 처리를 해줘야 할 수도 있을 거 같은데...
                annotation["occluded_state"] = float(0)
                
                
                # h = float(2.0 * annotation["extent"][2])
                # z = float(annotation["location"][2])

                # annotation["3d_dimensions"] = {
                #     "l": float(2.0 * annotation["extent"][0]),
                #     "w": float(2.0 * annotation["extent"][1]),
                #     "h": h,
                # }

                # annotation["3d_location"] = {
                #     "x": float(annotation["location"][0]),
                #     "y": float(annotation["location"][1]),
                #     "z": z + 0.5 * h,   # bottom -> center
                # }
                #### =============================================== ####
                                
                # annotation["token"] = gen_token(track_id, scene_token)  # V2X-REAL 여기서는 오브젝트마다 token은 없어보임. null 넣으면 나중에 모델에서 에러나서 일단 track id로 대체 
                
                anno_token = gen_token(track_id, mapping_token)  
                annotation["token"] = anno_token    # annotation token의 unique함이 필요함. instance마다, sample마다.

                total_annotations[mapping_token][anno_token] = annotation      
                
                
    return total_annotations


# def _get_total_annotations(root_path, sample_infos, sample_info_mappings):      # TODO: sample_infos 기반으로 코드 쉽게 변경 가능할듯
#     total_annotations = {}
#     for sample_info in sample_infos:
#         mapping_token = sample_info['scene_token'] + "_" + sample_info['token'] 
#         scene_token = sample_info_mappings[mapping_token]['scene_token']
#         # frame_idx = sample_info_mappings[mapping_token]['frame_idx']
#         # timestamp = sample_info_mappings[mapping_token]['timestamp']
#         run_path = Path(osp.join(root_path, sample_info['scene_token']))
#         p = run_path / "1"
#         if not p.is_dir():
#             p = run_path / "2"

#         annotation_path = osp.join(f"{p}/{sample_info['token']}.yaml")
#         # breakpoint()
#         anno_yaml = load_yaml(annotation_path)          # dict로 로드됨
#         vehicles = anno_yaml.get("vehicles", {})        # dict: {track_id: {...}}
#         total_annotations[mapping_token] = {}

#         # TODO: 이 아래로 잘 된건지 확인 부족. 
#         for track_id_str, v in vehicles.items():
#             track_id = int(track_id_str)
#             annotation = dict(v)  # 복사해서 수정
#             annotation["track_id"] = track_id
#             annotation["type"] = class_names_nuscenes_mappings[annotation["obj_type"]]
#             annotation["instance_token"] = gen_token(track_id, scene_token)
#             anno_token = gen_token(track_id, mapping_token)  
#             total_annotations[mapping_token][anno_token] = annotation      
                
                
#     return total_annotations


def _generate_unvisible_annotations(source_name, sample_info_mappings, secene_frame_mappings, instance_token_mappings,
                                    total_annotations,lidar_ego_global_infos):
    """Generate annotations for totally occluded objects and make trajectory complete.
    Args:
        root_path: data root
        data_infos: (list): data_infos loaded from data_info.json file.
    Return:
        dict[dict]: {'sample_token': {'anno_token': }}
    """
    ## Interpolate box   
    for instance_token in instance_token_mappings.keys():
        cur_instance_samples = instance_token_mappings[instance_token]
        cur_scene_token = cur_instance_samples[0]['scene_token']
        cur_scene_token_end = cur_instance_samples[-1]['scene_token']
        assert cur_scene_token == cur_scene_token_end

        for ii in range(len(cur_instance_samples) - 1):
            cur_frame_idx = cur_instance_samples[ii]['frame_idx'] + 1
            while cur_frame_idx != cur_instance_samples[ii + 1]['frame_idx']:
                # linear interpolation
                loc_ii_0 = cur_instance_samples[ii]['annotation']['3d_location']
                loc_ii_1 = cur_instance_samples[ii + 1]['annotation']['3d_location']
                rot_ii_0 = cur_instance_samples[ii]['annotation']['rotation']
                rot_ii_1 = cur_instance_samples[ii + 1]['annotation']['rotation']

                timestamp_ii_0 = cur_instance_samples[ii]['timestamp']
                timestamp_ii_1 = cur_instance_samples[ii + 1]['timestamp']

                cur_sample_token = secene_frame_mappings[(cur_scene_token, cur_frame_idx)]
                # cur_time_stamp = sample_data_mappings[cur_sample_token]['pointcloud_timestamp']
                cur_timestamp = sample_info_mappings[cur_sample_token]['timestamp']

                # if cur_timestamp == '1626155888.384136':
                #     cur_timestamp = cur_timestamp

                sample_token_0 = cur_instance_samples[ii]['sample_token']
                sample_token_1 = cur_instance_samples[ii+1]['sample_token']
                
                cur_loc = loc_linear_interpolation(loc_ii_0, loc_ii_1, timestamp_ii_0, timestamp_ii_1, cur_timestamp,
                                                   lidar_ego_global_infos[sample_token_0],lidar_ego_global_infos[sample_token_1],
                                                   lidar_ego_global_infos[cur_sample_token])
                cur_rot = rot_linear_interpolation(rot_ii_0, rot_ii_1, timestamp_ii_0, timestamp_ii_1, cur_timestamp)
                cur_anno_token = gen_token(source_name, cur_sample_token, str(cur_loc['x']), str(cur_loc['y']), str(cur_loc['z']))

                # angle (degrees) 보간: box_to_ego_coord 경로에서 x_to_world()에 필요
                angle_ii_0 = np.array(cur_instance_samples[ii]['annotation'].get('angle', np.zeros(3)), dtype=np.float64)
                angle_ii_1 = np.array(cur_instance_samples[ii + 1]['annotation'].get('angle', np.zeros(3)), dtype=np.float64)
                time_ratio = (float(cur_timestamp) - float(timestamp_ii_0)) / (float(timestamp_ii_1) - float(timestamp_ii_0))
                # 각 성분별 shortest-path 보간 (degrees, ±180 wrapping)
                angle_diff = angle_ii_1 - angle_ii_0
                angle_diff = (angle_diff + 180.0) % 360.0 - 180.0
                cur_angle = (angle_ii_0 + time_ratio * angle_diff).tolist()

                cur_instance_sample_anno = {
                    "token": cur_anno_token,
                    "type": cur_instance_samples[ii]['annotation']['type'],
                    "track_id": cur_instance_samples[ii]['annotation']['track_id'],
                    "truncated_state": 0,
                    "occluded_state": 3,
                    "3d_dimensions": cur_instance_samples[ii]['annotation']['3d_dimensions'],
                    "3d_location": cur_loc,
                    "rotation": cur_rot,
                    "angle": cur_angle,
                    "instance_token": instance_token
                }

                total_annotations[cur_sample_token][cur_anno_token] = cur_instance_sample_anno

                cur_frame_idx = cur_frame_idx + 1

    return total_annotations


def loc_linear_interpolation(loc_ii_0, loc_ii_1, timestamp_ii_0, timestamp_ii_1, cur_timestamp, \
                             lidar_ego_global_info_0,lidar_ego_global_info_1,cur_lidar_ego_global_info):
    """Use linear interpolation to estimate the 3d location for occluded objects.
    NOTE: V2X-REAL에서는 3d_location이 이미 global(world) 좌표이므로,
          좌표 변환 없이 global에서 직접 보간하고 global 좌표로 반환한다.
    """
    timestamp_ii_0 = float(timestamp_ii_0) / 1e6
    timestamp_ii_1 = float(timestamp_ii_1) / 1e6
    cur_timestamp = float(cur_timestamp) / 1e6

    # V2X-REAL: 3d_location은 이미 global 좌표 → 직접 보간
    center_0 = np.array([loc_ii_0['x'], loc_ii_0['y'], loc_ii_0['z']], dtype=np.float64)
    center_1 = np.array([loc_ii_1['x'], loc_ii_1['y'], loc_ii_1['z']], dtype=np.float64)

    # global interpolation
    cur_center = center_0 + (center_1 - center_0) * (cur_timestamp - timestamp_ii_0) / (timestamp_ii_1 - timestamp_ii_0)

    cur_loc = {}
    cur_loc['x'] = cur_center[0]
    cur_loc['y'] = cur_center[1]
    cur_loc['z'] = cur_center[2]

    return cur_loc


def rot_linear_interpolation(rot_ii_0, rot_ii_1, timestamp_ii_0, timestamp_ii_1, cur_timestamp):
    """Use linear interpolation to estimate the rotation for occluded objects.
    """
    timestamp_ii_0 = float(timestamp_ii_0) / 1e6
    timestamp_ii_1 = float(timestamp_ii_1) / 1e6
    cur_timestamp = float(cur_timestamp) / 1e6

    time_ratio = (cur_timestamp - timestamp_ii_0) / (timestamp_ii_1 - timestamp_ii_0)
    # shortest-path wrapping to [-pi, pi)
    diff = (rot_ii_1 - rot_ii_0 + pi) % (2 * pi) - pi
    
    cur_rot = rot_ii_0 + time_ratio * diff
    # normalize to [-pi, pi)
    cur_rot = (cur_rot + pi) % (2 * pi) - pi
    
    return cur_rot

def _add_annotation_velocity_prev_next(total_annotations, instance_token_mappings, lidar_ego_global_infos):
    """Generate velocity and prev/next token for annotations.
    Args:
        total_annotations: added occluded annotations
        sample_info_mappings
        data_infos
    """
    ## Generate Velocity and Successors
    for instance_token in instance_token_mappings.keys():
        cur_instance_samples = instance_token_mappings[instance_token]
        cur_scene_token = cur_instance_samples[0]['scene_token']
        
        for ii in range(len(cur_instance_samples)):
            if ii == 0:
                prev_anno_token = ''
            else:
                prev_anno_token = cur_instance_samples[ii - 1]['annotation']['token']

            if ii == len(cur_instance_samples) - 1:
                next_anno_token = ''
            else:
                next_anno_token = cur_instance_samples[ii + 1]['annotation']['token']

            if ii == len(cur_instance_samples) - 1:
                gt_velocity = [0, 0]
            else:
                loc_ii_0 = cur_instance_samples[ii]['annotation']['3d_location']
                loc_ii_1 = cur_instance_samples[ii + 1]['annotation']['3d_location']

                sample_token_0 = cur_instance_samples[ii]['sample_token']
                sample_token_1 = cur_instance_samples[ii+1]['sample_token']

                # cvt to global
                center_0 = np.array([loc_ii_0['x'], loc_ii_0['y'], loc_ii_0['z']])
                center_1 = np.array([loc_ii_1['x'], loc_ii_1['y'], loc_ii_1['z']])

                # lidar2ego
                # center_0 = np.dot(Quaternion(lidar_ego_global_infos[sample_token_0]['lidar2ego_rotation']).rotation_matrix, center_0)   \
                #                     + np.array(lidar_ego_global_infos[sample_token_0]['lidar2ego_translation'])
                center_1 = np.dot(Quaternion(lidar_ego_global_infos[sample_token_1]['lidar2ego_rotation']).rotation_matrix, center_1)   \
                                    + np.array(lidar_ego_global_infos[sample_token_1]['lidar2ego_translation'])

                # ego2global
                # center_0 = np.dot(Quaternion(lidar_ego_global_infos[sample_token_0]['ego2global_rotation']).rotation_matrix, center_0)  \
                #                     + np.array(lidar_ego_global_infos[sample_token_0]['ego2global_translation'])
                center_1 = np.dot(Quaternion(lidar_ego_global_infos[sample_token_1]['ego2global_rotation']).rotation_matrix, center_1)  \
                                    + np.array(lidar_ego_global_infos[sample_token_1]['ego2global_translation']) 
                
                global2ego_r0 = np.linalg.inv(Quaternion(lidar_ego_global_infos[sample_token_0]['ego2global_rotation']).rotation_matrix)
                global2ego_t0 = - np.array(lidar_ego_global_infos[sample_token_0]['ego2global_translation']).reshape(1, 3) @ global2ego_r0.T
                global2ego_t0 = global2ego_t0.reshape(3)

                ego2lidar_r0 = np.linalg.inv(Quaternion(lidar_ego_global_infos[sample_token_0]['lidar2ego_rotation']).rotation_matrix)
                ego2lidar_t0 = - np.array(lidar_ego_global_infos[sample_token_0]['lidar2ego_translation']).reshape(1, 3) @ ego2lidar_r0.T
                ego2lidar_t0 = ego2lidar_t0.reshape(3)                

                # ego2lidar
                center_1 = np.dot(global2ego_r0,center_1) + global2ego_t0  
                center_1 = np.dot(ego2lidar_r0,center_1) + ego2lidar_t0                         

                # time_delta
                timestamp_ii_0 = cur_instance_samples[ii]['timestamp']
                timestamp_ii_1 = cur_instance_samples[ii + 1]['timestamp']
                timestamp_ii_0 = float(timestamp_ii_0) / 1e6
                timestamp_ii_1 = float(timestamp_ii_1) / 1e6
                time_delta = timestamp_ii_1 - timestamp_ii_0

                # gt_velocity_dict = {}
                # for key in loc_ii_0.keys():
                #     gt_velocity_dict[key] = (loc_ii_1[key] - loc_ii_0[key]) / (timestamp_ii_1 - timestamp_ii_0)
                # gt_velocity = [gt_velocity_dict['x'], gt_velocity_dict['y']]
                gt_velocity = (center_1 - center_0) / time_delta
                gt_velocity = gt_velocity[:2]

            instance_token_mappings[instance_token][ii]['annotation']['gt_velocity'] = gt_velocity
            instance_token_mappings[instance_token][ii]['annotation']['prev'] = prev_anno_token
            instance_token_mappings[instance_token][ii]['annotation']['next'] = next_anno_token

    return total_annotations, instance_token_mappings

def _get_secene_frame_mappings(sample_info_mappings):
    secene_frame_mappings = {}
    for sample_token in sample_info_mappings.keys():
        scene_token = sample_info_mappings[sample_token]['scene_token']
        frame_idx = sample_info_mappings[sample_token]['frame_idx']
        secene_frame_mappings[(scene_token, frame_idx)] = sample_token

    return secene_frame_mappings


def _get_instance_token_mappings(total_annotations, sample_info_mappings):
    instance_token_mappings = {}

    for sample_token in total_annotations.keys():
        annotations = total_annotations[sample_token]
        scene_token = sample_info_mappings[sample_token]['scene_token']
        frame_idx = sample_info_mappings[sample_token]['frame_idx']
        timestamp = sample_info_mappings[sample_token]['timestamp']

        for anno_token in annotations.keys():
            annotation = annotations[anno_token]
            instance_token = annotation["instance_token"]
            if instance_token not in instance_token_mappings.keys():
                instance_token_mappings[instance_token] = []
            instance_token_mappings[instance_token].append({
                'scene_token': scene_token,
                'frame_idx': frame_idx,
                'sample_token': sample_token,
                'timestamp': timestamp,
                'annotation': annotation})

    # sorted by frame_idx, for downstream usage
    for instance_token in instance_token_mappings.keys():
        # sorted(instance_token_mappings[instance_token], key=lambda annotation: annotation['frame_idx'])       # NOTE: 기원 260211 수정
        instance_token_mappings[instance_token] = sorted(
            instance_token_mappings[instance_token],
            key=lambda a: a['frame_idx']
        )

    return instance_token_mappings


def generate_json_maps_files(data_root, version='v1.0-mini'):
    json_types = ['category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor',
                  'ego_pose', 'log', 'scene', 'sample', 'sample_data', 'sample_annotation', 'map']

    import shutil
    if not os.path.exists(osp.join(data_root, version)):
        tmp_nuscenes_json_root = '/data/ad_sharing/datasets/nuScenes/nuScenes_v1.0-mini/v1.0-mini'
        shutil.copytree(tmp_nuscenes_json_root, osp.join(data_root, version))

    if not os.path.exists(osp.join(data_root, 'maps')):
        tmp_nuscenes_map_root = '/data/ad_sharing/datasets/nuScenes/nuScenes_v1.0-mini/maps'
        shutil.copytree(tmp_nuscenes_map_root, osp.join(data_root, 'maps'))

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--data-root', type=str, default="./datasets/v2xreal")
    parser.add_argument('--save-root', type=str, default="./data/infos/v2xreal")
    parser.add_argument('--split-file', type=str, default="./data/split_datas_V2XREAL/split_datas_V2XREAL.json")
    parser.add_argument('--v2x-side', type=str, default="vehicle-side")
    parser.add_argument('--version', type=str, default="v1.0-trainval")
    parser.add_argument('--info-prefix', type=str, default="spd")
    parser.add_argument('--skip-noinfra', type=bool, default=True)
    # parser.add_argument('--infra_set_use_minus1', type=bool, default=True)
    parser.add_argument('--infra_set_use_minus1', type=str2bool, default=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    import os

    curDirectory = os.getcwd()
    basepath = os.path.basename(os.path.normpath(curDirectory))
    if basepath != 'UniV2X_REAL':
        os.chdir('UniV2X_REAL/')

    curDirectory = os.getcwd()
    print(curDirectory)

    v2x_side = args.v2x_side
    data_root = args.data_root
    save_root = args.save_root
    split_path = args.split_file
    can_bus_root_path = ''
    info_prefix = args.info_prefix
    skip_noinfra = args.skip_noinfra
    infra_set_use_minus1 = args.infra_set_use_minus1
    # breakpoint()
    if infra_set_use_minus1 == False:
        save_root = save_root + "_infra2"

    print(data_root)
    print(save_root)
    print(v2x_side)

    if v2x_side == 'cooperative':
        # generate_json_maps_files(data_root, version=v2x_side)
        total_annotations, sample_info_mappings, spd_infos, spd_infos_infra = create_spd_infos_coop(data_root,
                            save_root,
                            v2x_side,
                            split_path,
                            can_bus_root_path,
                            info_prefix,
                            version=args.version,
                            max_sweeps=10,
                            skip_noinfra=skip_noinfra)
        
    else:   
        # generate_json_maps_files(data_root, version=v2x_side)
        total_annotations, sample_info_mappings, spd_infos = create_spd_infos(data_root,
                                                                            save_root,
                                                                            v2x_side,
                                                                            split_path,
                                                                            can_bus_root_path,
                                                                            info_prefix,
                                                                            version=args.version,
                                                                            max_sweeps=10,
                                                                            skip_noinfra=skip_noinfra,
                                                                            infra_set_use_minus1=infra_set_use_minus1)

