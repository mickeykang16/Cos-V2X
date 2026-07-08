import random
import math
import os
from os import path as osp
import cv2
import tempfile
import copy
import prettytable

import numpy as np
import torch
from torch.utils.data import Dataset
import pyquaternion
from shapely.geometry import LineString
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.eval.detection.config import config_factory as det_configs
from nuscenes.eval.common.config import config_factory as track_configs

import mmcv
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from .utils import (
    draw_lidar_bbox3d_on_img,
    draw_lidar_bbox3d_on_bev,
)


@DATASETS.register_module()
class NuScenes3DDataset(Dataset):
    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }
    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }
    CLASSES = (
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    )
    MAP_CLASSES = (
        'ped_crossing',
        'divider',
        'boundary',
    )
    ID_COLOR_MAP = [
        (59, 59, 238),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 255),
        (0, 127, 255),
        (71, 130, 255),
        (127, 127, 0),
    ]

    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        test_mode=False,
        det3d_eval_version="detection_cvpr_2019",
        track3d_eval_version="tracking_nips_2019",
        version="v1.0-trainval",
        use_valid_flag=False,
        vis_score_threshold=0.25,
        data_aug_conf=None,
        sequences_split_num=1,
        with_seq_flag=False,
        keep_consistent_seq_aug=True,
        work_dir=None,
        eval_config=None,
        with_infra_cam=False,
        infra_cam_select=None,
        imu_noise=False,
        scene_tokens_filter_file="test_scene_tokens_5s_no_stop.txt",
    ):
        self.version = version
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.box_mode_3d = 0
        self.with_infra_cam = with_infra_cam
        self.infra_cam_select = infra_cam_select
        
        self.imu_noise = imu_noise
        self.imu_noise_data = None
        self._imu_noise_debug_count = 0  # print first N noise applications
        if self.imu_noise and data_root is not None:
            noise_path = osp.join(
                data_root, 'imu_noise_svx0.3_svy0.05_sa0.5_sw0.02_seed42.pkl'
            )
            self.imu_noise_data = mmcv.load(noise_path, file_format='pkl')
            print(f'[IMU noise] ✅ Loaded {len(self.imu_noise_data)} entries from {noise_path}')
        elif self.imu_noise and data_root is None:
            print('[IMU noise] ⚠️  imu_noise=True but data_root is None – skipping')
        else:
            print('[IMU noise] ℹ️  imu_noise=False – no noise will be applied')

        if classes is not None:
            self.CLASSES = classes
        if map_classes is not None: 
            self.MAP_CLASSES = map_classes
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.ann_file)

        # Filter to specific scenes if a scene token list file is provided (test_mode only)
        if test_mode and scene_tokens_filter_file is not None and osp.isfile(scene_tokens_filter_file):
            with open(scene_tokens_filter_file, 'r') as f:
                allowed_scene_tokens = set(line.strip() for line in f if line.strip())
            before = len(self.data_infos)
            self.data_infos = [
                info for info in self.data_infos
                if info.get('scene_token') in allowed_scene_tokens
            ]
            after = len(self.data_infos)
            print(f'[SceneFilter] ✅ Loaded {len(allowed_scene_tokens)} scene tokens from {scene_tokens_filter_file}')
            print(f'[SceneFilter] 📊 data_infos: {before} → {after} samples ({len(allowed_scene_tokens)} scenes)')
        else:
            if test_mode and scene_tokens_filter_file is not None and not osp.isfile(scene_tokens_filter_file):
                print(f'[SceneFilter] ⚠️  Filter file not found: {scene_tokens_filter_file} – no filter applied')
            print(f'[SceneFilter] ℹ️  Total samples: {len(self.data_infos)} (test_mode={test_mode})')

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        self.with_velocity = with_velocity
        self.det3d_eval_version = det3d_eval_version
        self.det3d_eval_configs = det_configs(self.det3d_eval_version)
        self.det3d_eval_configs.class_names = list(self.det3d_eval_configs.class_range.keys())
        self.track3d_eval_version = track3d_eval_version
        self.track3d_eval_configs = track_configs(self.track3d_eval_version)
        self.track3d_eval_configs.class_names = list(self.track3d_eval_configs.class_range.keys())
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.vis_score_threshold = vis_score_threshold

        self.data_aug_conf = data_aug_conf
        self.sequences_split_num = sequences_split_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug
        if with_seq_flag:
            self._set_sequence_group_flag()
        
        self.work_dir = work_dir
        self.eval_config = eval_config

    def __len__(self):
        return len(self.data_infos)

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        if self.sequences_split_num == -1:
            self.flag = np.arange(len(self.data_infos))
            return
        
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]["sweeps"]) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == "all":
                self.flag = np.array(
                    range(len(self.data_infos)), dtype=np.int64
                )
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(
                                    bin_counts[curr_flag]
                                    / self.sequences_split_num
                                ),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )

                    for sub_seq_idx in (
                        curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert (
                    len(np.bincount(new_flags))
                    == len(np.bincount(self.flag)) * self.sequences_split_num
                )
                self.flag = np.array(new_flags, dtype=np.int64)

    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
            rotate_3d = np.random.uniform(*self.data_aug_conf["rot3d_range"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
            rotate_3d = 0
        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_config

    def __getitem__(self, idx):
        if isinstance(idx, dict):
            aug_config = idx["aug_config"]
            idx = idx["idx"]
        else:
            aug_config = self.get_augmentation()
        data = self.get_data_info(idx)
        data["aug_config"] = aug_config
        data = self.pipeline(data)
        return data

    def get_cat_ids(self, idx):
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    # def load_annotations(self, ann_file):
    #     data = mmcv.load(ann_file, file_format="pkl")
    #     data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
    #     data_infos = data_infos[:: self.load_interval]
    #     self.metadata = data["metadata"]
    #     self.version = self.metadata["version"]
    #     print(self.metadata)
    #     return data_infos
    
    def load_annotations(self, ann_file): # HH
        data = mmcv.load(ann_file, file_format="pkl")
        # 수정된 정렬 방식: scene_token으로 먼저 정렬하고, 그 다음 timestamp로 정렬
        data_infos = list(sorted(data["infos"], key=lambda e: (e["scene_token"], e["timestamp"])))
        data_infos = data_infos[:: self.load_interval]
        
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        print(self.metadata)
        return data_infos

    def anno2geom(self, annos):
        map_geoms = {}
        for label, anno_list in annos.items():
            map_geoms[label] = []
            for anno in anno_list:
                geom = LineString(anno)
                map_geoms[label].append(geom)
        return map_geoms
    
    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            token=info["token"],
            map_location=info["map_location"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"] / 1e6,
            # timestamp=info["timestamp"],
            lidar2ego_translation=info["lidar2ego_translation"],
            lidar2ego_rotation=info["lidar2ego_rotation"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
            ego_status=info['ego_status'].astype(np.float32),
            map_infos=info["map_annos"],
            scene_token=info.get("scene_token"), # Add scene_token
        )
        # IMU noise injection: add per-token noise to ego_status if enabled
        # ego_status layout: [ax, ay, az, wx, wy, wz, vx, vy, vz, steer]  (10-dim)
        # noise layout     : [vx_noise, vy_noise, ax_noise, ay_noise, wz_noise] (5-dim)
        # noise -> ego_status index mapping: [6, 7, 0, 1, 5]
        _IMU_NOISE_IDX = [6, 7, 0, 1, 5]  # ego_status dims that noise applies to
        if self.imu_noise_data is not None:
            token = info["token"]
            if token in self.imu_noise_data:
                noise = np.array(self.imu_noise_data[token], dtype=np.float32)  # (5,)
                ego_status_before = input_dict["ego_status"].copy()
                ego_status_after = ego_status_before.copy()
                ego_status_after[_IMU_NOISE_IDX] += noise
                input_dict["ego_status"] = ego_status_after
                # if self._imu_noise_debug_count < 3:
                #     self._imu_noise_debug_count += 1
                #     print(f'[IMU noise DEBUG #{self._imu_noise_debug_count}] token={token[:40]}', flush=True)
                #     print(f'  noise (vx,vy,ax,ay,wz)             = {noise}', flush=True)
                #     print(f'  ego_status before [ax,ay,wz,vx,vy] = '
                #           f'ax={ego_status_before[0]:.4f} ay={ego_status_before[1]:.4f} '
                #           f'wz={ego_status_before[5]:.4f} vx={ego_status_before[6]:.4f} '
                #           f'vy={ego_status_before[7]:.4f}', flush=True)
                #     print(f'  ego_status after  [ax,ay,wz,vx,vy] = '
                #           f'ax={ego_status_after[0]:.4f} ay={ego_status_after[1]:.4f} '
                #           f'wz={ego_status_after[5]:.4f} vx={ego_status_after[6]:.4f} '
                #           f'vy={ego_status_after[7]:.4f}', flush=True)
            # else:
            #     if self._imu_noise_debug_count < 3:
            #         self._imu_noise_debug_count += 1
            #         print(f'[IMU noise DEBUG] ⚠️  token not found in noise dict: {token[:40]}', flush=True)
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = pyquaternion.Quaternion(
            info["lidar2ego_rotation"]
        ).rotation_matrix
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])
        ego2global = np.eye(4)
        ego2global[:3, :3] = pyquaternion.Quaternion(
            info["ego2global_rotation"]
        ).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])
        input_dict["lidar2global"] = ego2global @ lidar2ego

        map_geoms = self.anno2geom(info["map_annos"])
        input_dict["map_geoms"] = map_geoms

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsic = []
            cam_names = []
            # Sort to ensure consistent order (vehicle cams first, then infra)
            for cam_type in sorted(info["cams"].keys()):
                if not self.with_infra_cam and "infrastructure" in cam_type:
                    continue

                cam_names.append(cam_type)
                cam_info = info["cams"][cam_type]
                image_paths.append(cam_info["data_path"])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                lidar2cam_t = (
                    cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                )
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = copy.deepcopy(cam_info["cam_intrinsic"])
                cam_intrinsic.append(intrinsic)
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T
                lidar2img_rts.append(lidar2img_rt)
                lidar2cam_rts.append(lidar2cam_rt)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    lidar2cam=lidar2cam_rts,
                    cam_intrinsic=cam_intrinsic,
                    cam_names=cam_names,
                )
            )

        annos = self.get_ann_info(index)
        input_dict.update(annos)
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        if "instance_inds" in info:
            instance_inds = np.array(info["instance_inds"], dtype=np.int)[mask]
            anns_results["instance_inds"] = instance_inds
            
        if 'gt_agent_fut_trajs' in info:
            gt_agent_fut_trajs = info['gt_agent_fut_trajs'][mask]
            gt_agent_fut_masks = info['gt_agent_fut_masks'][mask]
            if gt_agent_fut_trajs.ndim == 3 and gt_agent_fut_trajs.shape[1] == 60:
                n = gt_agent_fut_trajs.shape[0]
                gt_agent_fut_trajs = gt_agent_fut_trajs.reshape(n, 12, 5, 2).sum(axis=2)
                gt_agent_fut_masks = (gt_agent_fut_masks.reshape(n, 12, 5).sum(axis=2) > 0)
            anns_results['gt_agent_fut_trajs'] = gt_agent_fut_trajs
            anns_results['gt_agent_fut_masks'] = gt_agent_fut_masks

        if 'gt_ego_fut_trajs' in info:
            gt_ego_fut_trajs = info['gt_ego_fut_trajs'].copy()
            gt_ego_fut_masks = info['gt_ego_fut_masks'].copy()
            step = 1
            if gt_ego_fut_trajs.shape[0] == 60:
                step = 5
                gt_ego_fut_trajs = gt_ego_fut_trajs.reshape(12, 5, 2).sum(axis=1)
                gt_ego_fut_masks = gt_ego_fut_masks.reshape(12, 5).sum(axis=1) > 0

            anns_results['gt_ego_fut_trajs'] = gt_ego_fut_trajs
            anns_results['gt_ego_fut_masks'] = gt_ego_fut_masks
            anns_results['gt_ego_fut_cmd'] = info['gt_ego_fut_cmd']
        
            ## get future box for planning eval
            fut_ts = int(info['gt_ego_fut_masks'].sum())
            fut_boxes = []
            cur_scene_token = info["scene_token"]
            cur_T_global = get_T_global(info)
            for i in range(step, fut_ts + 1, step):
                if index + i >= len(self.data_infos):
                    break
                fut_info = self.data_infos[index + i]
                fut_scene_token = fut_info["scene_token"]
                if cur_scene_token != fut_scene_token:
                    break
                if self.use_valid_flag:
                    mask = fut_info["valid_flag"]
                else:
                    mask = fut_info["num_lidar_pts"] > 0

                fut_gt_bboxes_3d = fut_info["gt_boxes"][mask]
                
                fut_T_global = get_T_global(fut_info)
                T_fut2cur = np.linalg.inv(cur_T_global) @ fut_T_global

                center = fut_gt_bboxes_3d[:, :3] @ T_fut2cur[:3, :3].T + T_fut2cur[:3, 3]
                yaw = np.stack([np.cos(fut_gt_bboxes_3d[:, 6]), np.sin(fut_gt_bboxes_3d[:, 6])], axis=-1)
                yaw = yaw @ T_fut2cur[:2, :2].T
                yaw = np.arctan2(yaw[..., 1], yaw[..., 0])

                fut_gt_bboxes_3d[:, :3] = center
                fut_gt_bboxes_3d[:, 6] = yaw

                fut_boxes.append(fut_gt_bboxes_3d)

            anns_results['fut_boxes'] = fut_boxes
        
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None, tracking=False):
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(
                det, threshold=self.tracking_threshold if tracking else None
            )
            sample_token = self.data_infos[sample_id]["token"]
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.det3d_eval_configs,
                self.det3d_eval_version,
            )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if tracking and name in [
                    "barrier",
                    "traffic_cone",
                    "construction_vehicle",
                ]:
                    continue
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = NuScenes3DDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = NuScenes3DDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                )
                if not tracking:
                    nusc_anno.update(
                        dict(
                            detection_name=name,
                            detection_score=box.score,
                            attribute_name=attr,
                        )
                    )
                else:
                    if i < 3 and sample_id < 3:
                        print(f"DEBUG TRACKING: sample={sample_token} box_token={box.token} score={box.score} name={name}")
                    nusc_anno.update(
                        dict(
                            tracking_name=name,
                            tracking_score=box.score,
                            tracking_id=str(box.token),
                        )
                    )

                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    # def _evaluate_single(
    #     self, result_path, logger=None, result_name="img_bbox", tracking=False
    # ):
    #     from nuscenes import NuScenes

    #     output_dir = osp.join(*osp.split(result_path)[:-1])
        
    #     # 임시 버전 관리를 위한 로직 추가
    #     eval_version = self.version
    #     if self.version == "v1.0-trainval_debug":
    #         eval_version = "v1.0-trainval" # 평가 시에만 'v1.0-mini' 버전으로 취급

        
        
    #     nusc = NuScenes(
    #         version=eval_version, dataroot=self.data_root, verbose=True
    #     )
    #     eval_set_map = {
    #         "v1.0-mini": "mini_val",
    #         "v1.0-trainval": "val",
    #         # "v1.0-trainval_debug": "mini_val",
    #     }
    #     if not tracking:
    #         from nuscenes.eval.detection.evaluate import NuScenesEval

    #         # nusc_eval = NuScenesEval(
    #         #     nusc,
    #         #     config=self.det3d_eval_configs,
    #         #     result_path=result_path,
    #         #     eval_set=eval_set_map[self.version],
    #         #     output_dir=output_dir,
    #         #     verbose=True,
    #         # )
    #         # nusc_eval.main(render_curves=False)

    #         from unittest.mock import patch
            
    #         # 현재 nusc에 로드된 모든 씬 이름을 가져옵니다.
    #         all_scene_names = [s['name'] for s in nusc.scene]
    #         print(f"DEBUG: Found {len(all_scene_names)} scenes in dataset.") # 디버깅용 출력

    #         def get_custom_splits(verbose=False):
    #             print("DEBUG: get_custom_splits called!") # 패치 작동 확인용
    #             return {
    #                 'val': all_scene_names,
    #                 'mini_val': all_scene_names,
    #                 'train': all_scene_names,
    #                 'test': all_scene_names,
    #             }
            
    #         # [중요 변경] patch 경로를 loaders 모듈 쪽으로 변경 + add_new_gt_loading_patch
    #         # 혹시 모를 상황에 대비해 두 경로 모두 패치 시도 (중첩 with 문)
    #         with patch('nuscenes.eval.common.loaders.create_splits_scenes', side_effect=get_custom_splits):
    #             with patch('nuscenes.utils.splits.create_splits_scenes', side_effect=get_custom_splits):
    #                 nusc_eval = NuScenesEval(
    #                     nusc,
    #                     config=self.det3d_eval_configs,
    #                     result_path=result_path,
    #                     eval_set=eval_set_map.get(eval_version, "val"),
    #                     output_dir=output_dir,
    #                     verbose=True, # 디버깅을 위해 True로 변경 권장
    #                 )
    #                 nusc_eval.main(render_curves=False)

    #         # record metrics
    #         metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
    #         detail = dict()
    #         metric_prefix = f"{result_name}_NuScenes"
    #         for name in self.CLASSES:
    #             for k, v in metrics["label_aps"][name].items():
    #                 val = float("{:.4f}".format(v))
    #                 detail[
    #                     "{}/{}_AP_dist_{}".format(metric_prefix, name, k)
    #                 ] = val
    #             for k, v in metrics["label_tp_errors"][name].items():
    #                 val = float("{:.4f}".format(v))
    #                 detail["{}/{}_{}".format(metric_prefix, name, k)] = val
    #             for k, v in metrics["tp_errors"].items():
    #                 val = float("{:.4f}".format(v))
    #                 detail[
    #                     "{}/{}".format(metric_prefix, self.ErrNameMapping[k])
    #                 ] = val

    #         detail["{}/NDS".format(metric_prefix)] = metrics["nd_score"]
    #         detail["{}/mAP".format(metric_prefix)] = metrics["mean_ap"]
    #     else:
    #         from nuscenes.eval.tracking.evaluate import TrackingEval

    #         nusc_eval = TrackingEval(
    #             config=self.track3d_eval_configs,
    #             result_path=result_path,
    #             eval_set=eval_set_map[self.version],
    #             output_dir=output_dir,
    #             verbose=True,
    #             nusc_version=self.version,
    #             nusc_dataroot=self.data_root,
    #         )
    #         metrics = nusc_eval.main()

    #         # record metrics
    #         metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
    #         print(metrics)
    #         detail = dict()
    #         metric_prefix = f"{result_name}_NuScenes"
    #         keys = [
    #             "amota",
    #             "amotp",
    #             "recall",
    #             "motar",
    #             "gt",
    #             "mota",
    #             "motp",
    #             "mt",
    #             "ml",
    #             "faf",
    #             "tp",
    #             "fp",
    #             "fn",
    #             "ids",
    #             "frag",
    #             "tid",
    #             "lgd",
    #         ]
    #         for key in keys:
    #             detail["{}/{}".format(metric_prefix, key)] = metrics[key]

    #     return detail

    def _evaluate_single(
        self, result_path, logger=None, result_name="img_bbox", tracking=False
    ):
        from nuscenes import NuScenes
        from unittest.mock import patch
        import json
        
        # NuScenes related classes
        from nuscenes.eval.common.data_classes import EvalBoxes
        from nuscenes.eval.detection.data_classes import DetectionBox

        output_dir = osp.join(*osp.split(result_path)[:-1])
        
        eval_version = self.version
        if self.version == "v1.0-trainval_debug":
            eval_version = "v1.0-trainval" 

        nusc = NuScenes(
            version=eval_version, dataroot=self.data_root, verbose=True
        )

        # ----------------------------------------------------------------------
        # PROACTIVE FIX: Patch timestamps in memory if they are detected as indices
        # # ----------------------------------------------------------------------
        # if len(nusc.sample) > 0:
        #     # Check the last sample or a middle sample to see if timestamps are suspiciously small
        #     # (First sample might be 0, which is valid but small)
        #     check_idx = len(nusc.sample) // 2
        #     ts_check = nusc.sample[check_idx]['timestamp']
            
        #     # Normal NuScenes timestamps are epoch microseconds (~16 digits)
        #     # If we see small integers (like 100, 200), it's the index-based issue.
        #     # Using 1e12 as a threshold (valid epoch us is > 1.6e15 approx)
        #     if ts_check < 1e12: 
        #         print(f"DEBUG: Detected non-epoch timestamps (val={ts_check}). Applying in-memory x100,000 patch for Evaluation.")
        #         for s in nusc.sample:
        #             s['timestamp'] = int(s['timestamp'] * 100000)
        # # ----------------------------------------------------------------------
        
        if len(nusc.sample) > 3:
            print(f"\n{'='*60}")
            print("TIMESTAMP CHECK:")
            print(f"{'='*60}")
            for i in [0, 1, 2, len(nusc.sample)//2]:
                sample = nusc.sample[i]
                ts = sample['timestamp']
                print(f"Sample[{i:3d}]: timestamp = {ts:15.0f} ({ts:.2e})")
            print(f"{'='*60}\n")

        with open(result_path, 'r') as f:
            pred_data = json.load(f)
        
        # prediction result tokens
        pred_tokens = set(pred_data['results'].keys())
        print(f"DEBUG: Pred tokens count: {len(pred_tokens)}")
        
        # Check if samples exist in nusc
        sample_tokens_in_nusc = set([s['token'] for s in nusc.sample])
        missing_tokens = pred_tokens - sample_tokens_in_nusc
        if len(missing_tokens) > 0:
            print(f"DEBUG: Found {len(missing_tokens)} tokens in prediction that are missing in loaded NuScenes DB.")
            print(f"DEBUG: First 5 missing: {list(missing_tokens)[:5]}")
            print(f"DEBUG: Nusc DB has {len(sample_tokens_in_nusc)} samples.")
            print(f"DEBUG: First 5 nusc samples: {list(sample_tokens_in_nusc)[:5]}")

        target_classes = ["car", "pedestrian"]

        target_classes = ["car", "pedestrian"]

        def custom_load_gt(nusc, eval_split, box_cls, verbose=False):
            print(f"DEBUG: custom_load_gt called! Filter classes: {target_classes}")
            gt_boxes = EvalBoxes()
            is_tracking = 'TrackingBox' in str(box_cls)

            # Velocity debugging counters
            vel_check_count = 0
            vel_samples = []

            for sample_token in pred_tokens:
                sample_boxes = []
                try:
                    sample = nusc.get('sample', sample_token)
                    for ann_token in sample['anns']:
                        ann = nusc.get('sample_annotation', ann_token)
                        
                        # V2X-Real: sample_annotation has no category_name/category_token.
                        # Must go through: instance_token -> instance.category_token -> category.name
                        if 'category_name' in ann:
                            category_name = ann['category_name']
                        elif 'category_token' in ann:
                            category_name = nusc.get('category', ann['category_token'])['name']
                        else:
                            instance = nusc.get('instance', ann['instance_token'])
                            category_name = nusc.get('category', instance['category_token'])['name']
                            
                        if category_name not in target_classes:
                            continue
                        
                        # Use nusc.box_velocity which computes diff based on timestamps
                        try:
                            velocity = nusc.box_velocity(ann_token)
                            velocity = (velocity[0], velocity[1])
                            
                            # Velocity debugging - 처음 10개만 출력
                            if vel_check_count < 10:
                                vel_mag = np.sqrt(velocity[0]**2 + velocity[1]**2)
                                vel_samples.append({
                                    'token': ann_token[:8],
                                    'class': category_name,
                                    'vel': velocity,
                                    'vel_mag': vel_mag
                                })
                                vel_check_count += 1
                                
                        except Exception:
                            velocity = (0.0, 0.0)
                        
                        if is_tracking:
                            box = box_cls(
                                sample_token=sample_token, translation=ann['translation'], size=ann['size'],
                                rotation=ann['rotation'], velocity=velocity, ego_translation=(0.0, 0.0, 0.0), 
                                num_pts=ann['num_lidar_pts'] + ann['num_radar_pts'],
                                tracking_id=ann['instance_token'], tracking_name=category_name, tracking_score=-1.0
                            )
                        else:
                            box = box_cls(
                                sample_token=sample_token, translation=ann['translation'], size=ann['size'],
                                rotation=ann['rotation'], velocity=velocity, ego_translation=(0.0, 0.0, 0.0), 
                                num_pts=ann['num_lidar_pts'] + ann['num_radar_pts'],
                                detection_name=category_name, detection_score=-1.0, attribute_name='' 
                            )
                        sample_boxes.append(box)
                
                    gt_boxes.add_boxes(sample_token, sample_boxes)

                except KeyError as e:
                    if verbose:
                        print(f"KeyError for sample {sample_token}: {e}")
                    # Even if error, add empty boxes to satisfy assertion
                    gt_boxes.add_boxes(sample_token, [])
                    pass
                except Exception as e:
                     if verbose:
                        print(f"Exception for sample {sample_token}: {e}")
                     # Even if error, add empty boxes to satisfy assertion
                     gt_boxes.add_boxes(sample_token, [])
                     pass
            
            # Print velocity samples
            if vel_samples:
                print(f"\n{'='*60}")
                print(f"VELOCITY CHECK (First {len(vel_samples)} boxes):")
                print(f"{'='*60}")
                for v in vel_samples:
                    print(f"Token: {v['token']} | Class: {v['class']:12s} | "
                          f"Vel: ({v['vel'][0]:7.2f}, {v['vel'][1]:7.2f}) m/s | "
                          f"Mag: {v['vel_mag']:6.2f} m/s")
                print(f"{'='*60}\n")
            
            total_gt_boxes = sum(len(v) for v in gt_boxes.boxes.values())
            print(f"DEBUG: Loaded {len(gt_boxes.boxes)} samples with GT, total {total_gt_boxes} boxes.")
            if total_gt_boxes > 0:
                # Print first non-empty sample's boxes
                for tok, boxes in gt_boxes.boxes.items():
                    if boxes:
                        print(f"DEBUG GT sample[0] boxes: {[(b.detection_name if hasattr(b,'detection_name') else b.tracking_name, b.num_pts, b.ego_translation) for b in boxes[:3]]}")
                        break
            else:
                print("DEBUG: WARNING - all GT boxes are empty!")
            return gt_boxes

        def patched_add_center_dist(nusc, eval_boxes):
            """add_center_dist that gracefully handles tokens missing from NuScenes DB.
            Also handles V2X-Real sensor keys like LIDAR_TOP_veh1, LIDAR_TOP_veh2, etc.
            Note: ego_dist is a read-only property computed from ego_translation,
            so we only set ego_translation (as relative vector: box pos - ego pos).
            """
            for sample_token in eval_boxes.sample_tokens:
                try:
                    sample_rec = nusc.get('sample', sample_token)
                    # Find the lidar sensor key: may be LIDAR_TOP or LIDAR_TOP_veh1, etc.
                    lidar_key = None
                    for key in sample_rec['data']:
                        if key.startswith('LIDAR_TOP'):
                            lidar_key = key
                            break
                    if lidar_key is None:
                        # No lidar key at all – fall back to ego at origin
                        for box in eval_boxes[sample_token]:
                            box.ego_translation = (0.0, 0.0, 0.0)
                        continue
                    sd_record = nusc.get('sample_data', sample_rec['data'][lidar_key])
                    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
                    for box in eval_boxes[sample_token]:
                        # ego_translation must be relative (box pos - ego pos);
                        # ego_dist is then auto-computed as its 2D magnitude.
                        box.ego_translation = (
                            box.translation[0] - pose_record['translation'][0],
                            box.translation[1] - pose_record['translation'][1],
                            box.translation[2] - pose_record['translation'][2],
                        )
                except KeyError:
                    # Token not in DB – place box at ego (dist=0) so it passes
                    # distance filter; GT for this token will be empty anyway.
                    for box in eval_boxes[sample_token]:
                        box.ego_translation = (0.0, 0.0, 0.0)
            return eval_boxes

        def patched_filter_eval_boxes(nusc, eval_boxes, max_dist, verbose=False):
            """filter_eval_boxes that gracefully handles tokens missing from NuScenes DB."""
            from nuscenes.eval.common.loaders import _get_box_class_field
            from nuscenes.utils.geometry_utils import points_in_box
            from pyquaternion import Quaternion as _Quat
            from nuscenes.utils.data_classes import Box as _Box

            # If there are no valid boxes (e.g. all GT tokens missing from DB),
            # _get_box_class_field will raise – nothing to filter, just return.
            try:
                class_field = _get_box_class_field(eval_boxes)
            except Exception:
                return eval_boxes
            total, dist_filter, point_filter, bike_rack_filter = 0, 0, 0, 0

            for ind, sample_token in enumerate(eval_boxes.sample_tokens):
                total += len(eval_boxes[sample_token])
                eval_boxes.boxes[sample_token] = [
                    box for box in eval_boxes[sample_token]
                    if box.ego_dist < max_dist[box.__getattribute__(class_field)]
                ]
                dist_filter += len(eval_boxes[sample_token])

                eval_boxes.boxes[sample_token] = [
                    box for box in eval_boxes[sample_token]
                    if not box.num_pts == 0
                ]
                point_filter += len(eval_boxes[sample_token])

                # Bike rack filtering – gracefully skip if token is not in DB
                try:
                    sample_anns = nusc.get('sample', sample_token)['anns']
                    bikerack_recs = [
                        nusc.get('sample_annotation', ann) for ann in sample_anns
                        if nusc.get('sample_annotation', ann)['category_name'] == 'static_object.bicycle_rack'
                    ]
                    bikerack_boxes = [
                        _Box(rec['translation'], rec['size'], _Quat(rec['rotation']))
                        for rec in bikerack_recs
                    ]
                    filtered_boxes = []
                    for box in eval_boxes[sample_token]:
                        if box.__getattribute__(class_field) in ['bicycle', 'motorcycle']:
                            in_a_bikerack = any(
                                np.sum(points_in_box(
                                    br, np.expand_dims(np.array(box.translation), axis=1)
                                )) > 0
                                for br in bikerack_boxes
                            )
                            if not in_a_bikerack:
                                filtered_boxes.append(box)
                        else:
                            filtered_boxes.append(box)
                    eval_boxes.boxes[sample_token] = filtered_boxes
                except KeyError:
                    pass  # Token not in DB, skip bike rack filtering

                bike_rack_filter += len(eval_boxes.boxes[sample_token])

            is_gt = any(getattr(b, 'detection_score', 1.0) == -1.0
                        for v in eval_boxes.boxes.values() for b in v[:1])
            label = 'GT' if is_gt else 'PRED'
            print(f"DEBUG filter [{label}]: total={total} -> dist={dist_filter} -> pts={point_filter} -> bikerack={bike_rack_filter}")
            if total > 0 and bike_rack_filter == 0:
                # Show a sample of what was removed
                print(f"  [{label}] WARNING: all boxes removed! max_dist={max_dist}")
            # Show a few sample boxes
            for tok, boxes in eval_boxes.boxes.items():
                if boxes:
                    b = boxes[0]
                    name = getattr(b, 'detection_name', getattr(b, 'tracking_name', '?'))
                    print(f"  [{label}] sample box: name={name} ego_dist={b.ego_dist:.1f} num_pts={b.num_pts}")
                    break
            return eval_boxes

        if not tracking:
            from nuscenes.eval.detection.evaluate import NuScenesEval
            # -----------------------------------------------------------------
            # 4. Modify Config to match Target Classes
            # -----------------------------------------------------------------
            # Overwrite class_names to only include the 4 classes
            self.det3d_eval_configs.class_names = target_classes 
            
            default_ranges = {
                "car": 50, 
                "pedestrian": 40, 
                # "bicycle": 40,
            }
            new_class_range = {}
            for cls_name in target_classes:
                new_class_range[cls_name] = default_ranges.get(cls_name, 50)
            
            self.det3d_eval_configs.class_range = new_class_range

            # -----------------------------------------------------------------
            # 5. Apply Patch and Run
            # -----------------------------------------------------------------
            # The model outputs tracking-format results (tracking_name/tracking_score).
            # NuScenesEval (detection) expects detection_name/detection_score.
            # Convert to detection format in a temp file.
            import tempfile
            det_result_path = result_path  # default: use as-is
            with open(result_path, 'r') as f:
                pred_data_raw = json.load(f)
            sample0 = next(iter(pred_data_raw['results'].values()), [])
            if sample0 and 'detection_name' not in sample0[0] and 'tracking_name' in sample0[0]:
                print("DEBUG: Converting tracking-format results to detection format for detection eval...")
                det_results = {}
                for tok, boxes in pred_data_raw['results'].items():
                    det_boxes = []
                    for b in boxes:
                        det_b = dict(b)
                        det_b['detection_name'] = b.get('tracking_name')
                        det_b['detection_score'] = b.get('tracking_score', 0.0)
                        if 'attribute_name' not in det_b:
                            det_b['attribute_name'] = ''
                        det_boxes.append(det_b)
                    det_results[tok] = det_boxes
                det_pred_data = {'results': det_results, 'meta': pred_data_raw.get('meta', {})}
                tmp_f = tempfile.NamedTemporaryFile(
                    mode='w', suffix='_det.json', dir=output_dir, delete=False
                )
                json.dump(det_pred_data, tmp_f)
                tmp_f.close()
                det_result_path = tmp_f.name
                print(f"DEBUG: Detection-format results written to {det_result_path}")

            with patch('nuscenes.eval.detection.evaluate.load_gt', side_effect=custom_load_gt):
              with patch('nuscenes.eval.detection.evaluate.add_center_dist', side_effect=patched_add_center_dist):
                with patch('nuscenes.eval.detection.evaluate.filter_eval_boxes', side_effect=patched_filter_eval_boxes):
                
                    nusc_eval = NuScenesEval(
                        nusc,
                        config=self.det3d_eval_configs,
                        result_path=det_result_path,
                        eval_set='val', 
                        output_dir=output_dir,
                        verbose=True,
                    )
                    
                    print(f"DEBUG: After init - GT Samples: {len(nusc_eval.gt_boxes.sample_tokens)}")
                    nusc_eval.main(render_curves=False)

            # --- Metrics Recording ---
            metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"
            
            # Record metrics only for available target classes
            for name in target_classes:
                if name in metrics["label_aps"]:
                    for k, v in metrics["label_aps"][name].items():
                        detail[f"{metric_prefix}/{name}_AP_dist_{k}"] = float("{:.4f}".format(v))
                if name in metrics["label_tp_errors"]:
                    for k, v in metrics["label_tp_errors"][name].items():
                        detail[f"{metric_prefix}/{name}_{k}"] = float("{:.4f}".format(v))
                        
            for k, v in metrics["tp_errors"].items():
                detail[f"{metric_prefix}/{self.ErrNameMapping[k]}"] = float("{:.4f}".format(v))

            detail["{}/NDS".format(metric_prefix)] = metrics["nd_score"]
            detail["{}/mAP".format(metric_prefix)] = metrics["mean_ap"]
        
        else:
            from nuscenes.eval.tracking.evaluate import TrackingEval
            
            # Filter config classes if applicable
            if hasattr(self.track3d_eval_configs, 'class_names'):
                 self.track3d_eval_configs.class_names = target_classes

            # ── Post-process: NMS + Hungarian re-tracking ──────────────────
            # SparseDrive with 6 cameras creates 1100+ unique IDs per scene
            # (should be ~20). Apply spatial NMS (2m) to deduplicate multi-camera
            # detections, then Hungarian tracking (4m) to maintain ID consistency.
            retrack_path = osp.join(osp.dirname(result_path), 'results_nusc_retracked.json')
            try:
                from scipy.optimize import linear_sum_assignment
                from sklearn.metrics.pairwise import euclidean_distances as _euc_dist
                from collections import defaultdict as _dd

                def _nms_frame(preds, nms_dist=2.0, min_score=None):
                    if min_score is not None:
                        preds = [p for p in preds if p.get('tracking_score', 0) >= min_score]
                    if len(preds) <= 1:
                        return list(preds)
                    by_cls = _dd(list)
                    for p in preds:
                        by_cls[p['tracking_name']].append(p)
                    kept = []
                    for cls_preds in by_cls.values():
                        cls_preds.sort(key=lambda p: -p['tracking_score'])
                        pos = np.array([p['translation'][:2] for p in cls_preds])
                        dists = _euc_dist(pos, pos)
                        supp = set()
                        for i in range(len(cls_preds)):
                            if i in supp: continue
                            kept.append(cls_preds[i])
                            for j in range(i+1, len(cls_preds)):
                                if j not in supp and dists[i,j] < nms_dist:
                                    supp.add(j)
                    return kept

                def _retrack_scene(sample_tokens, results, min_score, nms_dist, match_dist, next_id, dt=0.1):
                    """
                    Cross-frame velocity-predicted Hungarian tracking.
                    prev_tracks: int -> (x, y, vx, vy, class_name)
                    Predicted position at next frame: (x + vx*dt, y + vy*dt)
                    """
                    new_results = {}
                    prev_tracks = {}   # int -> (x, y, vx, vy, class_name)
                    for tok in sample_tokens:
                        preds = _nms_frame(results.get(tok, []), nms_dist=nms_dist, min_score=min_score)
                        new_preds = []
                        if prev_tracks and preds:
                            by_cls_c = _dd(list)
                            for p in preds: by_cls_c[p['tracking_name']].append(p)
                            by_cls_p = _dd(list)
                            for tid, (x, y, vx, vy, cls) in prev_tracks.items():
                                by_cls_p[cls].append((tid, x + vx * dt, y + vy * dt))
                            for cls in set(list(by_cls_c.keys()) + list(by_cls_p.keys())):
                                curr = by_cls_c.get(cls, [])
                                prev = by_cls_p.get(cls, [])
                                if curr and prev:
                                    cp = np.array([p['translation'][:2] for p in curr])
                                    pi = [t[0] for t in prev]
                                    pp = np.array([[t[1], t[2]] for t in prev])
                                    cost = np.where(_euc_dist(pp, cp) < match_dist, _euc_dist(pp, cp), 1e9)
                                    r_i, c_i = linear_sum_assignment(cost)
                                    asgn = {c: pi[r] for r, c in zip(r_i, c_i) if cost[r, c] < 1e9}
                                    for i, p in enumerate(curr):
                                        np_ = dict(p)
                                        vel = p.get('velocity') or (0.0, 0.0)
                                        vx_ = float(vel[0]) if not np.isnan(vel[0]) else 0.0
                                        vy_ = float(vel[1]) if not np.isnan(vel[1]) else 0.0
                                        if i in asgn:
                                            tid = asgn[i]
                                            np_['tracking_id'] = str(tid)
                                            prev_tracks[tid] = (p['translation'][0], p['translation'][1], vx_, vy_, cls)
                                        else:
                                            np_['tracking_id'] = str(next_id)
                                            prev_tracks[next_id] = (p['translation'][0], p['translation'][1], vx_, vy_, cls)
                                            next_id += 1
                                        new_preds.append(np_)
                                elif curr:
                                    for p in curr:
                                        vel = p.get('velocity') or (0.0, 0.0)
                                        vx_ = float(vel[0]) if not np.isnan(vel[0]) else 0.0
                                        vy_ = float(vel[1]) if not np.isnan(vel[1]) else 0.0
                                        np_ = dict(p); np_['tracking_id'] = str(next_id)
                                        prev_tracks[next_id] = (p['translation'][0], p['translation'][1], vx_, vy_, cls)
                                        next_id += 1; new_preds.append(np_)
                        else:
                            for p in preds:
                                vel = p.get('velocity') or (0.0, 0.0)
                                vx_ = float(vel[0]) if not np.isnan(vel[0]) else 0.0
                                vy_ = float(vel[1]) if not np.isnan(vel[1]) else 0.0
                                np_ = dict(p); np_['tracking_id'] = str(next_id)
                                prev_tracks[next_id] = (p['translation'][0], p['translation'][1], vx_, vy_, p['tracking_name'])
                                next_id += 1; new_preds.append(np_)
                        new_results[tok] = new_preds
                    return new_results, next_id

                with open(result_path) as _f:
                    _pred_data = json.load(_f)
                _pred_results = _pred_data['results']
                _pred_tokens_set = set(_pred_results.keys())

                # Build scene -> ordered sample_tokens
                _scene_order = {}
                for _scene in nusc.scene:
                    _toks = []
                    _s = nusc.get('sample', _scene['first_sample_token'])
                    while True:
                        if _s['token'] in _pred_tokens_set:
                            _toks.append(_s['token'])
                        if not _s['next']: break
                        _s = nusc.get('sample', _s['next'])
                    if _toks:
                        _scene_order[_scene['name']] = _toks

                _new_results = {}
                _next_id = 0
                _RT_MIN_SCORE = 0.15    # keep predictions with score >= this
                _RT_NMS_DIST  = 2.0     # spatial NMS radius (metres)
                _RT_MATCH_DIST = 3.0    # velocity-predicted Hungarian match radius (metres)
                for _scene_toks in _scene_order.values():
                    _scene_new, _next_id = _retrack_scene(
                        _scene_toks, _pred_results,
                        min_score=_RT_MIN_SCORE, nms_dist=_RT_NMS_DIST,
                        match_dist=_RT_MATCH_DIST, next_id=_next_id)
                    _new_results.update(_scene_new)
                # tokens not in any scene → empty
                for _tok in _pred_tokens_set - set(_new_results.keys()):
                    _new_results[_tok] = []

                _n_preds  = sum(len(v) for v in _new_results.values())
                _n_car_ids = len(set(b['tracking_id'] for v in _new_results.values() for b in v if b.get('tracking_name')=='car'))
                print(f"DEBUG retrack: {_n_preds} total preds, {_n_car_ids} unique car IDs after NMS+retrack")
                with open(retrack_path, 'w') as _f:
                    json.dump({'meta': _pred_data['meta'], 'results': _new_results}, _f)
                result_path = retrack_path   # use re-tracked file for eval
                print(f"DEBUG retrack: saved to {retrack_path}")
            except Exception as _rt_err:
                print(f"WARNING: retrack failed ({_rt_err}), using original predictions")
            # ── End re-tracking ─────────────────────────────────────────────

            # Patch NuScenes class AND load_gt
            # with patch('nuscenes.eval.tracking.evaluate.NuScenes', side_effect=PatchedNuScenes):
            with patch('nuscenes.eval.tracking.evaluate.load_gt', side_effect=custom_load_gt):
              with patch('nuscenes.eval.tracking.evaluate.add_center_dist', side_effect=patched_add_center_dist):
                with patch('nuscenes.eval.tracking.evaluate.filter_eval_boxes', side_effect=patched_filter_eval_boxes):
                    nusc_eval = TrackingEval(
                        config=self.track3d_eval_configs,
                        result_path=result_path,
                        eval_set='val',
                        output_dir=output_dir,
                        verbose=True,
                        nusc_version=self.version,
                        nusc_dataroot=self.data_root,
                    )
                    metrics = nusc_eval.main()
            
            # metrics recording (생략)
            metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"
            keys = [ "amota", "amotp", "recall", "motar", "gt", "mota", "motp", "mt", "ml", "faf", "tp", "fp", "fn", "ids", "frag", "tid", "lgd"]
            for key in keys:
                detail["{}/{}".format(metric_prefix, key)] = metrics[key]

        return detail

    # def format_results(self, results, jsonfile_prefix=None, tracking=False):
    #     assert isinstance(results, list), "results must be a list"

    #     if jsonfile_prefix is None:
    #         tmp_dir = tempfile.TemporaryDirectory()
    #         jsonfile_prefix = osp.join(tmp_dir.name, "results")
    #     else:
    #         tmp_dir = None

    #     if not ("pts_bbox" in results[0] or "img_bbox" in results[0]):
    #         result_files = self._format_bbox(
    #             results, jsonfile_prefix, tracking=tracking
    #         )
    #     else:
    #         result_files = dict()
    #         for name in results[0]:
    #             print(f"\nFormating bboxes of {name}")
    #             results_ = [out[name] for out in results]
    #             tmp_file_ = jsonfile_prefix
    #             result_files.update(
    #                 {
    #                     name: self._format_bbox(
    #                         results_, tmp_file_, tracking=tracking
    #                     )
    #                 }
    #             )
    #     return result_files, tmp_dir

    def format_results(self, results, jsonfile_prefix=None, tracking=False):
        assert isinstance(results, list), "results must be a list"

        # 결과 저장할 경로 설정 (없으면 임시 폴더 생성)
        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        # 다중 모달리티(img_bbox, pts_bbox 등) 결과 처리 로직
        if not ("pts_bbox" in results[0] or "img_bbox" in results[0]):
            # 일반적인 단일 결과 처리
            result_files = self._format_bbox(
                results, jsonfile_prefix, tracking=tracking
            )
        else:
            # 키 값(예: 'img_bbox')이 있는 dict 형태의 결과 처리
            result_files = dict()
            for name in results[0]:
                print(f"\nFormating bboxes of {name}")
                results_ = [out[name] for out in results]
                tmp_file_ = jsonfile_prefix
                result_files.update(
                    {
                        name: self._format_bbox(
                            results_, tmp_file_, tracking=tracking
                        )
                    }
                )
        
        # 임시 폴더 사용 시 디렉토리 객체 반환 (나중에 cleanup 가능하도록)
        return result_files, tmp_dir

    def format_map_results(self, results, prefix=None):
        submissions = {'results': {},}
        
        for j, pred in enumerate(results):
            '''
            For each case, the result should be formatted as Dict{'vectors': [], 'scores': [], 'labels': []}
            'vectors': List of vector, each vector is a array([[x1, y1], [x2, y2] ...]),
                contain all vectors predicted in this sample.
            'scores: List of score(float), 
                contain scores of all instances in this sample.
            'labels': List of label(int), 
                contain labels of all instances in this sample.
            '''
            if pred is None: # empty prediction
                continue
            pred = pred['img_bbox']

            single_case = {'vectors': [], 'scores': [], 'labels': []}
            token = self.data_infos[j]['token']
            for i in range(len(pred['scores'])):
                score = pred['scores'][i]
                label = pred['labels'][i]
                vector = pred['vectors'][i]

                # A line should have >=2 points
                if len(vector) < 2:
                    continue
                
                single_case['vectors'].append(vector)
                single_case['scores'].append(score)
                single_case['labels'].append(label)
            
            submissions['results'][token] = single_case
        
        out_path = osp.join(prefix, 'submission_vector.json')
        print(f'saving submissions results to {out_path}')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mmcv.dump(submissions, out_path)
        return out_path

    def format_motion_results(self, results, jsonfile_prefix=None, tracking=False, thresh=None):
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            # boxes_lidar: NuScenesBox objects in Lidar Frame
            boxes_lidar = output_to_nusc_box(
                det['img_bbox'], threshold=None
            )
            sample_token = self.data_infos[sample_id]["token"]
            
            # boxes_global: Converted to Global Frame for reporting
            boxes_global = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                copy.deepcopy(boxes_lidar), # Use deepcopy to preserve boxes_lidar
                mapped_class_names,
                self.det3d_eval_configs,
                self.det3d_eval_version,
                filter_with_cls_range=False,
            )
            
            for i, box in enumerate(boxes_global):
                if thresh is not None and box.score < thresh:
                    continue
                name = mapped_class_names[box.label]
                if tracking and name in [
                    "barrier",
                    "traffic_cone",
                    "construction_vehicle",
                ]:
                    continue
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = NuScenes3DDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = NuScenes3DDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                )
                if not tracking:
                    nusc_anno.update(
                        dict(
                            detection_name=name,
                            detection_score=box.score,
                            attribute_name=attr,
                        )
                    )
                else:
                    nusc_anno.update(
                        dict(
                            tracking_name=name,
                            tracking_score=box.score,
                            tracking_id=str(box.token),
                        )
                    )
                
                # Retrieve the corresponding Lidar Frame box and trajectory
                box_lidar = boxes_lidar[i] 
                trajs_lidar = det['img_bbox']['trajs_3d'][i].numpy() # shape (Modes, Steps, 2)
                
                # Convert Lidar Frame Trajectory -> Agent Frame Trajectory
                # 1. Translate relative to agent center (Lidar Frame)
                trajs_diff = trajs_lidar - box_lidar.center[:2]
                
                # 2. Rotate to align with agent heading using inverse orientation
                # box_lidar.orientation is a Quaternion. 
                # We need to rotate the 'vector' trajs_diff by the inverse of the box orientation.
                # However, box.orientation is 3D. We project to 2D for simplicity or use the quat inverse.
                # Since trajs are 2D (x,y), we pad to 3D, rotate, then slice back.
                
                # Reshape for broadcast: (Modes * Steps, 3)
                num_modes, num_steps, _ = trajs_lidar.shape
                trajs_diff_flat = np.zeros((num_modes * num_steps, 3))
                trajs_diff_flat[:, :2] = trajs_diff.reshape(-1, 2)
                
                # Rotate: v_local = R_inv * v_global
                # box_lidar.orientation.inverse allows rotating from Global(Lidar) back to Local(Agent body)
                # Note: 'Global' here means 'Lidar Frame' because box_lidar is in Lidar Frame.
                trajs_agent_flat = np.dot(trajs_diff_flat, box_lidar.orientation.rotation_matrix)
                
                trajs_agent = trajs_agent_flat[:, :2].reshape(num_modes, num_steps, 2)

                nusc_anno.update(
                    dict(
                        trajs=trajs_agent,
                    )
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        return nusc_submissions 

    def _evaluate_single_motion(self,
                         results,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from .evaluation.motion.motion_eval_uniad import NuScenesEval as NuScenesEvalMotion
        from .evaluation.motion.motion_utils import MotionBox, motion_name_mapping
        from nuscenes.eval.common.data_classes import EvalBoxes
        from nuscenes.eval.detection.utils import category_to_detection_name
        from nuscenes.prediction import PredictHelper
        from unittest.mock import patch
        import tqdm

        output_dir = result_path
        
        eval_version = self.version
        if self.version == "v1.0-trainval_debug":
            eval_version = "v1.0-trainval"

        nusc = NuScenes(
            version=eval_version, dataroot=self.data_root, verbose=False)
        
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }

        # Get pred tokens
        pred_tokens = set(results['results'].keys())

        def custom_load_motion_gt(nusc, eval_split, box_cls, verbose=False, seconds=6):
            print(f"DEBUG: custom_load_motion_gt called. Loaded {len(pred_tokens)} pred tokens.")
            predict_helper = PredictHelper(nusc)
            if box_cls == MotionBox:
                attribute_map = {a['token']: a['name'] for a in nusc.attribute}

            all_annotations = EvalBoxes()

            # Iterate over pred_tokens only
            for sample_token in tqdm.tqdm(pred_tokens, leave=verbose):
                try:
                    sample = nusc.get('sample', sample_token)
                except KeyError:
                    print(f"Skipping token not in nusc: {sample_token}")
                    continue

                sample_annotation_tokens = sample['anns']

                sample_boxes = []
                for sample_annotation_token in sample_annotation_tokens:
                    sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
                    
                    # detection_name = category_to_detection_name(sample_annotation['category_name'])
                    # if detection_name in motion_name_mapping:
                    #     detection_name = motion_name_mapping[detection_name]
                    
                    # V2X-Real: sample_annotation has no category_name/category_token.
                    # Must go through: instance_token -> instance.category_token -> category.name
                    if 'category_name' in sample_annotation:
                        category_name = sample_annotation['category_name']
                    elif 'category_token' in sample_annotation:
                        category_name = nusc.get('category', sample_annotation['category_token'])['name']
                    else:
                        instance = nusc.get('instance', sample_annotation['instance_token'])
                        category_name = nusc.get('category', instance['category_token'])['name']
                    
                    detection_name = category_name
                    if detection_name not in ['car', 'pedestrian']:
                        # Try standard mapping as fallback
                        detection_name = category_to_detection_name(category_name)
                        if detection_name in motion_name_mapping:
                             detection_name = motion_name_mapping[detection_name]
                    
                    if detection_name is None:
                        continue
                    
                    if detection_name not in ['car', 'pedestrian', 'bicycle', 'motorcycle', 'bus', 'truck', 'trailer', 'construction_vehicle', 'traffic_cone', 'barrier']:
                         continue

                    # Get attribute_name.
                    attr_tokens = sample_annotation['attribute_tokens']
                    attr_count = len(attr_tokens)
                    if attr_count == 0:
                        attribute_name = ''
                    elif attr_count == 1:
                        attribute_name = attribute_map[attr_tokens[0]]
                    else:
                        raise Exception('Error: GT annotations must not have more than one attribute!')

                    # get future trajs
                    instance_token = nusc.get('sample_annotation', sample_annotation['token'])['instance_token']
                    fut_traj_local = predict_helper.get_future_for_agent(
                        instance_token, 
                        sample_token, 
                        seconds=seconds, 
                        in_agent_frame=True
                    )
                    
                    try:
                        velocity = nusc.box_velocity(sample_annotation['token'])
                        velocity = (velocity[0], velocity[1])
                    except Exception:
                        velocity = (0.0, 0.0)

                    box = MotionBox(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=velocity,
                        ego_translation=(0.0, 0.0, 0.0),
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        detection_name=detection_name,
                        detection_score=-1.0,
                        attribute_name=attribute_name,
                        traj=fut_traj_local
                    )
                    sample_boxes.append(box)

                all_annotations.add_boxes(sample_token, sample_boxes)

            return all_annotations

        def patched_add_center_dist_motion(nusc, eval_boxes):
            """add_center_dist for motion eval: handles missing tokens and
            V2X-Real sensor keys like LIDAR_TOP_veh1, LIDAR_TOP_veh2, etc."""
            for sample_token in eval_boxes.sample_tokens:
                try:
                    sample_rec = nusc.get('sample', sample_token)
                    lidar_key = next(
                        (k for k in sample_rec['data'] if k.startswith('LIDAR_TOP')),
                        None
                    )
                    if lidar_key is None:
                        for box in eval_boxes[sample_token]:
                            box.ego_translation = (0.0, 0.0, 0.0)
                        continue
                    sd_record = nusc.get('sample_data', sample_rec['data'][lidar_key])
                    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
                    for box in eval_boxes[sample_token]:
                        box.ego_translation = (
                            box.translation[0] - pose_record['translation'][0],
                            box.translation[1] - pose_record['translation'][1],
                            box.translation[2] - pose_record['translation'][2],
                        )
                except KeyError:
                    for box in eval_boxes[sample_token]:
                        box.ego_translation = (0.0, 0.0, 0.0)
            return eval_boxes

        with patch('projects.mmdet3d_plugin.datasets.evaluation.motion.motion_eval_uniad.load_gt', side_effect=custom_load_motion_gt):
          with patch('projects.mmdet3d_plugin.datasets.evaluation.motion.motion_eval_uniad.add_center_dist', side_effect=patched_add_center_dist_motion):
            nusc_eval = NuScenesEvalMotion(
                nusc,
                config=copy.deepcopy(self.det3d_eval_configs),
                result_path=results,
                eval_set=eval_set_map.get(eval_version, "val"),
                output_dir=output_dir,
                verbose=False,
                seconds=6)
            metrics = nusc_eval.main(render_curves=False)
        
        MOTION_METRICS = ['EPA', 'min_ade_err', 'min_fde_err', 'miss_rate_err']
        class_names = ['car', 'pedestrian']

        table = prettytable.PrettyTable()
        table.field_names = ["class names"] + MOTION_METRICS
        for class_name in class_names:
            row_data = [class_name]
            for m in MOTION_METRICS:
                row_data.append('%.4f' % metrics[f'{class_name}_{m}'])
            table.add_row(row_data)
        print_log('\n'+str(table), logger=logger)
        return metrics

    def evaluate(
        self,
        results,
        eval_mode,
        metric=None,
        logger=None,
        jsonfile_prefix=None,
        result_names=["img_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        res_path = "results.pkl" if "trainval" in self.version else "results_mini.pkl"
        res_path = osp.join(self.work_dir, res_path)
        print('All Results write to', res_path)
        mmcv.dump(results, res_path)

        results_dict = dict()
        if eval_mode['with_det']:
            self.tracking = eval_mode["with_tracking"]
            self.tracking_threshold = eval_mode["tracking_threshold"]
            for metric in ["detection", "tracking"]:
                tracking = metric == "tracking"
                if tracking and not self.tracking:
                    continue
                result_files, tmp_dir = self.format_results(
                    results, jsonfile_prefix=self.work_dir, tracking=tracking
                )

                if isinstance(result_files, dict):
                    for name in result_names:
                        ret_dict = self._evaluate_single(
                            result_files[name], tracking=tracking
                        )
                    results_dict.update(ret_dict)
                elif isinstance(result_files, str):
                    ret_dict = self._evaluate_single(
                        result_files, tracking=tracking
                    )
                    results_dict.update(ret_dict)

                if tmp_dir is not None:
                    tmp_dir.cleanup()

        if eval_mode['with_map']:
            from .evaluation.map.vector_eval import VectorEvaluate
            self.map_evaluator = VectorEvaluate(self.eval_config)
            result_path = self.format_map_results(results, prefix=self.work_dir)
            map_results_dict = self.map_evaluator.evaluate(result_path, logger=logger)
            results_dict.update(map_results_dict)

        if eval_mode['with_motion']:
            thresh = eval_mode["motion_threshhold"]
            result_files = self.format_motion_results(results, jsonfile_prefix=self.work_dir, thresh=thresh)
            motion_results_dict = self._evaluate_single_motion(result_files, self.work_dir, logger=logger)
            results_dict.update(motion_results_dict)
        
        if eval_mode['with_planning']:
            from .evaluation.planning.planning_eval import planning_eval
            planning_results_dict = planning_eval(results, self.eval_config, logger=logger)
            results_dict.update(planning_results_dict)

        if show or out_dir:
            self.show(results, save_dir=out_dir, show=show, pipeline=pipeline)
        
        # print main metrics for recording
        metric_str = '\n'
        if "img_bbox_NuScenes/NDS" in results_dict:
            metric_str += f'mAP: {results_dict.get("img_bbox_NuScenes/mAP"):.4f}\n'
            metric_str += f'mATE: {results_dict.get("img_bbox_NuScenes/mATE"):.4f}\n'
            metric_str += f'mASE: {results_dict.get("img_bbox_NuScenes/mASE"):.4f}\n'
            metric_str += f'mAOE: {results_dict.get("img_bbox_NuScenes/mAOE"):.4f}\n' 
            metric_str += f'mAVE: {results_dict.get("img_bbox_NuScenes/mAVE"):.4f}\n' 
            metric_str += f'mAAE: {results_dict.get("img_bbox_NuScenes/mAAE"):.4f}\n' 
            metric_str += f'NDS: {results_dict.get("img_bbox_NuScenes/NDS"):.4f}\n\n'
        
        if "img_bbox_NuScenes/amota" in results_dict:
            metric_str += f'AMOTA: {results_dict["img_bbox_NuScenes/amota"]:.4f}\n' 
            metric_str += f'AMOTP: {results_dict["img_bbox_NuScenes/amotp"]:.4f}\n' 
            metric_str += f'RECALL: {results_dict["img_bbox_NuScenes/recall"]:.4f}\n' 
            metric_str += f'MOTAR: {results_dict["img_bbox_NuScenes/motar"]:.4f}\n' 
            metric_str += f'MOTA: {results_dict["img_bbox_NuScenes/mota"]:.4f}\n' 
            metric_str += f'MOTP: {results_dict["img_bbox_NuScenes/motp"]:.4f}\n' 
            metric_str += f'IDS: {results_dict["img_bbox_NuScenes/ids"]}\n\n' 
        
        if "mAP_normal" in results_dict:
            metric_str += f'ped_crossing= {results_dict.get("ped_crossing", 0.0):.4f}\n' 
            metric_str += f'lane= {results_dict.get("lane", 0.0):.4f}\n' 
            metric_str += f'divider= {results_dict.get("divider", 0.0):.4f}\n' 
            metric_str += f'boundary= {results_dict.get("boundary", 0.0):.4f}\n' 
            metric_str += f'intersection= {results_dict.get("intersection", 0.0):.4f}\n' 
            metric_str += f'mAP_normal= {results_dict["mAP_normal"]:.4f}\n\n' 

        if "car_EPA" in results_dict:
            metric_str += f'Car / Ped\n' 
            metric_str += f'epa= {results_dict["car_EPA"]:.4f} / {results_dict["pedestrian_EPA"]:.4f}\n'
            metric_str += f'ade= {results_dict["car_min_ade_err"]:.4f} / {results_dict["pedestrian_min_ade_err"]:.4f}\n'
            metric_str += f'fde= {results_dict["car_min_fde_err"]:.4f} / {results_dict["pedestrian_min_fde_err"]:.4f}\n'
            metric_str += f'mr= {results_dict["car_miss_rate_err"]:.4f} / {results_dict["pedestrian_miss_rate_err"]:.4f}\n\n' 

        if "L2" in results_dict:
            metric_str += f'obj_box_col: {(results_dict["obj_box_col"]*100):.3f}%\n'
            metric_str += f'L2: {results_dict["L2"]:.4f}\n\n'
        
        print_log(metric_str, logger=logger)
        return results_dict

    def show(self, results, save_dir=None, show=False, pipeline=None):
        save_dir = "./" if save_dir is None else save_dir
        save_dir = os.path.join(save_dir, "visual")
        print_log(os.path.abspath(save_dir))
        pipeline = Compose(pipeline)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        videoWriter = None

        for i, result in enumerate(results):
            if "img_bbox" in result.keys():
                result = result["img_bbox"]
            data_info = pipeline(self.get_data_info(i))
            imgs = []

            raw_imgs = data_info["img"]
            lidar2img = data_info["img_metas"].data["lidar2img"]
            pred_bboxes_3d = result["boxes_3d"][
                result["scores_3d"] > self.vis_score_threshold
            ]
            if "instance_ids" in result and self.tracking:
                color = []
                for id in result["instance_ids"].cpu().numpy().tolist():
                    color.append(
                        self.ID_COLOR_MAP[int(id % len(self.ID_COLOR_MAP))]
                    )
            elif "labels_3d" in result:
                color = []
                for id in result["labels_3d"].cpu().numpy().tolist():
                    color.append(self.ID_COLOR_MAP[id])
            else:
                color = (255, 0, 0)

            # ===== draw boxes_3d to images =====
            for j, img_origin in enumerate(raw_imgs):
                img = img_origin.copy()
                if len(pred_bboxes_3d) != 0:
                    img = draw_lidar_bbox3d_on_img(
                        pred_bboxes_3d,
                        img,
                        lidar2img[j],
                        img_metas=None,
                        color=color,
                        thickness=3,
                    )
                imgs.append(img)

            # ===== draw boxes_3d to BEV =====
            bev = draw_lidar_bbox3d_on_bev(
                pred_bboxes_3d,
                bev_size=img.shape[0] * 2,
                color=color,
            )

            # ===== put text and concat =====
            for j, name in enumerate(
                [
                    "front",
                    "front right",
                    "front left",
                    "rear",
                    "rear left",
                    "rear right",
                ]
            ):
                imgs[j] = cv2.rectangle(
                    imgs[j],
                    (0, 0),
                    (440, 80),
                    color=(255, 255, 255),
                    thickness=-1,
                )
                w, h = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                text_x = int(220 - w / 2)
                text_y = int(40 + h / 2)

                imgs[j] = cv2.putText(
                    imgs[j],
                    name,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            image = np.concatenate(
                [
                    np.concatenate([imgs[2], imgs[0], imgs[1]], axis=1),
                    np.concatenate([imgs[5], imgs[3], imgs[4]], axis=1),
                ],
                axis=0,
            )
            image = np.concatenate([image, bev], axis=1)

            # ===== save video =====
            if videoWriter is None:
                videoWriter = cv2.VideoWriter(
                    os.path.join(save_dir, "video.avi"),
                    fourcc,
                    7,
                    image.shape[:2][::-1],
                )
            cv2.imwrite(os.path.join(save_dir, f"{i}.jpg"), image)
            videoWriter.write(image)
        videoWriter.release()


def output_to_nusc_box(detection, threshold=None):
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()
    if "instance_ids" in detection:
        ids = detection["instance_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().numpy()
    if threshold is not None:
        if "cls_scores" in detection:
            mask = detection["cls_scores"].numpy() >= threshold
        else:
            mask = scores >= threshold
        box3d = box3d[mask]
        scores = scores[mask]
        labels = labels[mask]
        ids = ids[mask]

    if hasattr(box3d, "gravity_center"):
        box_gravity_center = box3d.gravity_center.numpy()
        box_dims = box3d.dims.numpy()
        nus_box_dims = box_dims[:, [1, 0, 2]]
        box_yaw = box3d.yaw.numpy()
    else:
        box3d = box3d.numpy()
        box_gravity_center = box3d[..., :3].copy()
        box_dims = box3d[..., 3:6].copy()
        nus_box_dims = box_dims[..., [1, 0, 2]]
        box_yaw = box3d[..., 6].copy()

    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    # box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if hasattr(box3d, "gravity_center"):
            if box3d.tensor.shape[1] >= 9:
                velocity = (*box3d.tensor[i, 7:9], 0.0)
            else:
                velocity = (0.0, 0.0, 0.0)
        else:
            if box3d.shape[1] >= 9:
                velocity = (*box3d[i, 7:9], 0.0)
            else:
                velocity = (0.0, 0.0, 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        if "instance_ids" in detection:
            box.token = ids[i]
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(
    info,
    boxes,
    classes,
    eval_configs,
    eval_version="detection_cvpr_2019",
    filter_with_cls_range=True,
):
    box_list = []
    for i, box in enumerate(boxes):
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))
        # filter det in ego.
        if filter_with_cls_range:
            cls_range_map = eval_configs.class_range
            radius = np.linalg.norm(box.center[:2], 2)
            det_range = cls_range_map[classes[box.label]]
            if radius > det_range:
                continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list


def get_T_global(info):
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = pyquaternion.Quaternion(
        info["lidar2ego_rotation"]
    ).rotation_matrix
    lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])
    ego2global = np.eye(4)
    ego2global[:3, :3] = pyquaternion.Quaternion(
        info["ego2global_rotation"]
    ).rotation_matrix
    ego2global[:3, 3] = np.array(info["ego2global_translation"])
    return ego2global @ lidar2ego