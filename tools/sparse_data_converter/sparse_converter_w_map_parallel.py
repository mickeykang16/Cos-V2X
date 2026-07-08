import os
import sys
import math
import copy
import argparse
from os import path as osp
from collections import OrderedDict
from typing import List, Tuple, Union
import functools
import traceback

import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

import mmcv

from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global
# from visualization import line_visualization
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# from projects.mmdet3d_plugin.datasets.map_utils.nuscmap_extractor import NuscMapExtractor
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../SparseDrive')))
# from projects.mmdet3d_plugin.datasets.map_utils.nuscmap_extractor import NuscMapExtractor
# from projects.mmdet3d_plugin.datasets.map_utils.utils import split_collections
##### hm 수정 Sparsedrive 파일을 복사해서, univ2x_real 폴더로 가져오고, 그걸 부르도록 수정.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from projects.mmdet3d_plugin.datasets.sparse_map_utils.nuscmap_extractor import NuscMapExtractor
from projects.mmdet3d_plugin.datasets.sparse_map_utils.utils import split_collections
import os

# NameMapping = {
#     "movable_object.barrier": "barrier",
#     "vehicle.bicycle": "bicycle",
#     "vehicle.bus.bendy": "bus",
#     "vehicle.bus.rigid": "bus",
#     "vehicle.car": "car",
#     "vehicle.construction": "construction_vehicle",
#     "vehicle.motorcycle": "motorcycle",
#     "human.pedestrian.adult": "pedestrian",
#     "human.pedestrian.child": "pedestrian",
#     "human.pedestrian.construction_worker": "pedestrian",
#     "human.pedestrian.police_officer": "pedestrian",
#     "movable_object.trafficcone": "traffic_cone",
#     "vehicle.trailer": "trailer",
#     "vehicle.truck": "truck",
# }

NameMapping = {
    "movable_object.barrier": "barrier",
    "bicycle": "car",
    "vehicle.bicycle": "car",
    "vehicle.bus.bendy": "car",
    "vehicle.bus.rigid": "car",
    "car": "car",
    "vehicle.car": "car",
    "vehicle.construction": "car",
    "vehicle.motorcycle": "car",
    "pedestrian": "pedestrian",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "car",
    "vehicle.truck": "car",
}

# --- GLOBAL VARIABLES FOR MULTIPROCESSING ---
G_nusc = None
G_nusc_infra = None
G_nusc_map_extractor = None
G_predict_helper = None
G_args = {}
G_infra_stats = {'train': {'matched': 0, 'missing': 0}, 'val': {'matched': 0, 'missing': 0}, 'test': {'matched': 0, 'missing': 0}}

def init_globals(nusc, nusc_infra, nusc_map_extractor, args):
    global G_nusc, G_nusc_infra, G_nusc_map_extractor, G_predict_helper, G_args, G_infra_stats
    G_nusc = nusc
    G_nusc_infra = nusc_infra
    G_nusc_map_extractor = nusc_map_extractor
    G_predict_helper = PredictHelper(nusc)
    G_args = args
    G_infra_stats = {'train': {'matched': 0, 'missing': 0}, 'val': {'matched': 0, 'missing': 0}, 'test': {'matched': 0, 'missing': 0}}

def rot_to_quat_wxyz(rot):
    """rot: 3x3 matrix or (w,x,y,z) -> np.array([w,x,y,z])"""
    rot = np.array(rot, dtype=np.float64)
    if rot.shape == (4,):
        q = Quaternion(rot)                 # already wxyz
    elif rot.shape == (3, 3):
        q = Quaternion(matrix=rot)          # from rotation matrix
    elif rot.shape == (9,):
        q = Quaternion(matrix=rot.reshape(3, 3))
    else:
        raise ValueError(f"Unsupported rot format: shape={rot.shape}, rot={rot}")
    return np.array(list(q), dtype=np.float64)

def quart_to_rpy(qua):
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw

def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i

def geom2anno(map_geoms):
    MAP_CLASSES = (
        'intersection',
        'lane',
        'ped_crossing',
    )
    vectors = {}
    for cls, geom_list in map_geoms.items():
        if cls in MAP_CLASSES:
            label = MAP_CLASSES.index(cls)
            vectors[label] = []
            for geom in geom_list:
                line = np.array(geom.coords)
                vectors[label].append(line)
        
    return vectors

def geom2anno2(map_geoms):
    MAP_CLASSES = (
        'ped_crossing',
        'divider',
        'boundary',
        'drivable_area',
        'lane',
    )
    vectors = {}
    for cls, geom_list in map_geoms.items():
        # breakpoint()
        if cls in MAP_CLASSES:
            label = MAP_CLASSES.index(cls)
            vectors[label] = []
            for geom in geom_list:
                if (cls == 'drivable_area'):
                    line = np.array(geom.exterior.coords)
                    vectors[label].append(line)
                elif cls == 'lane':
                    continue
                else:
                    line = np.array(geom.coords)
                    vectors[label].append(line)
                # elif cls == 'drivable_area' and cls == ''
                #     line = np.array(geom.exterior.coords)
                #     vectors[label].append(line)
    return vectors

def create_nuscenes_infos(root_path,
                          out_path,
                          can_bus_root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10,
                          roi_size=(30, 60),
                          infra_root_path='datasets/v2xreal/infrastructure-side/'):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    print(version, root_path)
    # ---- 핵심 수정: v1.0-test를 강제로 trainval로 매핑 ----
    test = ('test' in version)
    # test = True
    # print("Test ver")
    version = 'v1.0-trainval' if test else version
    if test: print("Test ver")

    # breakpoint()
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    print(f"Vehicle-side: {len(nusc.scene)} scenes, {len(nusc.sample)} samples")
    
    nusc_infra = None
    if infra_root_path and os.path.exists(infra_root_path):
        print(f"Loading infrastructure data from {infra_root_path}")
        try:
            nusc_infra = NuScenes(version=version, dataroot=infra_root_path, verbose=True)
            print(f"Infrastructure-side: {len(nusc_infra.scene)} scenes, {len(nusc_infra.sample)} samples")
            
            # Check token overlap
            vehicle_tokens = set(s['token'] for s in nusc.sample)
            infra_tokens = set(s['token'] for s in nusc_infra.sample)
            common_tokens = vehicle_tokens & infra_tokens
            print(f"Token overlap: {len(common_tokens)}/{len(vehicle_tokens)} vehicle samples have matching infra samples")
            
        except Exception as e:
            print(f"Warning: Failed to load infrastructure NuScenes: {e}")
    
    nusc_map_extractor = NuscMapExtractor(root_path, roi_size)
    # nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini', 'v1.0-trainval_debug']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
        out_path = osp.join(out_path, 'mini')
    elif version == 'v1.0-trainval_debug':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val

    else:
        raise ValueError('unknown')
    os.makedirs(out_path, exist_ok=True)
    # breakpoint()
    
    # filter existing scenes.
    # available_scenes = get_available_scenes(nusc)
    # available_scene_names = [s['name'] for s in available_scenes] 

    # train_scenes = list(
    #     filter(lambda x: x in available_scene_names, train_scenes))
    # val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    # train_scenes = set([
    #     available_scenes[available_scene_names.index(s)]['token']
    #     for s in train_scenes
    # ])
    # val_scenes = set([
    #     available_scenes[available_scene_names.index(s)]['token']
    #     for s in val_scenes
    # ])
    # NOTE 추후 수정
    import json

    # json_path = "/home/user/nvme1/v2x/UniV2X_REAL/data/split_datas_V2XREAL/split_datas_V2XREAL.json"
    json_path = "data/split_datas_V2XREAL/split_datas_V2XREAL_coop.json"

    with open(json_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    # breakpoint()
    
    # test = 'test' in version

    # train_scenes = split['batch_split']['debug_train']
    # val_scenes = split['batch_split']['debug_val']
    # test = 'test' in version
    # test = True
    
    debug_mode = False # TODO: 기원 추가 (현재 서브셋으로 진행중임.)
    if debug_mode == True:
        if test:
            train_scenes = split['batch_split']['debug']
            print('test scene: {}'.format(len(train_scenes)))
        else:
            train_scenes = split['batch_split']['debug']
            val_scenes = split['batch_split']['debug']
            print('train scene: {}, val scene: {}'.format(
                len(train_scenes), len(val_scenes)))
    else:
        if test:
            train_scenes = split['batch_split']['test']
            print('test scene: {}'.format(len(train_scenes)))
        else:
            train_scenes = split['batch_split']['train']
            val_scenes = split['batch_split']['val']
            print('train scene: {}, val scene: {}'.format(
                len(train_scenes), len(val_scenes)))
    
    # Convert scene names to tokens
    name_to_token = {s['name']: s['token'] for s in nusc.scene}
    
    # Filter train_scenes
    train_scenes_tokens = set()
    found_count = 0
    
    # Pre-process nusc scenes for flexible matching
    # Map "base name" -> token
    # e.g. "2023-03-17-15-53-02_1_0_folder_1_-2" -> "2023-03-17-15-53-02_1_0"
    nusc_base_names = {}
    for s_name, s_token in name_to_token.items():
        # Try to extract the base timestamp_id part
        # Strategy: Match the start of the string
        nusc_base_names[s_name] = s_token
        
    for name in train_scenes:
        matched = False
        # 1. Try exact match
        if name in name_to_token:
            train_scenes_tokens.add(name_to_token[name])
            matched = True
        
        # 2. Try startswith match against all nusc scenes
        if not matched:
            print("it should match!! - GW")
            breakpoint()
            for s_name, s_token in name_to_token.items():
                if s_name.startswith(name):
                    train_scenes_tokens.add(s_token)
                    matched = True
        
        if matched:
            found_count += 1
        else:
             pass 
    
    train_scenes = train_scenes_tokens
    
    if not test:
        val_scenes_tokens = set()
        for name in val_scenes:
            matched = False
            if name in name_to_token:
                val_scenes_tokens.add(name_to_token[name])
                matched = True
            
            if not matched:
                for s_name, s_token in name_to_token.items():
                    if s_name.startswith(name):
                        val_scenes_tokens.add(s_token)
                        matched = True
            
            if not matched:
                 pass 
        val_scenes = val_scenes_tokens
        # val_scenes = {name_to_token[name] for name in val_scenes if name in name_to_token}

    # Check infrastructure matching for train/val/test scenes
    if nusc_infra is not None:
        infra_tokens = set(s['token'] for s in nusc_infra.sample)
        
        if test:
            test_samples = [s for s in nusc.sample if s['scene_token'] in train_scenes]
            test_tokens = set(s['token'] for s in test_samples)
            test_matched = len(test_tokens & infra_tokens)
            print(f"TEST: {test_matched}/{len(test_tokens)} samples have matching infra data")
        else:
            train_samples = [s for s in nusc.sample if s['scene_token'] in train_scenes]
            train_tokens = set(s['token'] for s in train_samples)
            train_matched = len(train_tokens & infra_tokens)
            print(f"TRAIN: {train_matched}/{len(train_tokens)} samples have matching infra data")
            
            val_samples = [s for s in nusc.sample if s['scene_token'] in val_scenes]
            val_tokens = set(s['token'] for s in val_samples)
            val_matched = len(val_tokens & infra_tokens)
            print(f"VAL: {val_matched}/{len(val_tokens)} samples have matching infra data")

    # breakpoint()
    # train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
    #     nusc, nusc_map_extractor, nusc_can_bus, train_scenes, val_scenes, test, max_sweeps=max_sweeps)
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_map_extractor, train_scenes, val_scenes, test, root_path=root_path, max_sweeps=max_sweeps,
        nusc_infra=nusc_infra, infra_root_path=infra_root_path)

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:  # 
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(out_path,
                                 '{}_infos_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)
    # import pdb; pdb.set_trace()
# 2023-04-04-14-27-53_44_0_000004
def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene: # token, log_token, nbr_samples, first_sample_token, last_sample_token, name, description
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token']) # first sample
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes

def process_single_sample(i):
    args_dict = G_args
    nusc = G_nusc
    nusc_infra = G_nusc_infra
    nusc_map_extractor = G_nusc_map_extractor
    predict_helper = G_predict_helper
    
    try:
        sample = nusc.sample[i]
        test = args_dict['test']
        train_scenes = args_dict['train_scenes']
        val_scenes = args_dict['val_scenes']
        
        # Filtering
        if test:
            if sample['scene_token'] not in train_scenes: 
                return None
        
        # Original logic inside loop
        map_location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location'] # e.g.'singapore-onenorth'
        # import pdb; pdb.set_trace()
        
        lidar_key = 'LIDAR_TOP'
        suffix = ""
        # Check for LIDAR_TOP with possible suffixes
        if 'LIDAR_TOP' not in sample['data']:
            found = False
            for k in sample['data']:
                if k.startswith('LIDAR_TOP'):
                    lidar_key = k
                    suffix = k[len('LIDAR_TOP'):]
                    found = True
                    break
            
            if not found:
                 print(f"Error processing sample {i} ({sample['token']}): Missing LIDAR_TOP in vehicle data. Keys: {list(sample['data'].keys())}")
                 return None

        lidar_token = sample['data'][lidar_key]
        sd_rec = nusc.get('sample_data', lidar_token)
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])

        lidar_path, boxes, _ = nusc.get_sample_data_for_cv(lidar_token, None)

        ####### NOTE 데이터 root 경로 넣어줘야 함.
        lidar_path = lidar_path.replace(args_dict['root_path'], 'datasets/v2xreal/data/')
        mmcv.check_file_exist(lidar_path)

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'scene_token': sample['scene_token'],
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'map_location': map_location,
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix # 3*3 matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # extract map annos
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(
            info["lidar2ego_rotation"]
        ).rotation_matrix
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(
            info["ego2global_rotation"]
        ).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])
        lidar2global = ego2global @ lidar2ego

        translation = list(lidar2global[:3, 3])
        rotation = list(Quaternion(matrix=lidar2global).q)
        
        map_geoms = nusc_map_extractor.get_map_geom(map_location, translation, rotation) 
        map_annos = geom2anno(map_geoms)
        info['map_annos'] = map_annos

        # obtain 6 image's information per frame
        base_camera_types = [
            'cam1', 
            'cam2',
            'cam3',
            'cam4',
        ]
        
        # Apply suffix detected from LIDAR key (e.g., _veh1)
        camera_types = [f"{cam}{suffix}" for cam in base_camera_types]

        # TODO: 기원 추가 ====================== ##        
        # keys = list(sample['data'].keys())
        # suffix = ""
        # for k in keys:
        #     if k.startswith("cam1_"):
        #         suffix = k[len("cam1"):]   # "_veh2"
        #         break

        # camera_types = [f"{cam}{suffix}" for cam in camera_types]
        # =================================== ##
        
        for base_cam, cam in zip(base_camera_types, camera_types):
            if cam not in sample['data']:
                continue
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_path = cam_path.replace(args_dict['root_path'], 'datasets/v2xreal/data/')
            
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, args_dict['root_path'], cam) 
            cam_info.update(cam_intrinsic=cam_intrinsic)
            # Store with base name (e.g., 'cam1') instead of full name (e.g., 'cam1_veh1')
            info['cams'].update({base_cam: cam_info})
        # import pdb; pdb.set_trace()
        # Infrastructure-side camera processing
        split_type = 'test' if args_dict['test'] else ('train' if sample['scene_token'] in args_dict['train_scenes'] else 'val')
        
        if nusc_infra is not None:
            # print("Available infra sample tokens:", [s['token'] for s in nusc_infra.sample])
            # import pdb; pdb.set_trace()
            
            # Try to get infrastructure sample with same token
            sample_infra = None
            try:
                sample_infra = nusc_infra.get('sample', sample['token']) # Assume same token
            except KeyError:
                # Token not found in infrastructure data
                scene_name = nusc.get('scene', sample['scene_token'])['name']
                # Note: G_infra_stats is not thread-safe, but sufficient for rough statistics
                print(f"[{split_type}] Infra sample NOT FOUND for token: {sample['token']}, scene: {scene_name}")
            
            if sample_infra:
                ## DEBUGING 용 기원 추가
                assert sample_infra['timestamp'] == sample['timestamp'], (sample['token'], sample['timestamp'], sample_infra['timestamp'])
                assert sample_infra['scene_token'] == sample['scene_token'], (sample['token'], sample['scene_token'], sample_infra['scene_token'])
                ## ================ ##
                
                print(f"[{split_type}] Infra sample MATCHED for token: {sample['token']}")
                
                infra_lidar_key = 'LIDAR_TOP'
                infra_suffix = ""

                if 'LIDAR_TOP' not in sample_infra['data']:
                    found_infra = False
                    for k in sample_infra['data']:
                        if k.startswith('LIDAR_TOP'):
                            infra_lidar_key = k
                            infra_suffix = k[len('LIDAR_TOP'):]
                            found_infra = True
                            break
                    
                    if not found_infra:
                        print(f"[{split_type}] Infra sample {sample['token']} matched but missing LIDAR_TOP*. Keys: {list(sample_infra['data'].keys())}")
                        sample_infra = None

            if sample_infra:
                # Get infrastructure's own ego pose
                infra_lidar_token = sample_infra['data'][infra_lidar_key]
                infra_sd_rec = nusc_infra.get('sample_data', infra_lidar_token)
                infra_cs_record = nusc_infra.get('calibrated_sensor', infra_sd_rec['calibrated_sensor_token'])
                infra_pose_record = nusc_infra.get('ego_pose', infra_sd_rec['ego_pose_token'])
                
                infra_l2e_t = np.array(infra_cs_record['translation'])
                infra_l2e_r = infra_cs_record['rotation']
                infra_e2g_t = np.array(infra_pose_record['translation'])
                infra_e2g_r = infra_pose_record['rotation']
                
                infra_l2e_r_mat = Quaternion(infra_l2e_r).rotation_matrix
                infra_e2g_r_mat = Quaternion(infra_e2g_r).rotation_matrix
                
                base_infra_camera_types = ['cam1', 'cam2', 'cam3', 'cam4']
                
                # Apply detected suffix from LIDAR key (e.g., _inf1)
                infra_camera_types = [f"{cam}{infra_suffix}" for cam in base_infra_camera_types]
                
                # TODO: 기원 추가 ====================== ##
                # keys = list(sample_infra['data'].keys())
                # suffix = ""
                # for k in keys:
                #     if k.startswith("cam1_"):
                #         suffix = k[len("cam1"):]   # 예: "_veh2"
                #         break
                # infra_camera_types = [f"{cam}{suffix}" for cam in infra_camera_types]
                ## =================================== ##
                
                
                for base_cam, i_cam in zip(base_infra_camera_types, infra_camera_types):
                    if i_cam in sample_infra['data']:
                        cam_token = sample_infra['data'][i_cam]
                        _, _, cam_intrinsic = nusc_infra.get_sample_data(cam_token)
                        
                        # Use infrastructure's own transformation
                        # cam_info = obtain_sensor2top(nusc_infra, cam_token, infra_l2e_t, infra_l2e_r_mat,
                        #                              infra_e2g_t, infra_e2g_r_mat, args_dict['infra_root_path'], i_cam,
                        #                              target_path_prefix='datasets/v2xreal/data/')
                        
                        # Change to Vehicle-side LiDAR coordinate system
                        cam_info = obtain_sensor2top(nusc_infra, cam_token, l2e_t, l2e_r_mat,
                                                     e2g_t, e2g_r_mat, args_dict['infra_root_path'], i_cam,
                                                     target_path_prefix='datasets/v2xreal/data/')
                        
                        R_s2l_T = cam_info['sensor2lidar_rotation'] 
                        T_s2l = cam_info['sensor2lidar_translation']
                        
                        mat_s2l = np.eye(4)
                        mat_s2l[:3, :3] = R_s2l_T.T 
                        mat_s2l[:3, 3] = T_s2l
                        
                        # Use infrastructure's lidar2ego
                        # mat_l2e = np.eye(4)
                        # mat_l2e[:3, :3] = infra_l2e_r_mat
                        # mat_l2e[:3, 3] = infra_l2e_t
                        
                        # mat_s2e = mat_l2e @ mat_s2l # Infra Cam -> Infra Lidar -> Infra Ego ?? No. This logic was confusing if s2l was already relative to Infra Lidar.
                        
                        # Instead, since we are doing VEHICLE-centric.
                        # cam_info['sensor2lidar'] is now (InfraCam -> VehicleLidar)
                        
                        # We should update sensor2ego to be relative to VEHICLE Ego?
                        # Or keep it relative to INFRA Ego (which is physically true)?
                        # Usually, downstream wants sensor2lidar (to vehicle) and sensor2ego (to vehicle ego).
                        
                        # Let's compute InfraCam -> VehicleEgo
                        # InfraCam -> VehicleLidar -> VehicleEgo
                        
                        mat_vl2ve = np.eye(4)
                        mat_vl2ve[:3, :3] = l2e_r_mat
                        mat_vl2ve[:3, 3] = l2e_t
                        
                        mat_s2ve = mat_vl2ve @ mat_s2l
                        
                        new_s2e_t = mat_s2ve[:3, 3]
                        new_s2e_r_mat = mat_s2ve[:3, :3]
                        new_s2e_r_quat = rot_to_quat_wxyz(new_s2e_r_mat)
                        
                        cam_info['sensor2ego_translation'] = new_s2e_t
                        cam_info['sensor2ego_rotation'] = new_s2e_r_quat
                        
                        # Use infrastructure's ego2global -> NO, use VEHICLE's ego2global?
                        # If sensor2ego is relative to VehicleEgo, then ego2global must be relative to VehicleEgo.
                        cam_info['ego2global_translation'] = e2g_t
                        cam_info['ego2global_rotation'] = rot_to_quat_wxyz(e2g_r_mat)
                        
                        cam_info['cam_intrinsic'] = cam_intrinsic
                        
                        # Store with base name prefixed with 'infrastructure_' (e.g. 'infrastructure_cam1')
                        info['cams'].update({f'infrastructure_{base_cam}': cam_info})
        
        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data'][lidar_key])
        sweeps = []
        while len(sweeps) < args_dict['max_sweeps']:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, args_dict['root_path'], 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps

        # obtain annotation
        if True:
            annotations = [ 
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            
            locs = np.array([b.center for b in boxes]).reshape(-1, 3) 
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3) 
            rots = np.array([b.orientation.yaw_pitch_roll[0] 
                             for b in boxes]).reshape(-1, 1)
            
            velocity = np.array([nusc.box_velocity(token)[:2] for token in sample['anns']])
            
            for k in range(len(boxes)):
                velo = np.array([*velocity[k], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T 
                velocity[k] = velo[:2]
            
            names = [b.name for b in boxes]
            for k in range(len(names)):
                if names[k] in NameMapping:
                    names[k] = NameMapping[names[k]]
            names = np.array(names)
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations], 
                dtype=bool).reshape(-1)

            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)

            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            
            instance_inds = [nusc.getind('instance', anno['instance_token'])
                             for anno in annotations]

            num_box = len(boxes)
            fut_ts = args_dict['fut_ts']
            gt_fut_trajs = np.zeros((num_box, fut_ts, 2))
            gt_fut_masks = np.zeros((num_box, fut_ts))

            for k, anno in enumerate(annotations):
                instance_token = anno['instance_token']
                fut_traj_local = predict_helper.get_future_for_agent(   
                    instance_token, 
                    sample['token'], 
                    seconds=fut_ts/2,  
                    in_agent_frame=True
                )
                if fut_traj_local.shape[0] > 0:
                    box = boxes[k]
                    trans = box.center
                    rot = Quaternion(matrix=box.rotation_matrix)
                    fut_traj_scene = convert_local_coords_to_global(fut_traj_local, trans, rot)
                    valid_step = fut_traj_scene.shape[0]
                    gt_fut_trajs[k, 0] = fut_traj_scene[0] - box.center[:2]
                    gt_fut_trajs[k, 1:valid_step] = fut_traj_scene[1:] - fut_traj_scene[:-1] 
                    gt_fut_masks[k, :valid_step] = 1 

            ego_fut_ts = args_dict['ego_fut_ts']
            ego_fut_trajs = np.zeros((ego_fut_ts + 1, 3))
            ego_fut_masks = np.zeros((ego_fut_ts + 1))
            sample_cur = sample

            ego_status = get_ego_status_no_canbus(nusc, sample, default_hz=10.0) 

            for k in range(ego_fut_ts + 1):
                pose_mat = get_global_sensor_pose(sample_cur, nusc)
                ego_fut_trajs[k] = pose_mat[:3, 3]
                ego_fut_masks[k] = 1
                if sample_cur['next'] == '':
                    ego_fut_trajs[k+1:] = ego_fut_trajs[k]
                    break
                else:
                    sample_cur = nusc.get('sample', sample_cur['next'])
            
            ego_fut_trajs = ego_fut_trajs - np.array(pose_record['translation'])
            rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
            ego_fut_trajs = ego_fut_trajs - np.array(cs_record['translation'])
            rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T

            theta_rad = np.arctan2(2.9, 41.8759)  # 약 0.0691 rad
            q = Quaternion(axis=[0, 0, 1], angle=theta_rad)
            rot_matrix = q.rotation_matrix
            correct_ego_fut_trajs = (rot_matrix @ ego_fut_trajs.T).T

            if correct_ego_fut_trajs[-1][1] <= -4:  
                command = np.array([1, 0, 0])  # Turn Right
            elif correct_ego_fut_trajs[-1][1] >= 4:
                command = np.array([0, 1, 0])  # Turn Left
            else:
                command = np.array([0, 0, 1])  # Go Straight

            ego_fut_trajs = ego_fut_trajs[1:] - ego_fut_trajs[:-1]  
            
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag
            info['instance_inds'] = instance_inds
            info['gt_agent_fut_trajs'] = gt_fut_trajs.astype(np.float32)
            info['gt_agent_fut_masks'] = gt_fut_masks.astype(np.float32)
            info['gt_ego_fut_trajs'] = ego_fut_trajs[:, :2].astype(np.float32)
            info['gt_ego_fut_masks'] = ego_fut_masks[1:].astype(np.float32)
            info['gt_ego_fut_cmd'] = command.astype(np.float32)

            info['ego_status'] = ego_status

        if sample['scene_token'] in train_scenes:
            return ('train', info)
        elif sample['scene_token'] in val_scenes:
            return ('val', info)
        else:
             return None # Should be filtered out already, but just in case

    except Exception as e:
        print(f"Error processing sample {i} ({nusc.sample[i]['token']}): {e}")
        traceback.print_exc()
        return None

def _fill_trainval_infos(nusc,
                         nusc_map_extractor,
                        #  nusc_can_bus,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10,
                         fut_ts=12 * 5,
                         root_path = 'datasets/v2xreal/data',
                         ego_fut_ts=12 * 5,
                         nusc_infra=None,
                         infra_root_path='datasets/v2xreal/infrastructure-side/'):  
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    
    # Prepare arguments for globals
    
    # Filter samples to process based on selected scenes
    target_scenes = set(train_scenes) if test else (set(train_scenes) | set(val_scenes))
    sample_indices = []
    print(f"Filtering samples for {len(target_scenes)} scenes...")
    
    for i, sample in enumerate(nusc.sample):
        if sample['scene_token'] in target_scenes:
            sample_indices.append(i)            # 260217 NOTE: 현재 target_scene이 잘못 구성되어 있음. 때문에 sample_indices에는 일부만 들어가게 됨.
    
    # Randomly select 2% of samples if not test
    # if len(sample_indices) > 0:
    #     import random
    #     target_len = int(len(sample_indices) * 0.02)
    #     sample_indices = sample_indices[:target_len]
    #     print(f"Sampled 2% of data: {len(sample_indices)} samples")
            
    print(f"Total samples to process: {len(sample_indices)} (out of {len(nusc.sample)})")

    global_args = {
        'train_scenes': set(train_scenes),
        'val_scenes': set(val_scenes),
        'test': test,
        'max_sweeps': max_sweeps,
        'fut_ts': fut_ts,
        'ego_fut_ts': ego_fut_ts,
        'root_path': root_path,
        'infra_root_path': infra_root_path
    }

    # Parallel processing
    print(f"Starting parallel processing with {args.workers} workers...")
    
    results = mmcv.track_parallel_progress(
        process_single_sample,
        sample_indices,
        nproc=args.workers,
        initializer=init_globals,
        initargs=(nusc, nusc_infra, nusc_map_extractor, global_args)
    )

    # DEBUG: Serial processing for pdb
    # init_globals(nusc, nusc_infra, nusc_map_extractor, global_args)
    # results = []
    # print(f"DEBUG: Processing samples serially...")
    # for i in mmcv.track_iter_progress(sample_indices):
    #     results.append(process_single_sample(i))

    # DEBUG: Serial processing for pdb
    # init_globals(nusc, nusc_infra, nusc_map_extractor, global_args)
    # results = []
    # print(f"DEBUG: Processing first 10 samples serially out of {len(nusc.sample)}...")
    # for i in mmcv.track_iter_progress(range(min(10, len(nusc.sample)))):
    #     results.append(process_single_sample(i))

    train_nusc_infos = []
    val_nusc_infos = []

    for res in results:
        if res is None:
            continue
        tag, info = res
        if tag == 'train':
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos

# def get_ego_status(nusc, nusc_can_bus, sample):
#     ego_status = []
#     ref_scene = nusc.get("scene", sample['scene_token'])
#     try:
#         pose_msgs = nusc_can_bus.get_messages(ref_scene['name'],'pose') # accel, orientation, pose, rotation_rate, utime 등의 정보
#         steer_msgs = nusc_can_bus.get_messages(ref_scene['name'], 'steeranglefeedback') # utime & value
#         pose_uts = [msg['utime'] for msg in pose_msgs]
#         steer_uts = [msg['utime'] for msg in steer_msgs]
#         ref_utime = sample['timestamp']
#         pose_index = locate_message(pose_uts, ref_utime) # reference time과 가장 가까운 msg를 찾으려고 index 추출
#         pose_data = pose_msgs[pose_index]
#         steer_index = locate_message(steer_uts, ref_utime)
#         steer_data = steer_msgs[steer_index]
#         ego_status.extend(pose_data["accel"]) # acceleration in ego vehicle frame, m/s/s (x,y,z)
#         ego_status.extend(pose_data["rotation_rate"]) # angular velocity in ego vehicle frame, rad/s (roll, pitch, yaw)
#         ego_status.extend(pose_data["vel"]) # velocity in ego vehicle frame, m/s
#         ego_status.append(steer_data["value"]) # steering angle, positive: left turn, negative: right turn
#     except:
#         ego_status = [0] * 10
    
#     return np.array(ego_status).astype(np.float32)

def _sample_time_sec(nusc, sample_token, default_hz=10.0, time_type="ms"):
    """Return sample time in seconds. If sample['timestamp'] looks like frame index, use default_hz."""
    s = nusc.get('sample', sample_token)
    ts = float(s.get('timestamp', 0.0))
    # breakpoint()
    if ts > 1e9:  # likely microseconds
        return ts * 1e-6
    if time_type == 'ms':
        return ts / 1e6
    elif time_type == 's':
        return ts
    else:
        print(f'treat frame index as timestamp at {default_hz}Hz')
        return ts * (1.0 / default_hz)
    # else: treat as frame index
    # return ts * (1.0 / default_hz)

def _ego_pose_q_t(nusc, sample_token):
    """Get ego pose (ego->global) Quaternion and translation from LIDAR_TOP sample_data pose."""
    s = nusc.get('sample', sample_token)
    
    lidar_key = 'LIDAR_TOP'
    if 'LIDAR_TOP' not in s['data']:
        for k in s['data']:
            if k.startswith('LIDAR_TOP'):
                lidar_key = k
                break

    lidar_sd = nusc.get('sample_data', s['data'][lidar_key])
    pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    t = np.array(pose['translation'], dtype=np.float64)
    q = Quaternion(pose['rotation'])  # wxyz
    return q, t

def _yaw_from_q(q: Quaternion) -> float:
    R = q.rotation_matrix
    return float(np.arctan2(R[1, 0], R[0, 0]))

def get_ego_status_no_canbus(nusc, sample, default_hz=10.0, max_dt=1.5, time_type="ms"):
    """
    Replacement for CAN-based ego status.
    Returns 10-dim: [ax,ay,az, wx,wy,wz, vx,vy,vz, steer]
    accel/rotation_rate/vel are in ego vehicle frame.
    steer is set to 0 (no CAN).
    """
    # breakpoint()
    cur_tok = sample['token']
    prev_tok = sample['prev'] if sample['prev'] != '' else None
    next_tok = sample['next'] if sample['next'] != '' else None

    # Need at least one neighbor to estimate motion
    if prev_tok is None and next_tok is None:
        return np.zeros(10, dtype=np.float32)

    # Pick finite-diff endpoints (centered if possible)
    if prev_tok is None:
        tok0, tok1 = cur_tok, next_tok
        allowed = max_dt
    elif next_tok is None:
        tok0, tok1 = prev_tok, cur_tok
        allowed = max_dt
    else:
        tok0, tok1 = prev_tok, next_tok
        allowed = max_dt * 2.0

    t0_sec = _sample_time_sec(nusc, tok0, default_hz=default_hz, time_type=time_type)  # time type -> ms, s, index 중 하나 선택
    t1_sec = _sample_time_sec(nusc, tok1, default_hz=default_hz, time_type=time_type)
    dt = t1_sec - t0_sec
   

    if dt <= 1e-6 or dt > allowed:
        return np.zeros(10, dtype=np.float32)

    q0, p0 = _ego_pose_q_t(nusc, tok0)
    q1, p1 = _ego_pose_q_t(nusc, tok1)

    # --- velocity in global ---
    v_g = (p1 - p0) / dt  # (3,)

    # --- convert to ego frame at CURRENT time ---
    q_cur, _ = _ego_pose_q_t(nusc, cur_tok)
    R_ge = q_cur.inverse.rotation_matrix  # global->ego
    v_e = R_ge @ v_g

    # --- yaw rate (approx) ---
    yaw0 = _yaw_from_q(q0)
    yaw1 = _yaw_from_q(q1)
    # unwrap yaw difference into [-pi, pi]
    dyaw = (yaw1 - yaw0 + np.pi) % (2*np.pi) - np.pi
    wz = dyaw / dt

    # without CAN, roll/pitch rates are hard; set to 0 (or compute from full quaternion delta if needed)
    w_e = np.array([0.0, 0.0, wz], dtype=np.float64)

    # --- acceleration: use second difference if possible, else 0 ---
    # If we have both prev and next, we can estimate accel at current using one-sided vel estimates
    a_e = np.zeros(3, dtype=np.float64)
    # breakpoint()
    if prev_tok is not None and next_tok is not None:
        # v at cur from prev->cur and cur->next
        t_prev = _sample_time_sec(nusc, prev_tok, default_hz=default_hz, time_type=time_type)
        t_cur  = _sample_time_sec(nusc, cur_tok,  default_hz=default_hz, time_type=time_type)
        t_next = _sample_time_sec(nusc, next_tok, default_hz=default_hz, time_type=time_type)
        dt0 = t_cur - t_prev
        dt1 = t_next - t_cur

        if dt0 > 1e-6 and dt1 > 1e-6 and dt0 <= max_dt and dt1 <= max_dt:
            q_prev, p_prev = _ego_pose_q_t(nusc, prev_tok)
            q_next, p_next = _ego_pose_q_t(nusc, next_tok)

            v0_g = (p_cur := _ego_pose_q_t(nusc, cur_tok)[1]) - p_prev
            v0_g = v0_g / dt0
            v1_g = (p_next - p_cur) / dt1

            v0_e = R_ge @ v0_g
            v1_e = R_ge @ v1_g
            a_e = (v1_e - v0_e) / ((dt0 + dt1) * 0.5)

    steer = 0.0  # no CAN -> unknown

    ego_status = np.concatenate([a_e, w_e, v_e, np.array([steer])], axis=0)
    return ego_status.astype(np.float32)


def get_global_sensor_pose(rec, nusc):
    lidar_key = 'LIDAR_TOP'
    if 'LIDAR_TOP' not in rec['data']:
        for k in rec['data']:
            if k.startswith('LIDAR_TOP'):
                lidar_key = k
                break
                
    lidar_sample_data = nusc.get('sample_data', rec['data'][lidar_key])

    pose_record = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    cs_record = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])

    ego2global = transform_matrix(pose_record["translation"], Quaternion(pose_record["rotation"]), inverse=False)
    sensor2ego = transform_matrix(cs_record["translation"], Quaternion(cs_record["rotation"]), inverse=False)
    pose = ego2global.dot(sensor2ego)

    return pose

def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      root_path,
                      sensor_type='lidar',
                      target_path_prefix='datasets/v2xreal/data/'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    # data_path = data_path.replace(root_path, './')
    data_path = data_path.replace(root_path, target_path_prefix)
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        # 'sensor2ego_rotation': cs_record['rotation'],
        'sensor2ego_rotation': rot_to_quat_wxyz(cs_record['rotation']),   # hh
        'ego2global_translation': pose_record['translation'],
        # 'ego2global_rotation': pose_record['rotation'],
        'ego2global_rotation': rot_to_quat_wxyz(pose_record['rotation']), # hh
        'timestamp': sd_rec['timestamp']
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
     
    
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    
    
    return sweep

def nuscenes_data_prep(root_path,
                       can_bus_root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10,
                       infra_root_path='datasets/v2xreal/infrastructure-side/'):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    create_nuscenes_infos(
        root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps, infra_root_path=infra_root_path)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--canbus',
    type=str,
    default='./data',
    help='specify the root path of nuScenes canbus')
parser.add_argument(
    '--infra-root-path',
    type=str,
    default='datasets/v2xreal/infrastructure-side/',
    help='specify the root path of infrastructure dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required='False',
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=128, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':

    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        # if args.version == 'v1.0-debug':
        # train_version = f'{args.version}-trainval_debug'
        # else:
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            infra_root_path=args.infra_root_path)

        test_version = f'{args.version}-test'
        # test_version = f'{args.version}-trainval_debug'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            infra_root_path=args.infra_root_path)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            infra_root_path=args.infra_root_path)
